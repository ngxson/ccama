/////////////////////////////////////////////////////////////////////////////////////
// This file will be concatenated to llama.cpp in order to access all internal APIs
/////////////////////////////////////////////////////////////////////////////////////


#include "ggml-quants.h"
#define CCAMA_SAVE_QUANT GGML_TYPE_Q4_K

static size_t ccama_get_size_quant(size_t nelem) {
    assert(nelem % QK_K == 0); // for K-quant
    return (nelem/QK_K*sizeof(block_q4_K));
}

// Save state FP16 to quantized Q8

static size_t ccama_fp16_to_q8(ggml_tensor * tensor, uint8_t * input_fp16, uint8_t * output_q8, size_t nelements) {
    std::vector<float> tmp_buf_32;
    if (tensor->type != GGML_TYPE_F16) {
        throw std::runtime_error("Require tensor to be GGML_TYPE_F16");
    }
    // convert fp16 => fp32
    tmp_buf_32.resize(nelements);
    ggml_fp16_to_fp32_row((ggml_fp16_t *)input_fp16, tmp_buf_32.data(), nelements);
    /*if (fff0 < 4) {
        printf("nelements %ld, tmp_buf_32 %ld\n", nelements, tmp_buf_32.size());
        for (int i = 0; i < 16; i++) {
            printf("%f ", tmp_buf_32[i]);
            fflush(stdout);
        }
        printf("\n");
        fff0++;
    }*/
    // quantize to GGML_TYPE_Q8_0
    ggml_type_traits_t qfns = ggml_internal_get_type_traits(CCAMA_SAVE_QUANT);
    qfns.from_float(tmp_buf_32.data(), output_q8, nelements);
    return ccama_get_size_quant(nelements);
}

static void ccama_copy_state_quant(struct llama_context * ctx, llama_data_context * data_ctx) {
    // We only save KV cache for now
    const auto & kv_self = ctx->kv_self;
    const auto & hparams = ctx->model.hparams;
    const auto & cparams = ctx->cparams;

    const uint32_t n_layer      = hparams.n_layer;
    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();
    const uint32_t n_ctx        = cparams.n_ctx;

    const size_t   kv_buf_size = kv_self.total_size();
    const uint32_t kv_head     = kv_self.head;
    const uint32_t kv_size     = kv_self.size;
    const uint32_t kv_used     = kv_self.used;

    data_ctx->write(&kv_buf_size, sizeof(kv_buf_size));
    data_ctx->write(&kv_head,     sizeof(kv_head));
    data_ctx->write(&kv_size,     sizeof(kv_size));
    data_ctx->write(&kv_used,     sizeof(kv_used));

    printf("kv_buf_size %ld, kv_head %d, kv_size %d, kv_used %d\n", kv_buf_size, kv_head, kv_size, kv_used);

    if (kv_buf_size) {
        std::vector<uint8_t> tmp_buf;
        std::vector<uint8_t> tmp_buf_q;
        size_t new_size;
        for (int il = 0; il < (int) n_layer; ++il) {
            const size_t k_size = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*kv_head);

            tmp_buf.resize(k_size);
            tmp_buf_q.resize(k_size);
            const size_t nelem = k_size / sizeof(ggml_fp16_t);
            ggml_backend_tensor_get(kv_self.k_l[il], tmp_buf.data(), 0, tmp_buf.size());
            new_size = ccama_fp16_to_q8(kv_self.k_l[il], tmp_buf.data(), tmp_buf_q.data(), nelem);
            tmp_buf_q.resize(new_size);
            data_ctx->write(tmp_buf_q.data(), tmp_buf_q.size());

            // v is not contiguous, copy row by row
            const size_t v_row_size   = ggml_row_size(kv_self.v_l[il]->type, kv_head);
            const size_t v_row_stride = ggml_row_size(kv_self.v_l[il]->type, n_ctx);

            tmp_buf.resize(v_row_size);
            for (int ir = 0; ir < (int) n_embd_v_gqa; ++ir) {
                ggml_backend_tensor_get(kv_self.v_l[il], tmp_buf.data(), ir*v_row_stride, tmp_buf.size());

                // no quant for V
                data_ctx->write(tmp_buf.data(), tmp_buf.size());
            }
        }
    }

    for (uint32_t i = 0; i < kv_size; ++i) {
        const auto & cell = kv_self.cells[i];

        const llama_pos pos         = cell.pos;
        const size_t    seq_id_size = cell.seq_id.size();

        data_ctx->write(&pos,         sizeof(pos));
        data_ctx->write(&seq_id_size, sizeof(seq_id_size));

        for (auto seq_id : cell.seq_id) {
            data_ctx->write(&seq_id, sizeof(seq_id));
        }
    }
}

bool ccama_save_session_file_quant(struct llama_context * ctx, const char * path_session) {
    llama_file file(path_session, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    file.write_raw(&ctx->model.hparams, sizeof(llama_hparams));

    // save the context state using stream saving
    llama_data_file_context data_ctx(&file);
    ccama_copy_state_quant(ctx, &data_ctx);

    return true;
}

size_t cama_copy_state_data_quant(struct llama_context * ctx, uint8_t * dst) {
    llama_data_buffer_context data_ctx(dst);
    ccama_copy_state_quant(ctx, &data_ctx);

    return data_ctx.get_size_written();
}



/////////////////////////////////////////////////////////////////////////////////////


// Load state from quantized Q8 to FP16

static size_t ccama_q8_to_fp16(ggml_tensor * tensor, uint8_t * input_q8, uint8_t * output_fp16, size_t nelements) {
    std::vector<float> tmp_buf_32(nelements);
    if (tensor->type != GGML_TYPE_F16) {
        throw std::runtime_error("Require tensor to be GGML_TYPE_F16");
    }
    // dequantize it
    ggml_type_traits_t qfns = ggml_internal_get_type_traits(CCAMA_SAVE_QUANT);
    qfns.to_float(input_q8, tmp_buf_32.data(), nelements);
    // convert to fp32 ==> fp16
    ggml_fp32_to_fp16_row(tmp_buf_32.data(), (ggml_fp16_t *) output_fp16, nelements);
    return nelements * sizeof(float);
}

size_t ccama_set_state_data_quant(struct llama_context * ctx, uint8_t * src) {
    uint8_t * inp = src;

    // We only save KV cache for now
    const auto & kv_self = ctx->kv_self;
    const auto & hparams = ctx->model.hparams;
    const auto & cparams = ctx->cparams;

    const uint32_t n_layer      = hparams.n_layer;
    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();
    const uint32_t n_ctx        = cparams.n_ctx;

    size_t   kv_buf_size;
    uint32_t kv_head;
    uint32_t kv_size;
    uint32_t kv_used;

    memcpy(&kv_buf_size, inp, sizeof(kv_buf_size)); inp += sizeof(kv_buf_size);
    memcpy(&kv_head,     inp, sizeof(kv_head));     inp += sizeof(kv_head);
    memcpy(&kv_size,     inp, sizeof(kv_size));     inp += sizeof(kv_size);
    memcpy(&kv_used,     inp, sizeof(kv_used));     inp += sizeof(kv_used);

    printf("kv_buf_size %ld, kv_head %d, kv_size %d, kv_used %d\n", kv_buf_size, kv_head, kv_size, kv_used);

    std::vector<uint8_t> tmp_buf;
    size_t nelem;
    if (kv_buf_size) {
        GGML_ASSERT(kv_self.total_size() == kv_buf_size);

        for (int il = 0; il < (int) n_layer; ++il) {
            const size_t k_size = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*kv_head);
            nelem = k_size / sizeof(ggml_fp16_t);

            tmp_buf.resize(k_size);
            ccama_q8_to_fp16(kv_self.k_l[il], inp, tmp_buf.data(), nelem);
            ggml_backend_tensor_set(kv_self.k_l[il], tmp_buf.data(), 0, k_size);
            inp += ccama_get_size_quant(nelem);

            // v is not contiguous, copy row by row
            const size_t v_row_size   = ggml_row_size(kv_self.v_l[il]->type, kv_head);
            const size_t v_row_stride = ggml_row_size(kv_self.v_l[il]->type, n_ctx);

            for (int ir = 0; ir < (int) n_embd_v_gqa; ++ir) {
                // no quant for V
                ggml_backend_tensor_set(kv_self.v_l[il], inp, ir*v_row_stride, v_row_size);
                inp += v_row_size; //nelem * sizeof(ggml_fp16_t);
            }
        }
    }

    ctx->kv_self.head = kv_head;
    ctx->kv_self.size = kv_size;
    ctx->kv_self.used = kv_used;

    ctx->kv_self.cells.resize(kv_size);

    for (uint32_t i = 0; i < kv_size; ++i) {
        llama_pos pos;
        size_t    seq_id_size;

        memcpy(&pos,         inp, sizeof(pos));         inp += sizeof(pos);
        memcpy(&seq_id_size, inp, sizeof(seq_id_size)); inp += sizeof(seq_id_size);

        ctx->kv_self.cells[i].pos = pos;

        llama_seq_id seq_id;

        for (size_t j = 0; j < seq_id_size; ++j) {
            memcpy(&seq_id, inp, sizeof(seq_id)); inp += sizeof(seq_id);
            ctx->kv_self.cells[i].seq_id.insert(seq_id);
        }
    }

    return inp - src;
}

bool ccama_load_session_file_quant(struct llama_context * ctx, const char * path_session) {
    try {
        llama_file file(path_session, "rb");
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }

        llama_hparams session_hparams;
        file.read_raw(&session_hparams, sizeof(llama_hparams));

        if (session_hparams != ctx->model.hparams) {
            LLAMA_LOG_INFO("%s : model hparams didn't match from session file!\n", __func__);
            return false;
        }

        const size_t n_state_size_cur = file.size - file.tell();
        std::vector<uint8_t> state_data(n_state_size_cur);
        file.read_raw(state_data.data(), n_state_size_cur);

        ccama_set_state_data_quant(ctx, state_data.data());
        return true;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading session file: %s\n", err.what());
        return false;
    }
}
