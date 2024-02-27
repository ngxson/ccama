/////////////////////////////////////////////////////////////////////////////////////
// This file will be concatenated to llama.cpp in order to access all internal APIs
/////////////////////////////////////////////////////////////////////////////////////

static size_t ccama_fp16_to_q8(ggml_tensor * tensor, uint8_t * input_fp16, uint8_t * output_q8, size_t nelements) {
    std::vector<float> tmp_buf_32;
    std::array<int64_t, 1 << 4> hist_cur = {};
    const int n_per_row = nelements;
    const int nrows = 1; //nelements / n_per_row;
    // convert fp16 => fp32
    tmp_buf_32.resize(nelements);
    //printf("nelements %ld n_per_row %d nrows %d\n", nelements, n_per_row, nrows);
    //printf("Converted fp16 => fp32\n");
    ggml_fp16_to_fp32_row((ggml_fp16_t *)input_fp16, tmp_buf_32.data(), nelements);
    // quantize to GGML_TYPE_Q8_0
    //printf("quantize to GGML_TYPE_Q8_0\n");
    size_t new_size = ggml_quantize_chunk(GGML_TYPE_Q8_0, tmp_buf_32.data(), output_q8, 0, nrows, n_per_row, hist_cur.data(), nullptr);
    //size_t new_size = ggml_quantize_q4_0(tmp_buf_32.data(), output_q8, nrows, n_per_row, hist_cur.data());
    //printf("quantize ok, new_size %ld\n", new_size);
    return new_size;
}

static void ccama_copy_state_quant(struct llama_context * ctx, llama_data_context * data_ctx) {
    // copy kv cache
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
            tmp_buf_q.resize(v_row_size);
            for (int ir = 0; ir < (int) n_embd_v_gqa; ++ir) {
                ggml_backend_tensor_get(kv_self.v_l[il], tmp_buf.data(), ir*v_row_stride, tmp_buf.size());

                const size_t nelem = v_row_size / sizeof(ggml_fp16_t);
                new_size = ccama_fp16_to_q8(kv_self.v_l[il], tmp_buf.data(), tmp_buf_q.data(), nelem);
                tmp_buf_q.resize(new_size);

                data_ctx->write(tmp_buf_q.data(), tmp_buf_q.size());
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
