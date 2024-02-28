/////////////////////////////////////////////////////////////////////////////////////
// This file will be concatenated to llama.h in order to access all internal APIs
/////////////////////////////////////////////////////////////////////////////////////

LLAMA_API bool ccama_save_session_file_quant(struct llama_context * ctx, const char * path_session);
LLAMA_API size_t cama_copy_state_data_quant(struct llama_context * ctx, uint8_t * dst);

LLAMA_API bool ccama_load_session_file_quant(struct llama_context * ctx, const char * path_session);
LLAMA_API size_t ccama_set_state_data_quant(struct llama_context * ctx, uint8_t * src);
