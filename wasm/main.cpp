#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>

#include "llama.h"
#include "json.hpp"
#include "common.h"
#include "actions.hpp"

#define WLLAMA_ACTION(name)              \
  if (action == #name)                   \
  {                                      \
    res = action_##name(app, body_json); \
  }

static std::string result;
static app_t app;

extern "C" int wllama_start()
{
  log_disable();
  llama_backend_init();
  return 0;
}

extern "C" const char *wllama_action(const char *name, const char *body)
{
  try
  {
    json res;
    std::string body_str(body);
    json body_json = json::parse(body_str);
    std::string action(name);
    WLLAMA_ACTION(load);
    WLLAMA_ACTION(sampling_init);
    WLLAMA_ACTION(lookup_token);
    WLLAMA_ACTION(tokenize);
    WLLAMA_ACTION(detokenize);
    WLLAMA_ACTION(eval);
    WLLAMA_ACTION(decode_logits);
    WLLAMA_ACTION(sampling_accept);
    WLLAMA_ACTION(kv_remove);
    WLLAMA_ACTION(current_status);
    // WLLAMA_ACTION(session_save);
    // WLLAMA_ACTION(session_load);
    result = std::string(res.dump());
    return result.c_str();
  }
  catch (std::exception &e)
  {
    result = std::string(e.what());
    return result.c_str();
  }
}

extern "C" void wllama_exit()
{
  llama_free(app.ctx);
  llama_free_model(app.model);
  llama_sampling_free(app.ctx_sampling);
  llama_backend_free();
}

extern "C" const char *wllama_decode_exception(int exception_ptr)
{
  return reinterpret_cast<std::exception *>(exception_ptr)->what();
}

int main()
{
  std::cerr << "Unused\n";
  return 0;
}
