#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <bits/stdc++.h> 

#include "llama.h"
#include "common.h"
#include "actions.hpp"

int main(int argc, char ** argv) {
  if (argc < 2) {
    std::cerr << "model path is missing\n";
    return 1;
  }

  log_disable();
  llama_backend_init();
  app_t app;
  json res;
  json req;

  req = json{
    {"model_path", std::string(argv[1])},
    {"seed", 42},
    {"n_ctx", 2048},
    {"n_threads", 4},
  };
  res = action_load(app, req);
  std::cout << "action_load: " << res << "\n";

  req = json{};
  res = action_sampling_init(app, req);

  std::string text = "The sky is blue. The sun is yellow. Here we go. There and back again. ";
  req = json{
    {"text", text + text + text + text + text},
  };
  res = action_tokenize(app, req);
  std::cout << "action_tokenize: " << res << "\n";

  std::vector<int> tokens_to_eval = res["tokens"];
  req = json{
    {"tokens", tokens_to_eval},
  };
  res = action_eval(app, req);
  std::cout << "action_eval: " << res << "\n";

  time_t start, end; 
  time(&start);
  ccama_save_session_file_quant(app.ctx, "./state.bin");
  time(&end);
  double time_taken = double(end - start); 
  std::cout << "Time taken : " << time_taken << " sec";

  // clean up
  llama_free(app.ctx);
  llama_free_model(app.model);
  llama_sampling_free(app.ctx_sampling);
  llama_backend_free();

  return 0;
}
