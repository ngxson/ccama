#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <chrono>
#include <bits/stdc++.h> 

#include "llama.h"
#include "common.h"
#include "actions.hpp"

#define TIMED std::chrono::high_resolution_clock::now()
#define CALC_TIMED std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count()

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
  auto tstart = TIMED;
  auto tend = TIMED;
  double time_taken;
  size_t output_size;

  req = json{
    {"model_path", std::string(argv[1])},
    {"seed", 42},
    {"n_ctx", 2048},
    {"n_threads", 4},
    {"embeddings", true}, // enable embeddings
    {"pooling_type", "LLAMA_POOLING_TYPE_MEAN"},
  };
  res = action_load(app, req);
  std::cout << "action_load: " << res << "\n";
  int token_bos = res["token_bos"];
  int token_eos = res["token_eos"];

  req = json{};
  res = action_sampling_init(app, req);

  std::string text = "the quick brown fox jumps over the lazy dog";
  req = json{
    {"text", text},
  };
  res = action_tokenize(app, req);
  std::cout << "action_tokenize: " << res << "\n";

  std::vector<int> tokens_to_eval = res["tokens"];
  //tokens_to_eval.insert(tokens_to_eval.begin(), token_bos);
  req = json{
    {"tokens", tokens_to_eval},
  };
  res = action_embeddings(app, req);
  std::cout << "action_embeddings: " << res << "\n";

  // clean up
  llama_free(app.ctx);
  llama_free_model(app.model);
  llama_sampling_free(app.ctx_sampling);
  llama_backend_free();

  return 0;
}
