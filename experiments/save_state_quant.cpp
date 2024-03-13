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
  };
  res = action_load(app, req);
  std::cout << "action_load: " << res << "\n";
  int token_bos = res["token_bos"];
  int token_eos = res["token_eos"];

  req = json{};
  res = action_sampling_init(app, req);

  std::string text =
    "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
    "<|im_start|>user\nWho are you?<|im_end|>\n";
    "<|im_start|>assistant\n";
  req = json{
    {"text", text},
  };
  res = action_tokenize(app, req);
  std::cout << "action_tokenize: " << res << "\n";

  std::vector<int> tokens_to_eval = res["tokens"];
  tokens_to_eval.insert(tokens_to_eval.begin(), token_bos);
  req = json{
    {"tokens", tokens_to_eval},
  };
  res = action_decode(app, req);
  std::cout << "action_decode: " << res << "\n";

  std::vector<uint8_t> state(llama_get_state_size(app.ctx));

  for (int i = 0; i < 10; i++) {
    std::cout << "Save state" << res << "\n";
    tstart = TIMED;
    output_size = cama_copy_state_data_quant(app.ctx, state.data()) / 1024;
    tend = TIMED;
    std::cout << "Time taken : " << CALC_TIMED << " ms" << "\n";
    std::cout << "output_size : " << output_size << " kb" << "\n";

    std::cout << "Load state" << res << "\n";
    tstart = TIMED;
    ccama_set_state_data_quant(app.ctx, state.data());
    tend = TIMED;
    std::cout << "Time taken : " << CALC_TIMED << " ms" << "\n";
  }

  for (int i = 0; i < 10; i++) {
    req = json{};
    res = action_sampling_sample(app, req);
    std::vector<uint8_t> text_out;
    std::vector<int> text_buf = res["piece"];
    for (auto b : text_buf) text_out.push_back(b);

    std::cout << std::string((char *) text_out.data(), text_out.size()) << std::flush;

    std::vector<int> new_tokens;
    new_tokens.push_back(res["token"]);
    req = json{
      {"tokens", new_tokens},
    };
    res = action_decode(app, req);
  }

  std::cout << "\n";
  std::cout << "---------------" << "\n";

  // clean up
  llama_free(app.ctx);
  llama_free_model(app.model);
  llama_sampling_free(app.ctx_sampling);
  llama_backend_free();

  return 0;
}
