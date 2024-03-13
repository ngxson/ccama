# ccama

Experiments with llama.cpp internal API.

## wllama - WASM low-level binding for llama.cpp

This is a low-level binding for llama.cpp in WASM that supports low-level API like (de)tokenization, embeddings,...

**How to build**

```shell
# require having docker compose installed
cd wasm
./build.sh

# output binary can be found in /build/wllama.wasm
```

**How to use**

See `wasm/main.html` and `wasm/main.js`

Due to CORS limitation, to try the demode, please [install http-server](https://www.npmjs.com/package/http-server) and run `cd wasm && http-server . -c-1`

**TODO**
- Support multi-sequences: knowing the resource limitation when using WASM, I don't think having multi-sequences is a good idea
- Multi-modal: Awaiting refactoring LLaVA implementation from llama.cpp

## Save state quant

Experiment on how to save state to disk as Q4_K (while keeping inference with f16)

To build, run `./build.sh`
