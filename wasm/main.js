import Module from './build/wllama.js';

const cacheName = 'llama-cpp-wasm-cache';

const loadBinaryResource = async (url) => {
  let cache = null, window = self;

  // Try to find if the model data is cached in Web Worker memory.
  if (typeof window === 'undefined') {
    console.debug('`window` is not defined');
  } else if (window && window.caches) {
    cache = await window.caches.open(cacheName);
    const cachedResponse = await cache.match(url);

    if (cachedResponse) {
      const data = await cachedResponse.arrayBuffer();
      const byteArray = new Uint8Array(data);
      return byteArray;
    }
  }


  // Download model and store in cache
  const _promise = new Promise((resolve, reject) => {
    const req = new XMLHttpRequest();
    req.open('GET', url, true);
    req.responseType = 'arraybuffer';
    req.onload = async (_) => {
      const arrayBuffer = req.response; // Note: not req.responseText
      if (arrayBuffer) {
        const byteArray = new Uint8Array(arrayBuffer);
        if (cache) {
          await cache.put(url, new Response(arrayBuffer))
        };
        resolve(byteArray);
      }
    };
    req.send(null);
  });

  return await _promise;
}

///////////////////////////////////////////////////////

const initWorker = async (modelPath) => {
  const emscrModule = {
    noInitialRun: true,
    print: function(text) {
      if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
      console.log(text);
    },
    printErr: function(text) {
      if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
      console.warn(text);
    },
  };

  const wllama = await Module(emscrModule);
  const modelData = await loadBinaryResource(modelPath);

  // create vfs folder for storing model bins
  wllama['FS_createPath']('/', 'models', true, true);

  // load model
  wllama['FS_createDataFile']('/models', 'model.bin', modelData, true, true, true);
  
  // start the program
  console.log(wllama)
  const wllamaStart = wllama.cwrap('wllama_start', 'number', []);
  const wllamaAction = wllama.cwrap('wllama_action', 'string', ['string', 'string']);
  const wllamaExit = wllama.cwrap('wllama_exit', 'number', []);
  const wllamaDecodeException = wllama.cwrap('wllama_decode_exception', 'string', ['number']);
  console.log(await wllamaStart());
  try {
    const tmp = await wllamaAction('load', JSON.stringify({
      model_path: '/models/model.bin',
      seed: 42,
      n_ctx: 2048,
      n_threads: 1,
    }));
    console.log(tmp);
  } catch (e) {
    console.log(await wllamaDecodeException(e));
  }
  
  let decoder = new TextDecoder();
  let res;
  res = JSON.parse(await wllamaAction('sampling_init', JSON.stringify({})));
  console.log(res);
  res = JSON.parse(await wllamaAction('tokenize', JSON.stringify({text: 'Once upon a time'})));
  console.log(res);
  let tokens = res.tokens;
  res = JSON.parse(await wllamaAction('eval', JSON.stringify({tokens})));
  console.log(res);
  let buf = [];
  for (let i = 0; i < 20; i++) {
    res = JSON.parse(await wllamaAction('decode_logits', JSON.stringify({tokens})));
    //console.log(res);
    for (const c of res.piece) {
      buf.push(c);
      console.log(decoder.decode(new Uint8Array(buf)));
    }
    res = JSON.parse(await wllamaAction('eval', JSON.stringify({tokens: [res.token]})));
  }
}

export default async function Wllama() {
  try {
    await initWorker('https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf');
  } catch (e) {
    console.error(e);
  }
}
