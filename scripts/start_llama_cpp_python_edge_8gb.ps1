$ErrorActionPreference = "Stop"

if (-not $env:EDGE_GGUF_PATH) {
    throw "Please set EDGE_GGUF_PATH to your GGUF model path"
}

$host = if ($env:EDGE_HOST) { $env:EDGE_HOST } else { "127.0.0.1" }
$port = if ($env:EDGE_PORT) { $env:EDGE_PORT } else { "8002" }
$ctx = if ($env:EDGE_CTX) { $env:EDGE_CTX } else { "2048" }
$ngl = if ($env:EDGE_NGL) { $env:EDGE_NGL } else { "-1" }
$batch = if ($env:EDGE_BATCH) { $env:EDGE_BATCH } else { "256" }
$ubatch = if ($env:EDGE_UBATCH) { $env:EDGE_UBATCH } else { "256" }
$threads = if ($env:EDGE_THREADS) { $env:EDGE_THREADS } else { "12" }
$cacheSize = if ($env:EDGE_CACHE_SIZE) { $env:EDGE_CACHE_SIZE } else { "536870912" }
$modelAlias = if ($env:EDGE_MODEL_ALIAS) { $env:EDGE_MODEL_ALIAS } else { "edge-model" }
$chatFormat = if ($env:EDGE_CHAT_FORMAT) { $env:EDGE_CHAT_FORMAT } else { "chatml" }

# This machine only exposes one NVIDIA GPU for the experiment.
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:PYTHONIOENCODING = "utf-8"

& 'D:\anaconda\envs\edgeLLM\python.exe' -m llama_cpp.server `
  --model $env:EDGE_GGUF_PATH `
  --model_alias $modelAlias `
  --host $host `
  --port $port `
  --n_gpu_layers $ngl `
  --main_gpu 0 `
  --n_ctx $ctx `
  --n_batch $batch `
  --n_ubatch $ubatch `
  --n_threads $threads `
  --n_threads_batch $threads `
  --offload_kqv true `
  --flash_attn false `
  --cache true `
  --cache_size $cacheSize `
  --chat_format $chatFormat
