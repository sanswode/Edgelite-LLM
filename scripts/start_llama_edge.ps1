$ErrorActionPreference = "Stop"

if (-not $env:LLAMA_SERVER_BIN) {
    throw "Please set LLAMA_SERVER_BIN to llama-server.exe"
}

if (-not $env:EDGE_GGUF_PATH) {
    throw "Please set EDGE_GGUF_PATH to your edge GGUF model path"
}

$ctx = if ($env:EDGE_CTX) { $env:EDGE_CTX } else { "2048" }
$ngl = if ($env:EDGE_NGL) { $env:EDGE_NGL } else { "-1" }
$host = if ($env:EDGE_HOST) { $env:EDGE_HOST } else { "127.0.0.1" }
$port = if ($env:EDGE_PORT) { $env:EDGE_PORT } else { "8002" }
$slots = if ($env:EDGE_PARALLEL) { $env:EDGE_PARALLEL } else { "1" }

# Single-GPU laptop preset for RTX 4060 Laptop 8GB.
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }

& $env:LLAMA_SERVER_BIN `
  -m $env:EDGE_GGUF_PATH `
  -ngl $ngl `
  -c $ctx `
  -np $slots `
  --host $host `
  --port $port
