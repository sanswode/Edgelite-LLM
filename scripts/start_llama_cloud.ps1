$ErrorActionPreference = "Stop"

if (-not $env:LLAMA_SERVER_BIN) {
    throw "Please set LLAMA_SERVER_BIN to llama-server.exe"
}

if (-not $env:CLOUD_GGUF_PATH) {
    throw "Please set CLOUD_GGUF_PATH to your cloud GGUF model path"
}

$ctx = if ($env:CLOUD_CTX) { $env:CLOUD_CTX } else { "4096" }
$ngl = if ($env:CLOUD_NGL) { $env:CLOUD_NGL } else { "99" }
$host = if ($env:CLOUD_HOST) { $env:CLOUD_HOST } else { "127.0.0.1" }
$port = if ($env:CLOUD_PORT) { $env:CLOUD_PORT } else { "8003" }
$slots = if ($env:CLOUD_PARALLEL) { $env:CLOUD_PARALLEL } else { "2" }

& $env:LLAMA_SERVER_BIN `
  -m $env:CLOUD_GGUF_PATH `
  -ngl $ngl `
  -c $ctx `
  -np $slots `
  --host $host `
  --port $port
