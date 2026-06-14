# Model placeholder

Recommended real-model setup for this machine:

- Edge primary model: Qwen 2.5/3 4B Instruct GGUF Q4
- Optional stronger model: Qwen 2.5/3 7B or 8B GGUF Q4, but keep context conservative
- Local service backend: `llama.cpp`

Expected file formats:

- `llama.cpp`: `*.gguf`
- `vLLM`: Hugging Face model repo or local Transformers model directory

Suggested examples:

- `models/qwen-4b-edge.gguf`
- `models/qwen-8b-cloud.gguf`
