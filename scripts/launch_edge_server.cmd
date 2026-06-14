@echo off
setlocal
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=0
set EDGE_MODEL_PATH=C:\Users\不终止\Desktop\edgeTest\edgelite_llm_experiment\models\Qwen3-1.7B-GGUF\Qwen_Qwen3-1.7B-Q4_K_M.gguf
cd /d C:\Users\不终止\Desktop\edgeTest\edgelite_llm_experiment
"D:\anaconda\envs\edgeLLM\python.exe" -m llama_cpp.server --model "%EDGE_MODEL_PATH%" --model_alias edge-model --host 127.0.0.1 --port 8002 --n_gpu_layers -1 --main_gpu 0 --n_ctx 2048 --n_batch 256 --n_ubatch 256 --n_threads 12 --n_threads_batch 12 --offload_kqv true --flash_attn false --cache true --cache_size 536870912 --chat_format chatml
