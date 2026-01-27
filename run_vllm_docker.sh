#!/bin/bash
# Run vLLM using Intel's official Docker container

MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT=8000

docker run -d \
    --rm \
    --name vllm-xpu \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p ${PORT}:8000 \
    --ipc=host \
    intel/vllm:0.10.2-xpu \
    --model ${MODEL} \
    --device xpu

echo "vLLM server starting in Docker container 'vllm-xpu'"
echo "Server will be available at: http://0.0.0.0:${PORT}"
echo ""
echo "To view logs: docker logs -f vllm-xpu"
echo "To stop: docker stop vllm-xpu"
