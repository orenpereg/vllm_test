#!/bin/bash
# Run vLLM using Intel's official Docker container

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT=8000

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please create a .env file with: HF_TOKEN=your_token_here"
    exit 1
fi

docker run -d \
    --rm \
    --name vllm-xpu \
    --privileged \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p ${PORT}:8000 \
    --ipc=host \
    -e http_proxy=http://proxy-dmz.intel.com:912 \
    -e https_proxy=http://proxy-dmz.intel.com:912 \
    -e HTTP_PROXY=http://proxy-dmz.intel.com:912 \
    -e HTTPS_PROXY=http://proxy-dmz.intel.com:912 \
    -e HF_TOKEN=${HF_TOKEN} \
    intel/vllm:0.10.2-xpu \
    --model ${MODEL}

echo "vLLM server starting in Docker container 'vllm-xpu'"
echo "Server will be available at: http://0.0.0.0:${PORT}"
echo ""
echo "To view logs: docker logs -f vllm-xpu"
echo "To stop: docker stop vllm-xpu"
