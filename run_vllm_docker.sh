#!/bin/bash
# Run vLLM using Intel's official Docker container

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
PORT=8000

# GPU configuration (can be overridden with arguments)
NUM_GPUS=${1:-1}          # Default: 1 GPU

# Auto-generate GPU IDs if not provided
if [ -z "$2" ]; then
    # Auto-generate: 0,1,2,3... based on NUM_GPUS
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
else
    GPU_IDS=$2
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please create a .env file with: HF_TOKEN=your_token_here"
    exit 1
fi

echo "Starting vLLM with ${NUM_GPUS} GPU(s)"
echo "GPU IDs: ${GPU_IDS}"
echo "Model: ${MODEL}"
echo ""

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
    --model ${MODEL} \
    --tensor-parallel-size ${NUM_GPUS} \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --block-size 64 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --no-enable-log-requests

echo "vLLM server starting in Docker container 'vllm-xpu'"
echo "Server will be available at: http://0.0.0.0:${PORT}"
echo ""
echo "To view logs: docker logs -f vllm-xpu"
echo "To stop: docker stop vllm-xpu"
