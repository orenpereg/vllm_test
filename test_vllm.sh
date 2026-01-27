#!/bin/bash
# Test vLLM server with a chat completion request

MODEL="meta-llama/Llama-3.2-1B-Instruct"
PROMPT="${1:-Tell me about Intel GPUs and their use in AI workloads.}"
MAX_TOKENS="${2:-500}"

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": 0.7
  }" | jq -r '.choices[0].message.content'

echo ""
