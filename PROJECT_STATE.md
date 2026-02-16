# vLLM Benchmark Project State

**Last Updated:** February 12, 2026

## Project Overview
Benchmarking vLLM inference performance on Intel Battlemage GPUs using DeepSeek-R1-Distill-Llama-8B model.

**Goal:** Measure max throughput across different batch sizes and sequence lengths.

## Hardware & Software
- **CPU:** Intel Xeon 6730P (Granite Rapids, 2024)
- **GPUs:** Intel Battlemage (multi-GPU: 1/2/4/8 supported)
- **vLLM:** 0.10.2-xpu (Intel Docker image: `intel/vllm:0.10.2-xpu`)
- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Network:** Intel corporate proxy (http://proxy-dmz.intel.com:912)
- **Python:** python3 (no "python" symlink available)

## Key Files

### 1. run_vllm_docker.sh
Launches vLLM server in Docker container with multi-GPU support.

**Usage:**
```bash
bash run_vllm_docker.sh 8  # 8 GPUs
bash run_vllm_docker.sh 4  # 4 GPUs
bash run_vllm_docker.sh 1  # 1 GPU
```

**Current Configuration:**
```bash
--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
--tensor-parallel-size ${NUM_GPUS}
--max-model-len 4096              # Max context length (input+output)
--gpu-memory-utilization 0.9      # Use 90% of GPU memory for KV cache
--block-size 64                   # KV cache block granularity
--max-num-batched-tokens 8192     # Max tokens in single prefill iteration
--enforce-eager                   # DISABLE XPU graph optimization (currently causes slowness)
--no-enable-prefix-caching        # Disable prefix caching (no shared prompts)
--no-enable-log-requests          # Reduce logging overhead
```

**IMPORTANT NOTES:**
- `--enforce-eager` currently enabled → 10-30x slower (9-19 tokens/s vs 200-400+ tokens/s)
- Remove `--enforce-eager` for max performance benchmarking
- `--max-num-batched-tokens 8192` matches Intel blog benchmarks
- With `--max-num-batched-tokens 8192` and 1024 input tokens, batch 16+ splits into multiple prefill iterations

### 2. benchmark_vllm.py
Core benchmarking script supporting custom API mode (HTTP requests).

**Key Features:**
- Measures TTFT (Time To First Token) and TPOT (Time Per Output Token)
- Supports random-length prompts via `--random-input-len` and `--random-output-len`
- Parallel request execution with `--batch-size`
- Works with `/v1/chat/completions` endpoint (NOT `/v1/completions` - returns 404)

**API Configuration:**
```python
API_URL = "http://localhost:8000/v1/chat/completions"  # NOT /v1/completions
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
```

**Random Prompt Generation:**
- Uses 2 chars/token ratio (random text tokenizes poorly vs natural language)
- Asking for 1024 tokens generates ~2048 chars → ~1024 actual tokens

**Usage:**
```bash
python3 benchmark_vllm.py --batch-size 32 --random-input-len 1024 --random-output-len 512
```

### 3. batch_size_sweep.py
Automates benchmarks across multiple batch sizes (1,2,4,8,16,32,64,128,256).

**Usage:**
```bash
python3 batch_size_sweep.py --powers-of-2 --max-batch 256 --random-input-len 1024 --random-output-len 512 --timeout 1200
```

**Key Parameters:**
- `--powers-of-2`: Test batch sizes 1,2,4,8,16,32,64,128,256
- `--timeout 1200`: 20 minutes per batch size (600=10min default too short for large batches)
- `--random-input-len 1024`: Target ~1024 input tokens
- `--random-output-len 512`: Generate ~512 output tokens

**Output:**
- `output/batch_size_throughput.png`: Dual plot (total throughput + per-request throughput)
- `output/batch_size_throughput.csv`: Raw data

**Output Directory Structure:**
```
output/
├── DS-R1-Distill-Llama-8B/
│   ├── batch_size_throughput_1_GPU.csv
│   ├── batch_size_throughput_2_GPU.csv
│   ├── batch_size_throughput_4_GPU.csv
│   └── batch_size_throughput_8_GPU.csv
└── Llama-3.2-1B-Instruct/
```

## Running Benchmarks

### Standard Workflow
```bash
# 1. Start vLLM server with 8 GPUs
bash run_vllm_docker.sh 8

# 2. Wait for "Application startup complete" (check logs)
docker logs -f vllm-xpu

# 3. Run batch size sweep
python3 batch_size_sweep.py --powers-of-2 --max-batch 256 --random-input-len 1024 --random-output-len 512 --timeout 1200

# 4. View results
cat output/batch_size_throughput.csv
```

### Use tmux for Long-Running Tests
```bash
# Start tmux session (prevents disconnection issues)
tmux new -s benchmark

# Run benchmark (survives SSH disconnects)
python3 batch_size_sweep.py --powers-of-2 --max-batch 256 --random-input-len 1024 --random-output-len 512 --timeout 1200

# Detach: Ctrl+B then D
# Reattach: tmux attach -t benchmark
```

## Current Issues & Solutions

### Issue 1: Server Extremely Slow (9-19 tokens/s)
**Cause:** `--enforce-eager` disables XPU graph optimization  
**Solution:** Remove `--enforce-eager` from run_vllm_docker.sh  
**Expected Performance:** 200-400+ tokens/s without eager mode

### Issue 2: Docker Crashes at Batch 32+
**Cause:** OOM - running out of GPU memory during prefill with 1024 input tokens  
**Attempted Solutions:**
- ✓ `--enable-chunked-prefill` - processes long prompts in chunks (BUT makes it extremely slow)
- ✗ `--kv-cache-dtype fp8` - incompatible with Intel XPU (requires V2 attention, only V1 available)
- ✓ `--max-num-batched-tokens 8192` - limits prefill batch size (trades throughput for stability)

**Current Status:**
- Batch 1-16: Works ✓
- Batch 32+: Crashes with 1024 input tokens
- Batch 32+: Works with 128 input tokens ✓

**Options:**
1. Accept batch 16 max for 1024/512 sequences
2. Test shorter sequences (512/256 or 256/128)
3. Remove `--enforce-eager` and retry larger batches

### Issue 3: /v1/completions Returns 404
**Cause:** Intel vLLM 0.10.2-xpu only supports `/v1/chat/completions`  
**Solution:** Changed API_URL in benchmark_vllm.py to use chat completions endpoint

### Issue 4: UTF-8 Decode Errors at High Batch Sizes
**Cause:** Subprocess output decoding fails after Docker crashes  
**Solution:** Docker crashes are the root cause - see Issue 2

## Important Configuration Choices

### Why These Settings?

**`--max-num-batched-tokens 8192`**
- Matches Intel blog benchmark configuration
- With 1024 input: batch 8 fits in 1 iteration, batch 16+ split across multiple iterations
- Lower value = more iterations = slower but more stable
- Higher value = fewer iterations = faster but uses more memory

**`--max-model-len 4096`**
- Max sequence length (input + output combined)
- Our 1024+512=1536 fits comfortably
- Could lower to 2048 to save memory and fit larger batches

**`--block-size 64`**
- KV cache block granularity (default: 16)
- Larger blocks reduce overhead for long sequences
- Good for 1024/512 token sequences

**`--gpu-memory-utilization 0.9`**
- Use 90% of GPU memory for KV cache (default: 0.9)
- Higher = more memory for batches
- Already at recommended value

**`--tensor-parallel-size ${NUM_GPUS}`**
- Distributes model across multiple GPUs
- Required for 8B model on Intel Battlemage

**`--no-enable-prefix-caching`**
- Random prompts have no shared prefixes
- Disabling saves memory and avoids cache overhead

## Performance Expectations

### Without --enforce-eager (Recommended)
- Small batches (1-8): 25-35 tokens/s per request
- Medium batches (16-32): 30-35 tokens/s per request
- Large batches (64+): May crash due to memory

### With --enforce-eager (Current - NOT Recommended)
- All batches: 9-19 tokens/s (10-30x slower)
- Should only be used for debugging, not benchmarking

## Known Limitations

### Intel XPU Specific
- No FP8 KV cache support (would save 50% memory)
- Only V1 attention backend available
- XPU graph optimization required for good performance (disabled by `--enforce-eager`)

### DeepSeek-R1-Distill-Llama-8B Specific
- Requires `--tensor-parallel-size` for multi-GPU
- No `/v1/completions` endpoint support

### Memory Constraints
- Max batch size depends on sequence length:
  - 128/512 tokens: Can reach batch 256+
  - 1024/512 tokens: Limited to batch 8-16
- KV cache grows linearly with batch_size × max_seq_len

## TODO / Next Steps
1. **Remove `--enforce-eager`** to get proper benchmark performance
2. Test if batch 32+ works without eager mode
3. Consider testing shorter sequences (512/256) to reach higher batch sizes
4. Add GPU count to output filenames (optional - was discussed but skipped)
5. Clean up redundant files (benchmark_vllm_builtin.sh, sequence_length_sweep.py)

## Quick Reference Commands

```bash
# Start server
bash run_vllm_docker.sh 8

# Stop server
docker stop vllm-xpu

# Check server status
docker ps -a | grep vllm
docker logs vllm-xpu | tail -30

# Test single batch
python3 benchmark_vllm.py --batch-size 32 --random-input-len 1024 --random-output-len 512

# Full sweep
python3 batch_size_sweep.py --powers-of-2 --max-batch 256 --random-input-len 1024 --random-output-len 512 --timeout 1200

# Check if server is ready
curl http://localhost:8000/v1/models
```

## Environment
```bash
# Python virtual environment
source venv/bin/activate

# HuggingFace token in .env file
cat .env
# HF_TOKEN=your_token_here
```
