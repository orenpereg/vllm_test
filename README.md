# vLLM Benchmarking Suite

A comprehensive benchmarking toolkit for testing vLLM inference performance across different GPU configurations and batch sizes.

## Overview

This project provides tools to:
- Run vLLM servers with various configurations
- Benchmark throughput (TTFT & TPOT) across different batch sizes
- Visualize performance scaling across GPU counts
- Compare multiple models side-by-side

## Project Structure

```
vllm_test/
├── benchmark_vllm.py          # Core benchmarking script for TTFT/TPOT measurements
├── batch_size_sweep.py        # Automated batch size sweep runner
├── run_vllm_server.py         # Python script to launch vLLM server
├── run_vllm_docker.sh         # Docker-based vLLM server launcher
├── benchmark_vllm.sh          # Shell wrapper for benchmarking
├── check_gpu.sh               # GPU availability checker
├── test_vllm.sh               # Quick test script
├── eval/
│   ├── plot_combined_throughput.py  # Generic plotting script
│   └── plot_throughput.sh           # Shell wrapper for plotting
└── output/                    # Benchmark results by model
    ├── DS-R1-Distill-Llama-8B/
    └── Llama-3.2-1B-Instruct/
```

## Prerequisites

- Python 3.8+
- vLLM installation (either local or Docker)
- Intel GPUs
- Required Python packages: `requests`, `pandas`, `matplotlib`

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/orenpereg/vllm_test.git
   cd vllm_test
   ```

2. **Set up environment variables:**
   Create a `.env` file with your Hugging Face token:
   ```bash
   echo "HUGGING_FACE_HUB_TOKEN=your_token_here" > .env
   ```

3. **Install dependencies:**
   ```bash
   pip install requests pandas matplotlib
   ```

## Usage

### Running Benchmarks

#### 1. Start vLLM Server


**Option A: Using Docker**
```bash
./run_vllm_docker.sh
```
**With custom GPU configuration:**
```bash
bash run_vllm_docker.sh 4 0 "0,1,2,3" TP
```
*Parameters: `<num_gpus>` `<gpu_start_id>` `<gpu_ids>` `<parallelism_mode>`*
- **num_gpus**: Number of GPUs to use (e.g., 4)
- **parallelism_mode**: `TP` (Tensor Parallel), `PP` (Pipeline Parallel), or `DP` (Data Parallel)

**Option B: Using Python script**
```bash
python run_vllm_server.py
```


#### 2. Single Benchmark Run

For a specific configuration:
```bash
python benchmark_vllm.py \
    --batch-size 8 \
    --max-tokens 100 \
    --prompt "Write a short poem about artificial intelligence."
```

#### 3. Run Batch Size Sweep

Test across multiple batch sizes (1, 2, 4, 8, 16, 32, 64, 128, 256):
```bash
python batch_size_sweep.py
```
**With custom parameters:**
```bash
python3 batch_size_sweep.py --powers-of-2 --max-batch 256 --timeout 900
```
The script will:
- Run benchmarks for each batch size
- Save results to CSV files
- Generate throughput plots automatically


### Generating Plots

Use the generic plotting tools in the `eval/` directory:

**Basic usage:**
```bash
./eval/plot_throughput.sh -o output/DS-R1-Distill-Llama-8B
```

**With custom GPU configurations:**
```bash
./eval/plot_throughput.sh -o output/DS-R1-Distill-Llama-8B -g 1,2,4,8
```

**Filter by batch size range:**
```bash
./eval/plot_throughput.sh -o output/DS-R1-Distill-Llama-8B -b 1:32
```

**Custom model name:**
```bash
./eval/plot_throughput.sh -o output/DS-R1-Distill-Llama-8B -m "DeepSeek-R1" -b 1:128
```

**Python API:**
```bash
python eval/plot_combined_throughput.py \
    --output-dir output/DS-R1-Distill-Llama-8B \
    --gpus 1,2,4,8 \
    --batch-range 1:256 \
    --model-name "DS-R1-Distill-Llama-8B"
```

## Output Format

### CSV Files

Benchmark results are saved as:
```
output/<model-name>/batch_size_throughput_<N>_GPU.csv
```

Format:
```csv
batch_size,per_request_throughput,total_system_throughput
1,26.38,26.38
2,26.1,52.2
4,25.73,102.92
...
```

### Graphs

Generated plots show:
- **X-axis:** Batch Size (log scale)
- **Y-axis:** Total System Throughput (tokens/sec)
- **Lines:** Different GPU configurations
- **Output:** `combined_throughput_comparison.png`

## Benchmarking Parameters

- **Input Tokens:** ~12 tokens (fixed prompt: "Write a short poem about artificial intelligence.")
- **Output Tokens:** 100 tokens (configurable via `--max-tokens`)
- **Batch Sizes:** Powers of 2 from 1 to 256
- **Metrics:**
  - Time to First Token (TTFT)
  - Time Per Output Token (TPOT)
  - Per-request throughput (tokens/sec)
  - Total system throughput (tokens/sec)

## Models Tested

- DeepSeek-R1-Distill-Llama-8B
- Llama-3.2-1B-Instruct

## Hardware

Tested on Intel Battlemage GPUs with multi-GPU tensor parallelism support.

## Contributing

Feel free to submit issues or pull requests for:
- New benchmark scenarios
- Additional visualization options
- Performance optimizations
- Support for other hardware platforms

## License

MIT License

## Author

Oren Pereg
