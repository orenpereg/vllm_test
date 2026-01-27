#!/bin/bash
# Benchmark vLLM server for TTFT and TPOT metrics

# Activate virtual environment
source /root/oren/testenv/bin/activate

# Make the Python script executable
chmod +x /root/oren/benchmark_vllm.py

# Run the benchmark
python /root/oren/benchmark_vllm.py "$@"
