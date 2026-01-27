#!/usr/bin/env python3
"""
Simple script to run vLLM server on 4 Intel Battlemage GPUs
"""
import subprocess
import sys
import os

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Change to your desired model
HOST = "0.0.0.0"
PORT = 8000
TENSOR_PARALLEL_SIZE = 1  # Start with 1 GPU, change to 4 once working
GPU_ID = 0  # Which GPU to use (0-3 for the 4 Battlemage GPUs)

# Path to vLLM virtual environment
VLLM_VENV_PYTHON = "/root/vllm/.venv/bin/python"

def run_vllm_server():
    """Launch vLLM OpenAI-compatible API server"""
    
    # Build command to run vLLM
    cmd = [
        VLLM_VENV_PYTHON,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", HOST,
        "--port", str(PORT),
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--disable-log-requests"  # Reduce log spam
    ]
    
    # Set environment variables
    env = os.environ.copy()
    
    # Use ONLY venv libraries - don't source system oneAPI to avoid version conflicts
    venv_lib = "/root/vllm/.venv/lib"
    env["LD_LIBRARY_PATH"] = venv_lib
    env["LIBRARY_PATH"] = venv_lib
    env["CPATH"] = "/root/vllm/.venv/include"
    
    # Set oneAPI environment for device access
    env["ZE_ENABLE_PCI_ID_DEVICE_ORDER"] = "1"
    env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{GPU_ID}"  # Select specific GPU
    
    print(f"Starting vLLM server with model: {MODEL_NAME}")
    print(f"Server will be available at: http://{HOST}:{PORT}")
    print(f"Using Intel GPU #{GPU_ID}")
    print(f"Using venv libraries: {venv_lib}\n")
    
    try:
        # Run the server
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Error running vLLM server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_vllm_server()
