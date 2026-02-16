#!/usr/bin/env python3
"""
Benchmark vLLM server to measure TTFT (Time To First Token) and TPOT (Time Per Output Token)
Supports both custom API benchmarking and vLLM's built-in benchmark tool
"""
import requests
import time
import json
import argparse
from typing import Dict, List, Optional
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import os
import random
import string

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def generate_random_prompt(target_tokens: int) -> str:
    """
    Generate a random prompt with approximately target_tokens tokens.
    Uses ~2 characters per token for random text (more conservative than natural language).
    """
    # For random text, use more conservative ratio: 1 token ≈ 2 characters
    # (random strings tokenize poorly compared to natural language)
    target_chars = target_tokens * 2
    
    # Generate random words (3-10 chars each)
    words = []
    current_length = 0
    
    while current_length < target_chars:
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
        current_length += word_len + 1  # +1 for space
    
    return ' '.join(words)


def measure_streaming_metrics(
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    api_url: str = None
) -> Dict[str, float]:
    """
    Measure TTFT and TPOT using streaming API
    
    Returns:
        Dict with metrics: ttft, tpot, total_time, tokens_generated
    """
    if api_url is None:
        api_url = API_URL
    
    # Use different payload format based on endpoint
    if "chat/completions" in api_url:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
    else:
        # Standard completions endpoint
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
    
    start_time = time.time()
    first_token_time = None
    token_times = []
    tokens_generated = 0
    generated_content = ""
    
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if data.get('choices') and len(data['choices']) > 0:
                            choice = data['choices'][0]
                            
                            # Handle both chat and completions format
                            content = None
                            if 'delta' in choice and 'content' in choice['delta']:
                                # Chat completions format
                                content = choice['delta']['content']
                            elif 'text' in choice:
                                # Standard completions format
                                content = choice['text']
                            
                            if content:
                                current_time = time.time()
                                generated_content += content
                                
                                if first_token_time is None:
                                    first_token_time = current_time
                                else:
                                    token_times.append(current_time)
                                
                                tokens_generated += 1
                        
                        # Check for usage information in the response
                        if 'usage' in data and data.get('usage') and data['usage'].get('completion_tokens'):
                            tokens_generated = data['usage']['completion_tokens']
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        
        # TPOT: average time between tokens (excluding TTFT)
        if len(token_times) > 1:
            time_diffs = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            tpot = statistics.mean(time_diffs) if time_diffs else 0
        elif len(token_times) == 1 and first_token_time:
            tpot = token_times[0] - first_token_time
        else:
            tpot = 0
        
        return {
            "ttft": ttft,
            "tpot": tpot,
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "throughput": tokens_generated / total_time if total_time > 0 else 0,
            "content_length": len(generated_content)
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return {
            "ttft": 0,
            "tpot": 0,
            "total_time": 0,
            "tokens_generated": 0,
            "throughput": 0,
            "error": str(e)
        }


def run_benchmark(
    prompts: List[str],
    max_tokens: int = 1000,
    temperature: float = 0.7,
    num_runs: int = 1,
    api_url: str = None,
    batch_size: int = 1
) -> None:
    """
    Run benchmark on multiple prompts and report aggregated metrics
    
    Args:
        prompts: List of prompts to test
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_runs: Number of runs per prompt
        api_url: API endpoint URL
        batch_size: Number of parallel requests to send concurrently
    """
    if api_url is None:
        api_url = API_URL
    
    print("=" * 80)
    print(f"vLLM Benchmark - TTFT & TPOT Measurements")
    print(f"API Endpoint: {api_url}")
    print(f"Model: {MODEL}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Runs per prompt: {num_runs}")
    print(f"Batch Size (Parallel Requests): {batch_size}")
    print("=" * 80)
    print()
    
    all_results = []
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[Prompt {idx}/{len(prompts)}]")
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print("-" * 80)
        
        prompt_results = []
        
        for run in range(num_runs):
            if num_runs > 1:
                print(f"Run {run + 1}/{num_runs}...")
            
            if batch_size == 1:
                # Sequential execution (original behavior)
                metrics = measure_streaming_metrics(prompt, max_tokens, temperature, api_url)
                prompt_results.append(metrics)
                
                if num_runs > 1:
                    print(f"  TTFT: {metrics['ttft']*1000:.2f}ms, TPOT: {metrics['tpot']*1000:.2f}ms")
            else:
                # Parallel execution
                batch_start_time = time.time()
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # Submit all parallel requests
                    futures = []
                    for i in range(batch_size):
                        future = executor.submit(
                            measure_streaming_metrics,
                            prompt, max_tokens, temperature, api_url
                        )
                        futures.append(future)
                    
                    # Collect results as they complete
                    batch_results = []
                    for i, future in enumerate(as_completed(futures), 1):
                        result = future.result()
                        batch_results.append(result)
                        if num_runs > 1 or batch_size > 1:
                            print(f"  Request {i}/{batch_size} completed: TTFT={result['ttft']*1000:.2f}ms, "
                                  f"TPOT={result['tpot']*1000:.2f}ms, Tokens={result['tokens_generated']}")
                    
                    batch_end_time = time.time()
                    batch_total_time = batch_end_time - batch_start_time
                    
                    # Add batch metrics
                    for result in batch_results:
                        prompt_results.append(result)
                    
                    # Calculate and display batch statistics
                    # Filter out failed requests (those with errors)
                    successful_results = [r for r in batch_results if 'error' not in r and r.get('tokens_generated', 0) > 0]
                    
                    if successful_results:
                        total_tokens = sum(r['tokens_generated'] for r in successful_results)
                        batch_throughput = total_tokens / batch_total_time if batch_total_time > 0 else 0
                        
                        print(f"\n  Batch completed in {batch_total_time:.2f}s")
                        print(f"  Successful requests: {len(successful_results)}/{len(batch_results)}")
                        print(f"  Total tokens across all successful requests: {total_tokens}")
                        print(f"  Batch throughput: {batch_throughput:.2f} tokens/s")
                    else:
                        print(f"\n  Batch failed - all {len(batch_results)} requests had errors")
            
            # Small delay between runs to avoid overwhelming server
            if run < num_runs - 1:
                time.sleep(0.5)
        
        # Filter out failed results
        successful_results = [r for r in prompt_results if 'error' not in r and r.get('tokens_generated', 0) > 0]
        
        if not successful_results:
            print(f"\n✗ All requests failed for this prompt")
            continue
        
        # Calculate average metrics for this prompt
        avg_metrics = {
            "ttft": statistics.mean([r["ttft"] for r in successful_results]),
            "tpot": statistics.mean([r["tpot"] for r in successful_results]),
            "total_time": statistics.mean([r["total_time"] for r in successful_results]),
            "tokens_generated": statistics.mean([r["tokens_generated"] for r in successful_results]),
            "throughput": statistics.mean([r["throughput"] for r in successful_results])
        }
        
        # Add content_length if available
        if successful_results and "content_length" in successful_results[0]:
            avg_metrics["content_length"] = statistics.mean([r["content_length"] for r in successful_results])
        
        all_results.append(avg_metrics)
        
        # Print results for this prompt
        print(f"\nResults (average of {num_runs} run{'s' if num_runs > 1 else ''}):")
        print(f"  TTFT (Time to First Token):  {avg_metrics['ttft']*1000:8.2f} ms")
        print(f"  TPOT (Time per Output Token): {avg_metrics['tpot']*1000:8.2f} ms")
        print(f"  Total Time:                   {avg_metrics['total_time']:8.2f} s")
        print(f"  Tokens Generated:             {avg_metrics['tokens_generated']:8.0f}")
        if 'content_length' in avg_metrics:
            print(f"  Content Length (chars):       {avg_metrics.get('content_length', 0):8.0f}")
        print(f"  Throughput:                   {avg_metrics['throughput']:8.2f} tokens/s")
    
    # Print overall summary
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY (across all prompts)")
        print("=" * 80)
        avg_throughput = statistics.mean([r['throughput'] for r in all_results])
        total_system_throughput = avg_throughput * batch_size
        print(f"  Batch Size:                  {batch_size}")
        print(f"  Average Tokens Generated:    {statistics.mean([r['tokens_generated'] for r in all_results]):8.2f}")
        print(f"  Average TTFT:                {statistics.mean([r['ttft'] for r in all_results])*1000:8.2f} ms")
        print(f"  Average TPOT:                {statistics.mean([r['tpot'] for r in all_results])*1000:8.2f} ms")
        print(f"  Per-Request Throughput:      {avg_throughput:8.2f} tokens/s")
        print(f"  Total System Throughput:     {total_system_throughput:8.2f} tokens/s")
        print(f"  Min TTFT:                    {min([r['ttft'] for r in all_results])*1000:8.2f} ms")
        print(f"  Max TTFT:                    {max([r['ttft'] for r in all_results])*1000:8.2f} ms")
        print(f"  Min TPOT:                    {min([r['tpot'] for r in all_results])*1000:8.2f} ms")
        print(f"  Max TPOT:                    {max([r['tpot'] for r in all_results])*1000:8.2f} ms")
        print("=" * 80)


def run_builtin_benchmark(
    model: str,
    dataset_name: str = "random",
    dataset_path: Optional[str] = None,
    random_input_len: Optional[int] = None,
    random_output_len: Optional[int] = None,
    num_prompts: int = 100,
    max_concurrency: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    server_url: str = "http://localhost:8000",
    save_results: Optional[str] = None
) -> None:
    """
    Run vLLM's built-in benchmark tool
    
    Args:
        model: Model name
        dataset_name: Dataset to use (random, sharegpt, sonnet)
        dataset_path: Path to custom dataset file
        random_input_len: Random input sequence length
        random_output_len: Random output sequence length
        num_prompts: Number of prompts to test
        max_concurrency: Max concurrent requests
        temperature: Sampling temperature
        top_p: Top-p sampling
        server_url: vLLM server URL
        save_results: Path to save results JSON
    """
    print("=" * 80)
    print("vLLM Built-in Benchmark")
    print("=" * 80)
    print(f"Model:              {model}")
    print(f"Dataset:            {dataset_name}")
    if dataset_path:
        print(f"Dataset Path:       {dataset_path}")
    if random_input_len:
        print(f"Input Length:       {random_input_len} tokens")
    if random_output_len:
        print(f"Output Length:      {random_output_len} tokens")
    print(f"Num Prompts:        {num_prompts}")
    print(f"Max Concurrency:    {max_concurrency}")
    print(f"Temperature:        {temperature}")
    print(f"Top-P:              {top_p}")
    print(f"Server URL:         {server_url}")
    if save_results:
        print(f"Save Results:       {save_results}")
    print("=" * 80)
    
    # Construct endpoint URL - remove duplicate path if present
    if server_url.endswith('/v1/chat/completions'):
        endpoint = server_url
    else:
        endpoint = f"{server_url}/v1/chat/completions"
    
    # Build the benchmark command
    cmd = [
        "vllm", "bench", "serve",
        "--model", model,
        "--dataset-name", dataset_name,
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(max_concurrency),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--skip-chat-template",
        "--endpoint", endpoint
    ]
    
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])
    if random_input_len:
        cmd.extend(["--random-input-len", str(random_input_len)])
    if random_output_len:
        cmd.extend(["--random-output-len", str(random_output_len)])
    if save_results:
        cmd.extend(["--save-results", save_results])
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run the benchmark
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("\n✓ Benchmark completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Benchmark failed with error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n✗ Error: 'vllm' command not found. Make sure vLLM is installed.")
        print("   Install with: pip install vllm")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM server for TTFT and TPOT metrics"
    )
    
    # Mode selection
    parser.add_argument(
        "--use-builtin",
        action="store_true",
        help="Use vLLM's built-in benchmark tool instead of custom API benchmarking"
    )
    
    # Common parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=API_URL,
        help=f"API endpoint URL (default: {API_URL})"
    )
    
    # Custom API mode parameters
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to test (if not provided, uses default prompts)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per prompt (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of parallel requests to send concurrently (default: 1)"
    )
    
    # Built-in benchmark mode parameters
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"Model name (default: {MODEL})"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        help="Dataset to use: random, sharegpt, sonnet (default: random)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to custom dataset file"
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        help="Random input sequence length (e.g., 128, 512, 1024, 2048)"
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        help="Random output sequence length (e.g., 128, 512, 1024)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to test (built-in mode, default: 100)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max concurrent requests (built-in mode, default: 8)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Path to save benchmark results JSON"
    )
    
    args = parser.parse_args()
    
    try:
        if args.use_builtin:
            # Use vLLM's built-in benchmark tool (requires vllm CLI installed)
            run_builtin_benchmark(
                model=args.model,
                dataset_name=args.dataset_name,
                dataset_path=args.dataset_path,
                random_input_len=args.random_input_len,
                random_output_len=args.random_output_len,
                num_prompts=args.num_prompts,
                max_concurrency=args.max_concurrency,
                temperature=args.temperature,
                top_p=args.top_p,
                server_url=args.url,
                save_results=args.save_results
            )
        else:
            # Use custom API benchmarking (works with Docker server via HTTP)
            
            # Generate prompts based on random_input_len or use defaults
            if args.random_input_len:
                # Generate random prompts of specified length
                num_test_prompts = args.num_prompts if args.num_prompts != 100 else 5
                prompts = [generate_random_prompt(args.random_input_len) for _ in range(num_test_prompts)]
                print(f"Generated {len(prompts)} random prompts with ~{args.random_input_len} tokens each")
            elif args.prompt:
                prompts = [args.prompt]
            else:
                prompts = [
                    "Write a short poem about artificial intelligence.",
                    "Explain quantum computing in simple terms.",
                    "What are the benefits of using GPUs for AI workloads?",
                    "Tell me about the Python programming language.",
                    "Describe the process of training a neural network."
                ]
            
            # Use random_output_len for max_tokens if specified
            max_tokens = args.random_output_len if args.random_output_len else args.max_tokens
            
            run_benchmark(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=args.temperature,
                num_runs=args.runs,
                api_url=args.url,
                batch_size=args.batch_size
            )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
