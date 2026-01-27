#!/usr/bin/env python3
"""
Benchmark vLLM server to measure TTFT (Time To First Token) and TPOT (Time Per Output Token)
"""
import requests
import time
import json
import argparse
from typing import Dict, List, Optional
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"


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
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
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
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                current_time = time.time()
                                generated_content += delta['content']
                                
                                if first_token_time is None:
                                    first_token_time = current_time
                                else:
                                    token_times.append(current_time)
                                
                                tokens_generated += 1
                        
                        # Check for usage information in the response
                        if 'usage' in data and data['usage'].get('completion_tokens'):
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
                    total_tokens = sum(r['tokens_generated'] for r in batch_results)
                    batch_throughput = total_tokens / batch_total_time if batch_total_time > 0 else 0
                    
                    print(f"\n  Batch completed in {batch_total_time:.2f}s")
                    print(f"  Total tokens across all {batch_size} requests: {total_tokens}")
                    print(f"  Batch throughput: {batch_throughput:.2f} tokens/s")
            
            # Small delay between runs to avoid overwhelming server
            if run < num_runs - 1:
                time.sleep(0.5)
        
        # Calculate average metrics for this prompt
        avg_metrics = {
            "ttft": statistics.mean([r["ttft"] for r in prompt_results]),
            "tpot": statistics.mean([r["tpot"] for r in prompt_results]),
            "total_time": statistics.mean([r["total_time"] for r in prompt_results]),
            "tokens_generated": statistics.mean([r["tokens_generated"] for r in prompt_results]),
            "throughput": statistics.mean([r["throughput"] for r in prompt_results])
        }
        
        # Add content_length if available
        if prompt_results and "content_length" in prompt_results[0]:
            avg_metrics["content_length"] = statistics.mean([r["content_length"] for r in prompt_results])
        
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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM server for TTFT and TPOT metrics"
    )
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
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per prompt (default: 1)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=API_URL,
        help=f"API endpoint URL (default: {API_URL})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of parallel requests to send concurrently (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Define test prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "Write a short poem about artificial intelligence.",
            "Explain quantum computing in simple terms.",
            "What are the benefits of using GPUs for AI workloads?",
            "Tell me about the Python programming language.",
            "Describe the process of training a neural network."
        ]
    
    try:
        run_benchmark(
            prompts=prompts,
            max_tokens=args.max_tokens,
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
