#!/usr/bin/env python3
"""
Run benchmark_vllm across different batch sizes and plot results
"""
import subprocess
import json
import re
import matplotlib.pyplot as plt
import sys
import os

def run_benchmark_for_batch_size(batch_size, max_tokens=100, timeout=600):
    """
    Run benchmark for a specific batch size and extract throughput metrics
    """
    print(f"\n{'='*80}")
    print(f"Running benchmark with batch_size={batch_size} (timeout={timeout}s)")
    print(f"{'='*80}")
    
    cmd = [
        "python",
        "/root/oren/benchmark_vllm.py",
        "--batch-size", str(batch_size),
        "--max-tokens", str(max_tokens),
        "--prompt", "Write a short poem about artificial intelligence."
    ]
    
    try:
        print(f"Starting benchmark... (this may take a while for large batch sizes)")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout
        print(output)  # Print the output as it runs
        
        # Extract metrics from output
        per_request_throughput = None
        total_system_throughput = None
        
        # Look for the throughput lines in individual results
        for line in output.split('\n'):
            if 'Throughput:' in line and 'tokens/s' in line:
                match = re.search(r'(\d+\.\d+)\s+tokens/s', line)
                if match and per_request_throughput is None:
                    per_request_throughput = float(match.group(1))
            elif 'Per-Request Throughput:' in line:
                match = re.search(r'(\d+\.\d+)\s+tokens/s', line)
                if match:
                    per_request_throughput = float(match.group(1))
            elif 'Total System Throughput:' in line:
                match = re.search(r'(\d+\.\d+)\s+tokens/s', line)
                if match:
                    total_system_throughput = float(match.group(1))
        
        # Fallback: calculate from per-request if total not found
        if per_request_throughput and not total_system_throughput:
            total_system_throughput = per_request_throughput * batch_size
        
        return {
            'batch_size': batch_size,
            'per_request_throughput': per_request_throughput,
            'total_system_throughput': total_system_throughput
        }
    
    except subprocess.TimeoutExpired:
        print(f"ERROR: Benchmark timed out for batch_size={batch_size}")
        return None
    except Exception as e:
        print(f"ERROR running benchmark for batch_size={batch_size}: {e}")
        return None


def plot_results(results, output_file='output/batch_size_throughput.png'):
    """
    Plot batch size vs throughput
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    
    batch_sizes = [r['batch_size'] for r in results]
    per_request = [r['per_request_throughput'] for r in results]
    total_system = [r['total_system_throughput'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Total System Throughput
    ax1.plot(batch_sizes, total_system, marker='o', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Total System Throughput (tokens/s)', fontsize=12)
    ax1.set_title('Total System Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(batch_sizes)
    
    # Plot 2: Per-Request Throughput
    ax2.plot(batch_sizes, per_request, marker='s', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Per-Request Throughput (tokens/s)', fontsize=12)
    ax2.set_title('Per-Request Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(batch_sizes)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    # Also save data to CSV
    csv_file = output_file.replace('.png', '.csv')
    with open(csv_file, 'w') as f:
        f.write("batch_size,per_request_throughput,total_system_throughput\n")
        for r in results:
            f.write(f"{r['batch_size']},{r['per_request_throughput']},{r['total_system_throughput']}\n")
    print(f"✓ Data saved to: {csv_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sweep batch sizes and plot throughput")
    parser.add_argument('--min-batch', type=int, default=1, help='Minimum batch size (default: 1)')
    parser.add_argument('--max-batch', type=int, default=32, help='Maximum batch size (default: 32)')
    parser.add_argument('--step', type=int, default=1, help='Step size (default: 1)')
    parser.add_argument('--powers-of-2', action='store_true', help='Use powers of 2 (1, 2, 4, 8, 16, 32...)')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens per request (default: 100)')
    parser.add_argument('--output', type=str, default='output/batch_size_throughput.png', help='Output plot filename')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout per batch size in seconds (default: 600)')
    parser.add_argument('--skip-on-timeout', action='store_true', help='Skip remaining batch sizes if one times out')
    
    args = parser.parse_args()
    
    # Update output path to use output-dir if output doesn't have a directory
    if not os.path.dirname(args.output):
        args.output = os.path.join(args.output_dir, args.output)
    
    # Generate batch sizes to test
    if args.powers_of_2:
        batch_sizes = []
        bs = 1
        while bs <= args.max_batch:
            if bs >= args.min_batch:
                batch_sizes.append(bs)
            bs *= 2
    else:
        batch_sizes = list(range(args.min_batch, args.max_batch + 1, args.step))
    
    print(f"\n{'='*80}")
    print(f"BATCH SIZE SWEEP")
    print(f"{'='*80}")
    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Max tokens per request: {args.max_tokens}")
    print(f"{'='*80}\n")
    
    results = []
    
    for batch_size in batch_sizes:
        result = run_benchmark_for_batch_size(batch_size, args.max_tokens, args.timeout)
        if result and result['total_system_throughput'] is not None:
            results.append(result)
            print(f"\n✓ Batch size {batch_size}: "
                  f"Per-request={result['per_request_throughput']:.2f} tokens/s, "
                  f"Total={result['total_system_throughput']:.2f} tokens/s")
        else:
            print(f"\n✗ Failed to get results for batch size {batch_size}")
            if args.skip_on_timeout:
                print(f"⚠ Skipping remaining batch sizes due to failure/timeout")
                break
    
    if not results:
        print("\nERROR: No valid results collected!")
        sys.exit(1)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Batch Size':<12} {'Per-Request (tok/s)':<20} {'Total System (tok/s)':<20}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['batch_size']:<12} {r['per_request_throughput']:<20.2f} {r['total_system_throughput']:<20.2f}")
    print(f"{'='*80}\n")
    
    # Plot results
    plot_results(results, args.output)
    
    print("\n✓ Batch size sweep complete!")


if __name__ == "__main__":
    main()
