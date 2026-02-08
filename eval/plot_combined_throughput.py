#!/usr/bin/env python3
"""
Generate a combined throughput graph for different GPU configurations.
X-axis: Batch Size
Y-axis: Total System Throughput (tokens/sec)
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate throughput comparison graphs across GPU configurations')
    parser.add_argument('--output-dir', '-o', required=True, 
                        help='Output directory containing the CSV files (e.g., output/DS-R1-Distill-Llama-8B)')
    parser.add_argument('--gpus', '-g', default='1,2,4,8',
                        help='Comma-separated list of GPU counts (default: 1,2,4,8)')
    parser.add_argument('--batch-range', '-b', default=None,
                        help='Batch size range to filter (e.g., 1:32 or 1:256). If not specified, plots all available data')
    parser.add_argument('--model-name', '-m', default=None,
                        help='Model name for the plot title (defaults to directory name)')
    
    args = parser.parse_args()
    
    # Parse GPU configurations
    gpu_configs = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Parse batch size range if provided
    batch_min, batch_max = None, None
    if args.batch_range:
        try:
            batch_min, batch_max = map(int, args.batch_range.split(':'))
        except ValueError:
            print(f"Error: Invalid batch range format '{args.batch_range}'. Use format like '1:32' or '1:256'")
            sys.exit(1)
    
    # Determine model name
    model_name = args.model_name if args.model_name else os.path.basename(args.output_dir)
    
    # Color palette (extend if more GPUs are specified)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    if len(gpu_configs) > len(colors):
        colors = colors * ((len(gpu_configs) // len(colors)) + 1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Read and plot data for each GPU configuration
    data_found = False
    all_batch_sizes = set()
    
    for gpu_count, color in zip(gpu_configs, colors):
        csv_file = os.path.join(args.output_dir, f'batch_size_throughput_{gpu_count}_GPU.csv')
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # Filter by batch size range if specified
            if batch_min is not None and batch_max is not None:
                df = df[(df['batch_size'] >= batch_min) & (df['batch_size'] <= batch_max)]
            
            if len(df) > 0:
                plt.plot(df['batch_size'], df['total_system_throughput'], 
                        marker='o', linewidth=2, markersize=8,
                        color=color, label=f'{gpu_count} GPU{"s" if gpu_count > 1 else ""}')
                all_batch_sizes.update(df['batch_size'].tolist())
                print(f"Loaded data from {csv_file} ({len(df)} data points)")
                data_found = True
            else:
                print(f"Warning: No data in range for {csv_file}")
        else:
            print(f"Warning: {csv_file} not found")
    
    if not data_found:
        print("Error: No data found to plot. Check your output directory and parameters.")
        sys.exit(1)
    
    # Customize the plot
    plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
    plt.ylabel('Total System Throughput (tokens/sec)', fontsize=14, fontweight='bold')
    
    # Build title
    title = f'{model_name} Throughput vs Batch Size, TP (out toks=100)\nAcross Different GPU Configurations'
    if batch_min is not None and batch_max is not None:
        title += f'\n(Batch Size Range: {batch_min}-{batch_max})'
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper left')
    
    # Use log scale for x-axis since batch sizes are powers of 2
    plt.xscale('log', base=2)
    sorted_batch_sizes = sorted(all_batch_sizes)
    plt.xticks(sorted_batch_sizes, sorted_batch_sizes)
    
    # Format y-axis with comma separator for thousands
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add grid for better readability
    plt.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(args.output_dir, 'combined_throughput_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
