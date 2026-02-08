#!/bin/bash
# Shell script to generate throughput comparison graphs

set -e

# Default values
OUTPUT_DIR=""
GPUS="1,2,4,8"
BATCH_RANGE=""
MODEL_NAME=""

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -o OUTPUT_DIR [-g GPUS] [-b BATCH_RANGE] [-m MODEL_NAME]

Generate throughput comparison graphs across GPU configurations.

Required arguments:
    -o OUTPUT_DIR       Directory containing CSV files (e.g., output/DS-R1-Distill-Llama-8B)

Optional arguments:
    -g GPUS            Comma-separated list of GPU counts (default: 1,2,4,8)
    -b BATCH_RANGE     Batch size range to filter (e.g., 1:32 or 1:256)
    -m MODEL_NAME      Model name for plot title (defaults to directory name)
    -h                 Show this help message

Examples:
    # Basic usage with all GPUs and all batch sizes
    $0 -o output/DS-R1-Distill-Llama-8B

    # Specific GPUs only
    $0 -o output/Llama-3.2-1B-Instruct -g 1,2,4

    # Filter batch size range
    $0 -o output/DS-R1-Distill-Llama-8B -b 1:32

    # Custom model name and specific configurations
    $0 -o output/DS-R1-Distill-Llama-8B -g 1,4,8 -b 1:128 -m "DeepSeek-R1"

EOF
    exit 1
}

# Parse command line arguments
while getopts "o:g:b:m:h" opt; do
    case $opt in
        o)
            OUTPUT_DIR="$OPTARG"
            ;;
        g)
            GPUS="$OPTARG"
            ;;
        b)
            BATCH_RANGE="$OPTARG"
            ;;
        m)
            MODEL_NAME="$OPTARG"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory (-o) is required" >&2
    usage
fi

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory '$OUTPUT_DIR' does not exist" >&2
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/plot_combined_throughput.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT" >&2
    exit 1
fi

# Build the command
CMD="python3 $PYTHON_SCRIPT -o $OUTPUT_DIR -g $GPUS"

if [ -n "$BATCH_RANGE" ]; then
    CMD="$CMD -b $BATCH_RANGE"
fi

if [ -n "$MODEL_NAME" ]; then
    CMD="$CMD -m \"$MODEL_NAME\""
fi

# Execute the command
echo "Running: $CMD"
echo "----------------------------------------"
eval $CMD
