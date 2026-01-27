#!/bin/bash
# Monitor which GPU is being used by the vLLM process

echo "=== Intel GPU Device Info ==="
echo ""

for i in {0..4}; do
    if [ -f /sys/class/drm/card${i}/device/vendor ]; then
        vendor=$(cat /sys/class/drm/card${i}/device/vendor)
        device=$(cat /sys/class/drm/card${i}/device/device 2>/dev/null || echo "unknown")
        echo "card${i}: vendor=$vendor device=$device"
    fi
done

echo ""
echo "=== Active vLLM Process GPU Usage ==="
echo ""

# Find vLLM process
VLLM_PID=$(pgrep -f "vllm.entrypoints.openai.api_server" | head -1)

if [ -z "$VLLM_PID" ]; then
    echo "vLLM server is not running"
    exit 1
fi

echo "vLLM PID: $VLLM_PID"
echo ""

# Check which render nodes are in use
echo "Open file descriptors for render nodes:"
ls -la /proc/$VLLM_PID/fd 2>/dev/null | grep -E "renderD|card" || echo "No GPU file descriptors found"

echo ""
echo "To monitor GPU activity in real-time, try:"
echo "  watch -n 1 'cat /sys/class/drm/card*/device/drm/card*/gt/gt0/rps_cur_freq_mhz 2>/dev/null'"
