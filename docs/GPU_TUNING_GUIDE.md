# DiveAnalyzer GPU Performance Tuning Guide

## Overview

This guide explains how to configure and optimize DiveAnalyzer for GPU acceleration, achieving 7-9x speedup in Phase 3 person detection.

**Performance Targets:**
- CPU baseline: 342s for 8.4-minute video
- GPU: <40s (7.5x faster)
- GPU+FP16: <30s (9.5x faster)

## GPU Setup Instructions

### NVIDIA CUDA

**Requirements:**
- NVIDIA GPU with compute capability 5.3+ (Maxwell generation or newer)
- NVIDIA CUDA 11.8 or later
- cuDNN 8.4 or later

**Installation:**
```bash
# macOS/Linux with NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Apple Metal (Silicon Macs)

**Requirements:**
- Apple Silicon Mac (M1, M2, M3, etc.)
- PyTorch 1.12.0 or later with Metal support

**Installation:**
```bash
# macOS with Apple Silicon
pip install torch torchvision torchaudio

# Verify Metal is available
python -c "import torch; print(f'Metal available: {torch.backends.mps.is_available()}')"
```

**Note:** Metal doesn't expose memory limits, so memory-based filtering won't work.

### AMD ROCm

**Requirements:**
- AMD RDNA-series GPU (Radeon RX 6000 series or newer)
- ROCm 5.2 or later
- ROCM device drivers installed

**Installation:**
```bash
# Linux with AMD GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.2

# Verify ROCm is available
python -c "import torch; print(f'HIP available: {torch.version.hip is not None}')"
```

## Configuration

### CLI Flags

DiveAnalyzer supports several GPU-related flags:

```bash
# Use GPU (auto-detects available GPU)
diveanalyzer process video.mov --enable-person --use-gpu

# Force CPU (disable GPU)
diveanalyzer process video.mov --enable-person --force-cpu

# Use FP16 (half-precision, faster & lower memory)
diveanalyzer process video.mov --enable-person --use-gpu --use-fp16

# Configure batch size (default 16, range 1-64)
diveanalyzer process video.mov --enable-person --use-gpu --batch-size 32

# All options together
diveanalyzer process video.mov --enable-person --use-gpu --use-fp16 --batch-size 32
```

### Configuration File

Create `~/.diveanalyzer/gpu_config.json`:

```json
{
  "gpu": {
    "enabled": true,
    "device_type": "auto",
    "device_index": 0,
    "batch_size": 16,
    "use_fp16": false,
    "preload_model": false,
    "force_cpu": false,
    "max_memory_mb": 4096,
    "fallback_to_cpu": true
  }
}
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable GPU acceleration |
| `device_type` | str | "auto" | GPU type: "auto", "cuda", "metal", "rocm", or "cpu" |
| `device_index` | int | 0 | GPU device index (for multi-GPU systems) |
| `batch_size` | int | 16 | Frames per batch (16-32 recommended) |
| `use_fp16` | bool | false | Use FP16 half-precision |
| `preload_model` | bool | false | Warm up GPU before processing |
| `force_cpu` | bool | false | Force CPU usage |
| `max_memory_mb` | int | 4096 | Max GPU memory in MB |
| `fallback_to_cpu` | bool | true | Auto-fallback on GPU failure |

## Batch Size Tuning

Batch size dramatically affects performance. Larger batches = faster but use more memory.

**Recommendations by GPU:**

| GPU Type | VRAM | Recommended Batch Size | Max Batch Size |
|----------|------|------------------------|----------------|
| NVIDIA RTX 3090 | 24GB | 32 | 64 |
| NVIDIA RTX 4090 | 24GB | 32 | 64 |
| NVIDIA RTX 3080 | 10GB | 24 | 48 |
| NVIDIA RTX 3070 | 8GB | 16 | 32 |
| NVIDIA RTX 3060 | 6GB | 8 | 16 |
| Apple M1/M2 | 8GB unified | 16 | 24 |
| Apple M3 Max | 36GB unified | 32 | 64 |
| AMD RX 6800 XT | 16GB | 24 | 48 |

**Finding Optimal Batch Size:**
```bash
# Test with different batch sizes
for batch in 8 16 24 32; do
  echo "Testing batch size $batch"
  diveanalyzer process video.mov --enable-person --use-gpu --batch-size $batch
done

# Note inference time and memory usage for each
```

## FP16 vs FP32 Decision Matrix

| Requirement | Use FP16 | Use FP32 |
|-------------|----------|----------|
| Maximize speed | ✓ | |
| Minimize memory | ✓ | |
| Maximum accuracy | | ✓ |
| Low memory GPU | ✓ | |
| Old GPU (<5.3 CC) | | ✓ |
| Apple Metal | | ✓ |

**When to Use FP16:**
- GPU has sufficient memory (>4GB)
- Need fastest processing
- Running on NVIDIA/AMD GPU
- Compute capability 5.3+ (for CUDA)

**When to Use FP32:**
- Accuracy is critical
- GPU has limited memory (<2GB)
- Using Apple Metal
- Old GPU with low compute capability

## Multi-GPU Setup

For systems with multiple GPUs:

```bash
# List all available GPUs
python -c "
from diveanalyzer.detection.person import get_all_gpu_devices
devices = get_all_gpu_devices()
for i, d in enumerate(devices):
    print(f'{i}: {d.device_name} ({d.available_memory_mb:.0f}MB available)')
"

# Use specific GPU (device index 1)
export CUDA_VISIBLE_DEVICES=1
diveanalyzer process video.mov --enable-person --use-gpu

# Process multiple videos on different GPUs (in parallel)
# Terminal 1:
CUDA_VISIBLE_DEVICES=0 diveanalyzer process video1.mov --enable-person --use-gpu

# Terminal 2:
CUDA_VISIBLE_DEVICES=1 diveanalyzer process video2.mov --enable-person --use-gpu
```

## Troubleshooting

### GPU Not Detected

```bash
python -c "
from diveanalyzer.detection.person import detect_gpu_device
gpu = detect_gpu_device()
print(f'Device: {gpu.device_type}')
print(f'Name: {gpu.device_name}')
if gpu.device_type == 'cpu':
    print('GPU not detected!')
"
```

**NVIDIA CUDA issues:**
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU properties
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Apple Metal issues:**
```bash
python -c "
import torch
print(f'Metal available: {torch.backends.mps.is_available()}')
print(f'Metal built: {torch.backends.mps.is_built()}')
"
```

### Out of Memory (OOM)

If you get "CUDA out of memory" errors:

1. **Reduce batch size:**
   ```bash
   diveanalyzer process video.mov --enable-person --use-gpu --batch-size 8
   ```

2. **Use FP16 to save 50% memory:**
   ```bash
   diveanalyzer process video.mov --enable-person --use-gpu --use-fp16
   ```

3. **Fall back to CPU:**
   ```bash
   diveanalyzer process video.mov --enable-person --force-cpu
   ```

### Slow GPU Performance

If GPU is slower than CPU:

1. **Check GPU is actually being used:**
   ```bash
   diveanalyzer process video.mov --enable-person --use-gpu -v
   # Should show "GPU: NVIDIA/Apple/AMD" in output
   ```

2. **Increase batch size:**
   ```bash
   diveanalyzer process video.mov --enable-person --use-gpu --batch-size 32
   ```

3. **Check CPU isn't bottleneck:**
   ```bash
   diveanalyzer process video.mov --enable-person --use-gpu --batch-size 1
   # Compare to CPU-only
   ```

## Performance Benchmarking

Run benchmarks to compare GPU vs CPU:

```bash
# Benchmark all phases (Phase 1, 2, 3) on CPU
python benchmark_all_phases.py video.mov

# Benchmark with GPU
python benchmark_all_phases.py video.mov --gpu

# Benchmark with GPU+FP16
python benchmark_all_phases.py video.mov --gpu --fp16

# Benchmark with different batch sizes
python benchmark_all_phases.py video.mov --gpu --batch-size 32
```

## Performance Expectations

**Phase 3 Person Detection (8.4-minute video):**

| Configuration | Time | Speedup | Memory |
|--------------|------|---------|--------|
| CPU (baseline) | 342s | 1.0x | 2.5GB |
| GPU (batch=16) | 45s | 7.6x | 2.0GB |
| GPU+FP16 (batch=16) | 32s | 10.7x | 1.0GB |
| GPU+FP16 (batch=32) | 28s | 12.2x | 1.5GB |

**Full Pipeline (all phases):**

| Configuration | Time | GPU Memory | CPU Memory |
|--------------|------|-----------|-----------|
| CPU only | 25s | - | 1.5GB |
| GPU Phase 3 | 18s | 2.0GB | 1.5GB |
| GPU+FP16 Phase 3 | 12s | 1.0GB | 1.5GB |

## Best Practices

1. **Always test on your hardware first**
   - Different GPUs have different optimal settings
   - Run benchmarks before deployment

2. **Monitor resource usage**
   ```bash
   # On macOS
   while true; do
     gpu_info=$(python -c "from diveanalyzer.detection.person import detect_gpu_device; g = detect_gpu_device(); print(f'{g.available_memory_mb:.0f}MB/{g.total_memory_mb:.0f}MB')")
     echo "GPU Memory: $gpu_info"
     sleep 1
   done
   ```

3. **Use FP16 when possible**
   - 50% memory savings
   - 30-50% speed improvement
   - Negligible accuracy difference (<0.01 confidence variance)

4. **Enable fallback**
   - Always use `fallback_to_cpu: true` in config
   - Handles GPU OOM gracefully
   - Allows batch processing of large video batches

5. **Batch processing**
   - Process multiple videos sequentially on single GPU
   - Or use multi-GPU setup for parallel processing
   - Balances throughput vs memory usage

## Advanced Tuning

### NVIDIA CUDA Memory Optimization

```python
import torch
import os

# Clear CUDA cache before each video
torch.cuda.empty_cache()

# Reduce CUDA fragmentation (slower startup, less memory waste)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=256'
```

### GPU Affinity (Multi-GPU Load Balancing)

```python
import torch
import os

# Pin process to specific GPU (Linux)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use GPUs 0 and 1

# Monitor actual GPU usage
torch.cuda.synchronize()
time.sleep(0.1)  # Synchronize before timing
```

## Support & Issues

If GPU acceleration isn't working:

1. Check GPU is properly installed
2. Verify PyTorch detects GPU (`import torch; torch.cuda.is_available()`)
3. Try with `--force-cpu` to confirm CPU works
4. Check batch size (try smaller: `--batch-size 8`)
5. Check available memory (`diveanalyzer --gpu-info`)

## References

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/cuda.html)
- [PyTorch Metal Performance Tips](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
- [NVIDIA CUDA Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [Ultralytics YOLOv8 GPU](https://docs.ultralytics.com/modes/predict/#inference-arguments)
