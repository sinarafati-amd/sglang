# Optimized MOE Kernel for AMD MI300X/MI350

This directory contains optimized Triton kernels for Mixture of Experts (MOE) layers, specifically tuned for AMD Instinct MI300X (gfx942) and MI350 (gfx950) GPUs.

## Performance Results

**Tested on AMD Instinct MI300X with DeepSeek-V3 style MOE (256 experts, top-8)**

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| GEMM Throughput (TFLOPS) | 229 | 377 | +66% |
| Target seq_len=1K Latency | 9.44ms | 5.75ms | 1.64x faster |

## Quick Start

### For MI350 (gfx950) Users

**See detailed deployment guide**: [DEPLOYMENT_GUIDE_GFX950.md](./DEPLOYMENT_GUIDE_GFX950.md)

Quick verification:
```bash
bash verify_gfx950.sh
```

### Enable the Optimized Kernel

```bash
export SGLANG_USE_OPTIMIZED_MOE=1
python3 -m sglang.launch_server --model-path=... [other args]
```

### Verify It's Working

The server will log when the optimized kernel is enabled:
```
INFO: Optimized MOE kernel enabled for gfx942
# or
INFO: Optimized MOE kernel enabled for gfx950
```

## End-to-End Testing

### Option 1: Quick Test (Inside Existing Container)

1. Copy the optimized code to your container:
```bash
docker cp /path/to/fused_moe_triton CONTAINER_NAME:/tmp/fused_moe_triton
```

2. Run inside container:
```bash
cd /tmp/fused_moe_triton
MODEL=/path/to/model TP=8 ISL=1024 OSL=1024 CONC=4 bash quick_moe_test.sh
```

### Option 2: Docker-based A/B Test (From Host)

```bash
cd /path/to/sglang
./docker_benchmark_moe.sh
```

This will:
1. Start a new Docker container
2. Install the optimized MOE kernel
3. Run baseline benchmark
4. Run optimized benchmark
5. Compare and report results

### Option 3: Manual A/B Test

**Baseline (without optimization):**
```bash
unset SGLANG_USE_OPTIMIZED_MOE
python3 -m sglang.launch_server --model-path=$MODEL ...
# Run benchmark
python3 -m sglang.bench_serving --result-filename=baseline.json ...
```

**Optimized:**
```bash
export SGLANG_USE_OPTIMIZED_MOE=1
python3 -m sglang.launch_server --model-path=$MODEL ...
# Run benchmark
python3 -m sglang.bench_serving --result-filename=optimized.json ...
```

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package init with GPU detection |
| `fused_moe_optimized_kernels.py` | Optimized Triton GEMM kernels |
| `fused_moe_optimized.py` | Entry point wrapper |
| `config_gfx942.py` | MI300X tuned configurations |
| `config_gfx950.py` | MI350 tuned configurations |
| `standalone_test.py` | Kernel-level correctness/perf test |
| `full_moe_test.py` | Full MOE pipeline test |
| `benchmark_optimized.py` | Detailed A/B benchmarking |
| `autotune_gfx942.py` | Autotuning for MI300X |
| `autotune_gfx950.py` | Autotuning for MI350 |

## Key Optimizations

1. **MFMA-aligned tile sizes**: 128x128x64 (vs 64x64x32 baseline)
   - Aligned to `mfma_f32_32x32x16_fp8` instruction
   
2. **L2 cache optimization**: GROUP_SIZE_M=8 for better data reuse across expert blocks

3. **Software pipelining**: 3-4 stages for memory latency hiding

4. **Expert batching**: Group tokens by expert for parallel GEMM

5. **Fused activation**: SiLU/GELU computed in Triton kernel

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SGLANG_USE_OPTIMIZED_MOE=1` | Enable optimized kernel |
| `SGLANG_USE_AITER=1` | Enable AITER (recommended) |

## Autotuning

To find optimal configurations for your specific workload:

```bash
cd /path/to/optimized
python3 autotune_gfx942.py --model deepseek-v3-small --dtype fp8_w8a8 --quick
```

## Troubleshooting

### Kernel not being used
- Check GPU architecture: Must be gfx942 (MI300X) or gfx950 (MI350)
- Verify environment variable: `echo $SGLANG_USE_OPTIMIZED_MOE`
- Check server logs for "Optimized MOE kernel enabled"

### Performance regression
- Try different batch sizes - optimized kernel is best for larger batches
- Run `standalone_test.py` to verify kernel performance
- Check if you're using FP8 quantization (recommended)

### Numerical differences
- Small differences (< 0.1) in BF16/FP8 are expected
- The kernel maintains correctness within quantization tolerance
