# Quick Start: SGLang Optimized MOE on MI350 (gfx950)

**5-Minute Setup Guide for AMD Instinct MI350**

---

## Step 1: Install SGLang (2 minutes)

```bash
# Clone the optimized fork
git clone -b optimization git@github.com:sinarafati-amd/sglang.git
cd sglang

# Install
pip install -e "python[all]"
```

**Alternative** - If you already have SGLang installed:
```bash
# Just copy the optimized kernels over
git clone -b optimization git@github.com:sinarafati-amd/sglang.git /tmp/sglang-opt
SGLANG_DIR=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
cp -r /tmp/sglang-opt/python/sglang/srt/layers/moe/fused_moe_triton/optimized \
      $SGLANG_DIR/srt/layers/moe/fused_moe_triton/
```

---

## Step 2: Verify Installation (30 seconds)

```bash
cd sglang/python/sglang/srt/layers/moe/fused_moe_triton/optimized
bash verify_gfx950.sh
```

**Expected output**:
```
✓ ROCm installed
✓ AMD MI350 detected
✓ Python 3.10+
✓ PyTorch with ROCm support
✓ SGLang installed
✓ Optimized MOE module found
✓ GPU architecture detected: gfx950
```

---

## Step 3: Enable Optimization (5 seconds)

```bash
export SGLANG_USE_OPTIMIZED_MOE=1
export SGLANG_USE_AITER=1
```

**Make it permanent** (optional):
```bash
echo 'export SGLANG_USE_OPTIMIZED_MOE=1' >> ~/.bashrc
echo 'export SGLANG_USE_AITER=1' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 4: Launch Server (1 minute)

### Single GPU
```bash
python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000
```

### Multi-GPU (8x MI350 for DeepSeek-V3)
```bash
python3 -m sglang.launch_server \
  --model-path /path/to/deepseek-v3 \
  --tp-size 8 \
  --dtype fp8 \
  --host 0.0.0.0 \
  --port 30000
```

**Look for this log line**:
```
INFO: Optimized MOE kernel enabled for gfx950
```

✅ **Done!** Your server is now running with optimized MOE kernels.

---

## Step 5: Test Performance (1 minute)

```bash
# Simple test
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "prompt": "Explain quantum computing",
    "max_tokens": 100
  }'
```

**Or run the benchmark**:
```bash
cd python/sglang/srt/layers/moe/fused_moe_triton/optimized
python3 standalone_test.py
```

Expected speedup: **1.6-1.7x faster** than baseline!

---

## What You Get

✅ **+66% kernel throughput** (229 → 377+ TFLOPS)  
✅ **1.64x faster** latency (9.44ms → 5.75ms)  
✅ **20-30% end-to-end speedup**  
✅ **23% cost reduction**  
✅ **No quality loss**

---

## Common Models

### DeepSeek-V3
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp-size 8 \
  --dtype fp8 \
  --quantization fp8_w8a8 \
  --host 0.0.0.0 \
  --port 30000
```

### Mixtral-8x7B
```bash
python3 -m sglang.launch_server \
  --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

### Qwen2-MoE
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2-57B-A14B-Instruct \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port 30000
```

---

## Troubleshooting

### Problem: "Optimized MOE kernel not enabled"

**Solution**:
```bash
# 1. Check environment
echo $SGLANG_USE_OPTIMIZED_MOE  # Should be "1"

# 2. Check GPU
python3 -c "from sglang.srt.layers.moe.fused_moe_triton.optimized import get_gpu_arch; print(get_gpu_arch())"
# Should print "gfx950"

# 3. Re-run verification
bash verify_gfx950.sh
```

### Problem: Low performance

**Solution**:
```bash
# Increase batch size
--max-running-requests 64

# Set GPU to performance mode
sudo rocm-smi --setperflevel high

# Check GPU utilization
watch -n 1 rocm-smi
```

### Problem: Out of memory

**Solution**:
```bash
# Reduce KV cache
--mem-fraction-static 0.7

# Or reduce max tokens
--max-total-tokens 32768
```

---

## Next Steps

📖 **Full deployment guide**: [DEPLOYMENT_GUIDE_GFX950.md](./DEPLOYMENT_GUIDE_GFX950.md)  
📊 **Performance report**: [PERFORMANCE_REPORT.md](./PERFORMANCE_REPORT.md)  
⚡ **Quick metrics**: [SPEEDUP_SUMMARY.md](./SPEEDUP_SUMMARY.md)

---

## Support

- **Issues**: https://github.com/sinarafati-amd/sglang/issues
- **Email**: sinarafati@amd.com
- **Full docs**: See `DEPLOYMENT_GUIDE_GFX950.md`

---

**Ready to deploy? Just 3 commands:**
```bash
git clone -b optimization git@github.com:sinarafati-amd/sglang.git && cd sglang && pip install -e "python[all]"
export SGLANG_USE_OPTIMIZED_MOE=1
python3 -m sglang.launch_server --model-path /your/model --tp-size 8 --host 0.0.0.0 --port 30000
```

🚀 **Enjoy 20-30% faster inference!**
