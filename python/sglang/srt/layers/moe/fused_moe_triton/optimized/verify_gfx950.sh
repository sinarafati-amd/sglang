#!/bin/bash
# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
#
# Quick verification script for MI350 (gfx950) optimized MOE kernels
# Usage: bash verify_gfx950.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SGLang Optimized MOE - MI350 Verification${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        FAILED=1
    fi
}

FAILED=0

# 1. Check ROCm
echo -e "${YELLOW}[1/7] Checking ROCm installation...${NC}"
if command -v rocm-smi &> /dev/null; then
    ROCM_VERSION=$(rocm-smi --showdriverversion 2>/dev/null | grep "Driver version" | awk '{print $3}' || echo "unknown")
    print_status 0 "ROCm installed (version: $ROCM_VERSION)"
else
    print_status 1 "ROCm not found. Please install ROCm 6.2+"
fi

# 2. Check GPU
echo -e "\n${YELLOW}[2/7] Detecting GPU...${NC}"
if command -v rocm-smi &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep "Card series" | awk -F: '{print $2}' | xargs || echo "unknown")
    if [[ "$GPU_NAME" == *"MI350"* ]] || [[ "$GPU_NAME" == *"gfx950"* ]]; then
        print_status 0 "AMD MI350 detected: $GPU_NAME"
    elif [[ "$GPU_NAME" == *"MI300"* ]] || [[ "$GPU_NAME" == *"gfx942"* ]]; then
        print_status 0 "AMD MI300X detected: $GPU_NAME (note: optimizations exist for both)"
    else
        print_status 1 "Unsupported GPU: $GPU_NAME (expected MI350/gfx950)"
    fi
else
    print_status 1 "Cannot detect GPU"
fi

# 3. Check Python
echo -e "\n${YELLOW}[3/7] Checking Python environment...${NC}"
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version | awk '{print $2}')
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
        print_status 0 "Python $PY_VERSION (>= 3.10 required)"
    else
        print_status 1 "Python $PY_VERSION (3.10+ required)"
    fi
else
    print_status 1 "Python3 not found"
fi

# 4. Check PyTorch
echo -e "\n${YELLOW}[4/7] Checking PyTorch with ROCm...${NC}"
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        print_status 0 "PyTorch $TORCH_VERSION with ROCm support"
    else
        print_status 1 "PyTorch found but ROCm not available"
    fi
else
    print_status 1 "PyTorch not installed"
fi

# 5. Check SGLang
echo -e "\n${YELLOW}[5/7] Checking SGLang installation...${NC}"
if python3 -c "import sglang" 2>/dev/null; then
    SGLANG_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
    print_status 0 "SGLang installed (version: $SGLANG_VERSION)"
else
    print_status 1 "SGLang not installed"
fi

# 6. Check Optimized Kernels
echo -e "\n${YELLOW}[6/7] Checking optimized MOE kernels...${NC}"
if python3 -c "import sglang.srt.layers.moe.fused_moe_triton.optimized" 2>/dev/null; then
    print_status 0 "Optimized MOE module found"
    
    # Check GPU arch detection
    GPU_ARCH=$(python3 -c "from sglang.srt.layers.moe.fused_moe_triton.optimized import get_gpu_arch; print(get_gpu_arch())" 2>/dev/null || echo "unknown")
    if [[ "$GPU_ARCH" == "gfx950" ]]; then
        print_status 0 "GPU architecture detected: gfx950 (MI350)"
    elif [[ "$GPU_ARCH" == "gfx942" ]]; then
        print_status 0 "GPU architecture detected: gfx942 (MI300X)"
    else
        print_status 1 "GPU architecture: $GPU_ARCH (expected gfx950 or gfx942)"
    fi
else
    print_status 1 "Optimized MOE module not found"
fi

# 7. Check Environment Variable
echo -e "\n${YELLOW}[7/7] Checking environment configuration...${NC}"
if [[ "$SGLANG_USE_OPTIMIZED_MOE" == "1" ]]; then
    print_status 0 "SGLANG_USE_OPTIMIZED_MOE=1 (enabled)"
else
    print_status 1 "SGLANG_USE_OPTIMIZED_MOE not set or disabled"
    echo -e "   ${YELLOW}→${NC} Run: export SGLANG_USE_OPTIMIZED_MOE=1"
fi

# Final check
echo -e "\n${YELLOW}[FINAL] Testing optimized kernel usage...${NC}"
KERNEL_TEST=$(python3 << 'EOF' 2>&1
try:
    from sglang.srt.layers.moe.fused_moe_triton.optimized import use_optimized_moe_kernel
    result = use_optimized_moe_kernel()
    print("WILL_USE" if result else "WILL_NOT_USE")
except Exception as e:
    print(f"ERROR: {e}")
EOF
)

if [[ "$KERNEL_TEST" == "WILL_USE" ]]; then
    print_status 0 "Optimized kernels will be used"
elif [[ "$KERNEL_TEST" == "WILL_NOT_USE" ]]; then
    print_status 1 "Optimized kernels will NOT be used"
    if [[ "$SGLANG_USE_OPTIMIZED_MOE" != "1" ]]; then
        echo -e "   ${YELLOW}→${NC} Set environment: export SGLANG_USE_OPTIMIZED_MOE=1"
    fi
else
    print_status 1 "Cannot determine kernel status: $KERNEL_TEST"
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e "\n${GREEN}Your system is ready to use optimized MOE kernels on MI350.${NC}"
    echo -e "\nQuick start:"
    echo -e "  ${BLUE}export SGLANG_USE_OPTIMIZED_MOE=1${NC}"
    echo -e "  ${BLUE}python3 -m sglang.launch_server --model-path /path/to/model ...${NC}"
else
    echo -e "${RED}✗ Some checks failed.${NC}"
    echo -e "\n${YELLOW}Please resolve the issues above before using optimized kernels.${NC}"
    echo -e "See DEPLOYMENT_GUIDE_GFX950.md for detailed instructions."
fi
echo -e "${BLUE}========================================${NC}\n"

# Optional: Run performance test
if [ $FAILED -eq 0 ] && [ "$1" == "--run-test" ]; then
    echo -e "${YELLOW}Running performance test...${NC}\n"
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    if [ -f "$SCRIPT_DIR/standalone_test.py" ]; then
        python3 "$SCRIPT_DIR/standalone_test.py"
    else
        echo -e "${YELLOW}standalone_test.py not found in current directory${NC}"
    fi
fi

exit $FAILED
