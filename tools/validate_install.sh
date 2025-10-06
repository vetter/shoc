#!/bin/bash
#
# SHOC Installation Validator
# Verifies that SHOC was installed correctly and can run benchmarks
#
# Usage: ./validate_install.sh [install_directory]
#        Default: current directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Emoji/symbols for better readability
CHECK="${GREEN}✓${NC}"
CROSS="${RED}✗${NC}"
WARN="${YELLOW}⚠${NC}"
INFO="${BLUE}ℹ${NC}"

# Default to current directory if not specified
INSTALL_DIR="${1:-.}"

# Track issues
ERRORS=0
WARNINGS=0

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SHOC Installation Validator             ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo ""
echo -e "${INFO} Checking installation at: ${BLUE}$INSTALL_DIR${NC}"
echo ""

# Function to check if directory exists
check_dir() {
    local dir="$1"
    local name="$2"

    if [ -d "$dir" ]; then
        echo -e "  ${CHECK} $name"
        return 0
    else
        echo -e "  ${CROSS} $name ${RED}(missing)${NC}"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Function to check if file exists
check_file() {
    local file="$1"
    local name="$2"

    if [ -f "$file" ]; then
        echo -e "  ${CHECK} $name"
        return 0
    else
        echo -e "  ${WARN} $name ${YELLOW}(not found - might be optional)${NC}"
        WARNINGS=$((WARNINGS + 1))
        return 1
    fi
}

# Function to count executables
count_executables() {
    local dir="$1"
    if [ -d "$dir" ]; then
        find "$dir" -type f -executable 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# 1. Check directory structure
echo -e "${BLUE}═══ Checking Directory Structure ═══${NC}"
check_dir "$INSTALL_DIR/bin" "bin/"
check_dir "$INSTALL_DIR/bin/Serial" "bin/Serial/"
check_dir "$INSTALL_DIR/lib" "lib/" || WARNINGS=$((WARNINGS + 1)) # lib might not exist if static

# Check for backend directories
CUDA_DIR="$INSTALL_DIR/bin/Serial/CUDA"
OPENCL_DIR="$INSTALL_DIR/bin/Serial/OpenCL"

HAS_CUDA=false
HAS_OPENCL=false

if [ -d "$CUDA_DIR" ]; then
    echo -e "  ${CHECK} bin/Serial/CUDA/ ${GREEN}(CUDA support detected)${NC}"
    HAS_CUDA=true
else
    echo -e "  ${INFO} bin/Serial/CUDA/ ${BLUE}(not built)${NC}"
fi

if [ -d "$OPENCL_DIR" ]; then
    echo -e "  ${CHECK} bin/Serial/OpenCL/ ${GREEN}(OpenCL support detected)${NC}"
    HAS_OPENCL=true
else
    echo -e "  ${INFO} bin/Serial/OpenCL/ ${BLUE}(not built)${NC}"
fi

if [ "$HAS_CUDA" = false ] && [ "$HAS_OPENCL" = false ]; then
    echo -e "  ${CROSS} ${RED}No CUDA or OpenCL benchmarks found!${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check for MPI directories
if [ -d "$INSTALL_DIR/bin/EP" ]; then
    echo -e "  ${CHECK} bin/EP/ ${GREEN}(MPI EP support detected)${NC}"
fi

if [ -d "$INSTALL_DIR/bin/TP" ]; then
    echo -e "  ${CHECK} bin/TP/ ${GREEN}(MPI TP support detected)${NC}"
fi

echo ""

# 2. Check libraries
echo -e "${BLUE}═══ Checking Libraries ═══${NC}"
check_file "$INSTALL_DIR/lib/libSHOCCommon.a" "libSHOCCommon.a"

if [ "$HAS_OPENCL" = true ]; then
    check_file "$INSTALL_DIR/lib/libSHOCCommonOpenCL.a" "libSHOCCommonOpenCL.a"
fi

echo ""

# 3. Count executables by category
echo -e "${BLUE}═══ Counting Executables ═══${NC}"

if [ "$HAS_CUDA" = true ]; then
    CUDA_SERIAL=$(count_executables "$INSTALL_DIR/bin/Serial/CUDA")
    CUDA_EP=$(count_executables "$INSTALL_DIR/bin/EP/CUDA")
    CUDA_TP=$(count_executables "$INSTALL_DIR/bin/TP/CUDA")
    CUDA_TOTAL=$((CUDA_SERIAL + CUDA_EP + CUDA_TP))

    echo -e "  ${INFO} CUDA Benchmarks:"
    echo -e "      Serial: $CUDA_SERIAL"
    [ $CUDA_EP -gt 0 ] && echo -e "      EP/MPI: $CUDA_EP"
    [ $CUDA_TP -gt 0 ] && echo -e "      TP/MPI: $CUDA_TP"
    echo -e "      ${GREEN}Total: $CUDA_TOTAL${NC}"
fi

if [ "$HAS_OPENCL" = true ]; then
    OPENCL_SERIAL=$(count_executables "$INSTALL_DIR/bin/Serial/OpenCL")
    OPENCL_EP=$(count_executables "$INSTALL_DIR/bin/EP/OpenCL")
    OPENCL_TP=$(count_executables "$INSTALL_DIR/bin/TP/OpenCL")
    OPENCL_TOTAL=$((OPENCL_SERIAL + OPENCL_EP + OPENCL_TP))

    echo -e "  ${INFO} OpenCL Benchmarks:"
    echo -e "      Serial: $OPENCL_SERIAL"
    [ $OPENCL_EP -gt 0 ] && echo -e "      EP/MPI: $OPENCL_EP"
    [ $OPENCL_TP -gt 0 ] && echo -e "      TP/MPI: $OPENCL_TP"
    echo -e "      ${GREEN}Total: $OPENCL_TOTAL${NC}"
fi

TOTAL_BENCHMARKS=$((${CUDA_TOTAL:-0} + ${OPENCL_TOTAL:-0}))
echo -e "  ${GREEN}═══════════════════════════${NC}"
echo -e "  ${GREEN}Grand Total: $TOTAL_BENCHMARKS benchmarks${NC}"

echo ""

# 4. Try running a benchmark
echo -e "${BLUE}═══ Testing Benchmark Execution ═══${NC}"

# Function to test a benchmark
test_benchmark() {
    local bench_path="$1"
    local bench_name=$(basename "$bench_path")
    local backend=$(basename $(dirname "$bench_path"))

    echo -e "  ${INFO} Testing: ${BLUE}$bench_name${NC} (${backend})"

    # Run with minimal parameters and timeout
    if timeout 10s "$bench_path" -s 1 -n 1 >/dev/null 2>&1; then
        echo -e "  ${CHECK} ${GREEN}Benchmark executed successfully!${NC}"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "  ${WARN} ${YELLOW}Benchmark timed out (might be slow or hanging)${NC}"
        else
            echo -e "  ${WARN} ${YELLOW}Benchmark failed (exit code: $exit_code)${NC}"
            echo -e "      ${YELLOW}This might be normal if no GPU is available${NC}"
        fi
        return 1
    fi
}

# Try to find and test one benchmark from each available backend
TESTED=false

if [ "$HAS_CUDA" = true ]; then
    CUDA_BENCH=$(find "$CUDA_DIR" -type f -executable | head -1)
    if [ -n "$CUDA_BENCH" ]; then
        test_benchmark "$CUDA_BENCH"
        TESTED=true
    fi
fi

if [ "$HAS_OPENCL" = true ]; then
    OPENCL_BENCH=$(find "$OPENCL_DIR" -type f -executable | head -1)
    if [ -n "$OPENCL_BENCH" ]; then
        test_benchmark "$OPENCL_BENCH"
        TESTED=true
    fi
fi

if [ "$TESTED" = false ]; then
    echo -e "  ${WARN} ${YELLOW}No benchmarks found to test${NC}"
fi

echo ""

# 5. Check for documentation
echo -e "${BLUE}═══ Checking Documentation ═══${NC}"

check_file "$INSTALL_DIR/../CMAKE_BUILD.md" "CMAKE_BUILD.md" || true
check_file "$INSTALL_DIR/../CMAKE_PRESETS_GUIDE.md" "CMAKE_PRESETS_GUIDE.md" || true
check_file "$INSTALL_DIR/../README.txt" "README.txt" || true

if [ -d "$INSTALL_DIR/doc" ]; then
    check_file "$INSTALL_DIR/doc/shoc-manual.pdf" "shoc-manual.pdf" || true
fi

echo ""

# 6. Check for common tools
echo -e "${BLUE}═══ Checking Available Tools ═══${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "  ${CHECK} nvidia-smi ${GREEN}(NVIDIA GPU tools available)${NC}"
    # Try to show GPU info
    if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 > /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo -e "      ${INFO} Detected GPU: ${BLUE}$GPU_INFO${NC}"
    fi
elif [ "$HAS_CUDA" = true ]; then
    echo -e "  ${WARN} nvidia-smi not found ${YELLOW}(CUDA benchmarks may not work)${NC}"
fi

if command -v clinfo &> /dev/null; then
    echo -e "  ${CHECK} clinfo ${GREEN}(OpenCL tools available)${NC}"
    # Try to show OpenCL device info
    if clinfo -l 2>/dev/null | grep -q "Platform"; then
        DEVICE_COUNT=$(clinfo -l 2>/dev/null | grep -c "Device" || echo "0")
        echo -e "      ${INFO} OpenCL devices found: ${BLUE}$DEVICE_COUNT${NC}"
    fi
elif [ "$HAS_OPENCL" = true ]; then
    echo -e "  ${WARN} clinfo not found ${YELLOW}(useful for debugging OpenCL)${NC}"
fi

if command -v mpirun &> /dev/null || command -v mpiexec &> /dev/null; then
    echo -e "  ${CHECK} MPI runtime available"
else
    if [ -d "$INSTALL_DIR/bin/EP" ] || [ -d "$INSTALL_DIR/bin/TP" ]; then
        echo -e "  ${WARN} MPI runtime not found ${YELLOW}(MPI benchmarks won't work)${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

echo ""

# 7. Suggest next steps
echo -e "${BLUE}═══ Suggested Next Steps ═══${NC}"

if [ "$HAS_CUDA" = true ]; then
    EXAMPLE_CUDA=$(find "$CUDA_DIR" -type f -executable | head -1)
    if [ -n "$EXAMPLE_CUDA" ]; then
        echo -e "  ${INFO} Run a CUDA benchmark:"
        echo -e "      ${BLUE}$EXAMPLE_CUDA -s 2${NC}"
    fi
fi

if [ "$HAS_OPENCL" = true ]; then
    EXAMPLE_OPENCL=$(find "$OPENCL_DIR" -type f -executable | head -1)
    if [ -n "$EXAMPLE_OPENCL" ]; then
        echo -e "  ${INFO} Run an OpenCL benchmark:"
        echo -e "      ${BLUE}$EXAMPLE_OPENCL -s 2${NC}"
    fi
fi

if [ -d "$INSTALL_DIR/bin/EP" ]; then
    echo -e "  ${INFO} Run MPI benchmarks (use 1 rank per GPU):"
    echo -e "      ${BLUE}mpirun -np 2 $INSTALL_DIR/bin/EP/CUDA/<benchmark>${NC}"
fi

echo -e "  ${INFO} List all benchmarks:"
echo -e "      ${BLUE}find $INSTALL_DIR/bin -type f -executable${NC}"

echo ""

# 8. Final summary
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Validation Summary                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ Installation validated successfully!${NC}"
    echo -e "  • No errors or warnings found"
    echo -e "  • $TOTAL_BENCHMARKS benchmarks ready to use"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}✓ Installation appears OK with minor warnings${NC}"
    echo -e "  • $WARNINGS warning(s) found"
    echo -e "  • $TOTAL_BENCHMARKS benchmarks ready to use"
    exit 0
else
    echo -e "${RED}✗ Installation has issues${NC}"
    echo -e "  • $ERRORS error(s) found"
    echo -e "  • $WARNINGS warning(s) found"
    echo ""
    echo -e "${INFO} Troubleshooting tips:"
    echo -e "  1. Did the build complete successfully?"
    echo -e "  2. Did you run 'make install' or 'cmake --build . --target install'?"
    echo -e "  3. Check CMAKE_INSTALL_PREFIX in your build"
    echo -e "  4. See ${BLUE}CMAKE_BUILD.md${NC} for build instructions"
    exit 1
fi
