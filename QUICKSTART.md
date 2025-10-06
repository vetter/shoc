# SHOC Quick Start Guide

Get up and running with SHOC benchmarks in **5 minutes**! ⚡

## What is SHOC?

SHOC (Scalable HeterOgeneous Computing) is a benchmark suite for GPUs that supports:
- **CUDA** (NVIDIA GPUs)
- **OpenCL** (AMD, Intel, NVIDIA GPUs)
- **MPI** (Multi-GPU systems)

## Prerequisites

### Minimum Requirements
- CMake 3.18 or newer
- C++ compiler (g++, clang++)
- **One of**:
  - CUDA Toolkit (for NVIDIA GPUs)
  - OpenCL SDK (for any GPU)

### Optional
- MPI implementation (for multi-GPU benchmarks)
- Python 3 (for advanced tools)

## 🚀 Installation (5 minutes)

### Step 1: Get SHOC

```bash
# Clone the repository
git clone https://github.com/your-org/shoc.git
cd shoc
```

### Step 2: Build

**Choose the approach that matches your system:**

#### Option A: I have an NVIDIA GPU (Easiest)

```bash
cmake --preset=cuda-only
cmake --build --preset=cuda-only -j
cmake --build build-cuda-only --target install
```

#### Option B: I have an AMD/Intel GPU

```bash
cmake --preset=opencl-only
cmake --build --preset=opencl-only -j
cmake --build build-opencl-only --target install
```

#### Option C: I want everything

```bash
cmake --preset=full
cmake --build --preset=full -j
cmake --build build-full --target install
```

#### Option D: Traditional CMake (if you prefer)

```bash
mkdir build && cd build
cmake .. -DSHOC_BUILD_CUDA=ON -DSHOC_BUILD_OPENCL=OFF
make -j
make install
cd ..
```

### Step 3: Verify Installation

```bash
# Run the validator
./tools/validate_install.sh install-cuda  # or install-opencl, install-full
```

You should see:
```
✓ Installation validated successfully!
  • No errors or warnings found
  • 18 benchmarks ready to use
```

## 🎯 Running Your First Benchmark

### Single GPU Benchmark

```bash
# CUDA (NVIDIA)
./install-cuda/bin/Serial/CUDA/MaxFlops

# OpenCL (any GPU)
./install-opencl/bin/Serial/OpenCL/MaxFlops
```

**Expected output:**
```
Running benchmark: MaxFlops
Device: NVIDIA GeForce RTX 3080
Result: 29845.2 GFLOPS
```

### More Examples

```bash
# Memory bandwidth test
./install-cuda/bin/Serial/CUDA/BusSpeedDownload -s 2

# FFT benchmark, size 3
./install-cuda/bin/Serial/CUDA/FFT -s 3

# Reduction with 10 iterations
./install-cuda/bin/Serial/CUDA/Reduction -s 2 -n 10
```

### Multi-GPU with MPI

If you built with MPI support:

```bash
# Run on 2 GPUs (use 1 rank per GPU)
mpirun -np 2 ./install-cuda-mpi/bin/EP/CUDA/BusSpeedDownload

# Run on 4 GPUs
mpirun -np 4 ./install-cuda-mpi/bin/EP/CUDA/FFT -s 2
```

## 📊 Benchmark Categories

SHOC organizes benchmarks into three categories:

### Serial
Single-GPU benchmarks
```bash
./install-cuda/bin/Serial/CUDA/<benchmark>
```

### EP (Embarrassingly Parallel)
Multi-GPU with no inter-GPU communication
```bash
mpirun -np 4 ./install-cuda-mpi/bin/EP/CUDA/<benchmark>
```

### TP (Truly Parallel)
Multi-GPU with inter-GPU communication
```bash
mpirun -np 4 ./install-cuda-mpi/bin/TP/CUDA/<benchmark>
```

## 🔍 Available Benchmarks

### Level 0 (Device Capabilities)
- `BusSpeedDownload` - PCIe bandwidth (host→device)
- `BusSpeedReadback` - PCIe bandwidth (device→host)
- `DeviceMemory` - Device memory bandwidth
- `MaxFlops` - Peak FLOPS measurement
- `KernelCompile` - Kernel compilation time (OpenCL only)
- `QueueDelay` - Queue latency (OpenCL only)

### Level 1 (Core Algorithms)
- `BFS` - Breadth-First Search
- `FFT` - Fast Fourier Transform
- `GEMM` - Matrix multiplication
- `MD` - Molecular Dynamics
- `MD5Hash` - MD5 hashing
- `NeuralNet` - Neural network layer
- `Reduction` - Parallel reduction
- `Scan` - Parallel prefix sum
- `Sort` - Radix sort
- `Spmv` - Sparse matrix-vector multiply
- `Stencil2D` - 2D stencil computation
- `Triad` - STREAM Triad memory test

### Level 2 (Applications)
- `S3D` - Combustion chemistry simulation
- `QTC` - Quality Threshold Clustering

## ⚙️ Common Options

All benchmarks support these flags:

```bash
-s SIZE     # Problem size (1-4), default: 1
-n NUM      # Number of iterations, default: varies
-d DEVICE   # Device ID to use, default: 0
-v          # Verbose output
```

**Examples:**
```bash
# Small problem, quick test
./install-cuda/bin/Serial/CUDA/FFT -s 1 -n 5

# Large problem, device 1
./install-opencl/bin/Serial/OpenCL/GEMM -s 4 -d 1

# Verbose output
./install-cuda/bin/Serial/CUDA/BFS -s 2 -v
```

## 🆘 Troubleshooting

### "No GPU detected"

**CUDA:**
```bash
nvidia-smi  # Check if NVIDIA GPU is visible
```

**OpenCL:**
```bash
clinfo  # Check if OpenCL devices are visible
```

### "Benchmark crashes or gives errors"

Try a smaller problem size:
```bash
./install-cuda/bin/Serial/CUDA/FFT -s 1  # Start small
```

### "Build failed - CUDA not found"

Add CUDA to PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake --preset=cuda-only  # Try again
```

### "Build failed - OpenCL not found"

Install OpenCL headers:
```bash
# Ubuntu/Debian
sudo apt-get install ocl-icd-opencl-dev opencl-headers

# Then rebuild
cmake --preset=opencl-only
```

### "MPI benchmarks don't work"

Install MPI:
```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# Rebuild with MPI
cmake --preset=cuda-mpi
```

## 📖 Next Steps

### Learn More
- **Detailed build options**: [CMAKE_BUILD.md](CMAKE_BUILD.md)
- **All CMake presets**: [CMAKE_PRESETS_GUIDE.md](CMAKE_PRESETS_GUIDE.md)
- **Complete documentation**: [doc/shoc-manual.pdf](doc/shoc-manual.pdf)

### Advanced Usage
- **Run full benchmark suite**: Use the Python driver tool
- **Compare results**: Visualize with plotting tools
- **CI/CD integration**: Use CTest and Docker

### Get Help
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Read the [RECOMMENDATIONS.md](RECOMMENDATIONS.md) for improvements
- File issues on GitHub

## 🎓 Tutorial: Complete Workflow

Here's a complete example workflow:

```bash
# 1. Clone
git clone https://github.com/your-org/shoc.git
cd shoc

# 2. Build (CUDA + OpenCL + MPI)
cmake --preset=full
cmake --build --preset=full -j

# 3. Verify
./tools/validate_install.sh install-full

# 4. Run single benchmark
./install-full/bin/Serial/CUDA/MaxFlops

# 5. Run OpenCL version
./install-full/bin/Serial/OpenCL/MaxFlops

# 6. Compare CUDA vs OpenCL on same algorithm
echo "=== CUDA FFT ==="
./install-full/bin/Serial/CUDA/FFT -s 2

echo "=== OpenCL FFT ==="
./install-full/bin/Serial/OpenCL/FFT -s 2

# 7. Multi-GPU benchmark
mpirun -np 2 ./install-full/bin/EP/CUDA/Reduction -s 3

# 8. List all available benchmarks
find install-full/bin -type f -executable
```

## 💡 Pro Tips

### Tip 1: Use CMake Presets
Instead of remembering long CMake commands, use presets:
```bash
cmake --list-presets              # See all presets
cmake --preset=cuda-only          # Configure
cmake --build --preset=cuda-only  # Build
```

### Tip 2: Validate Before Running
Always validate your installation:
```bash
./tools/validate_install.sh install-cuda
```

### Tip 3: Start Small
When testing a new benchmark, start with size 1:
```bash
./install-cuda/bin/Serial/CUDA/NewBenchmark -s 1
```

### Tip 4: Multiple Configurations
You can have multiple builds simultaneously:
```bash
cmake --preset=cuda-only    # Creates build-cuda-only/
cmake --preset=opencl-only  # Creates build-opencl-only/
# Both coexist peacefully!
```

### Tip 5: Quick Rebuild
After code changes:
```bash
cmake --build --preset=cuda-only -j  # Fast incremental build
```

## 🚀 You're Ready!

You now have:
- ✅ SHOC built and installed
- ✅ Benchmarks verified and working
- ✅ Knowledge of how to run benchmarks
- ✅ Troubleshooting resources

**Start benchmarking!** 🎉

```bash
# Go ahead, try it:
./install-cuda/bin/Serial/CUDA/MaxFlops
```

---

**Questions?** See [CMAKE_BUILD.md](CMAKE_BUILD.md) or [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
