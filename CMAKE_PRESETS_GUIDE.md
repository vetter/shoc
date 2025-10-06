# CMake Presets Guide for SHOC

CMake Presets make building SHOC incredibly easy! Instead of remembering long command lines, you can use simple preset names.

## 🚀 Quick Start

### List Available Presets

```bash
cd shoc
cmake --list-presets
```

You'll see output like:
```
Available configure presets:

  "cuda-only"            - CUDA Only
  "cuda-mpi"             - CUDA with MPI
  "opencl-only"          - OpenCL Only
  "opencl-mpi"           - OpenCL with MPI
  "full"                 - Full Build (CUDA + OpenCL + MPI)
  "minimal"              - Minimal Build
  "dev-debug"            - Development (Debug)
  "dev-asan"             - Development (AddressSanitizer)
  "cuda-arch-ampere"     - CUDA for Ampere GPUs (RTX 30xx, A100)
  "cuda-arch-turing"     - CUDA for Turing GPUs (RTX 20xx, T4)
  "cuda-arch-volta"      - CUDA for Volta GPUs (V100)
  "cuda-arch-pascal"     - CUDA for Pascal GPUs (GTX 10xx, P100)
```

### Build with a Preset

**Three simple steps:**

```bash
# 1. Configure
cmake --preset=cuda-only

# 2. Build
cmake --build --preset=cuda-only

# 3. Install
cmake --build build-cuda-only --target install
```

That's it! Your benchmarks are ready in `install-cuda/bin/`.

## 📋 Available Presets

### Production Builds

#### `cuda-only`
**Best for: NVIDIA GPU users who only need CUDA**

```bash
cmake --preset=cuda-only
cmake --build --preset=cuda-only
```

- ✅ CUDA benchmarks
- ❌ OpenCL benchmarks
- ❌ MPI support
- 📁 Installs to: `install-cuda/`
- ⚡ Fastest build time

#### `opencl-only`
**Best for: AMD GPU users, Intel GPU users, or portable builds**

```bash
cmake --preset=opencl-only
cmake --build --preset=opencl-only
```

- ❌ CUDA benchmarks
- ✅ OpenCL benchmarks
- ❌ MPI support
- 📁 Installs to: `install-opencl/`
- 🔧 Works with any OpenCL-compatible GPU

#### `cuda-mpi`
**Best for: Multi-GPU systems with CUDA**

```bash
cmake --preset=cuda-mpi
cmake --build --preset=cuda-mpi
```

- ✅ CUDA benchmarks
- ❌ OpenCL benchmarks
- ✅ MPI support (EP and TP variants)
- 📁 Installs to: `install-cuda-mpi/`

#### `opencl-mpi`
**Best for: Multi-GPU systems with OpenCL**

```bash
cmake --preset=opencl-mpi
cmake --build --preset=opencl-mpi
```

- ❌ CUDA benchmarks
- ✅ OpenCL benchmarks
- ✅ MPI support (EP and TP variants)
- 📁 Installs to: `install-opencl-mpi/`

#### `full`
**Best for: Systems with everything, complete testing**

```bash
cmake --preset=full
cmake --build --preset=full
```

- ✅ CUDA benchmarks
- ✅ OpenCL benchmarks
- ✅ MPI support
- ✅ Stability tests
- 📁 Installs to: `install-full/`
- ⏱️ Longest build time (~10 minutes)

#### `minimal`
**Best for: Quick testing, CI/CD**

```bash
cmake --preset=minimal
cmake --build --preset=minimal
```

- ✅ CUDA benchmarks only
- ❌ Everything else disabled
- 📁 Installs to: `install-minimal/`
- ⚡ Fastest build (~2 minutes)

### Development Builds

#### `dev-debug`
**Best for: Debugging, development work**

```bash
cmake --preset=dev-debug
cmake --build --preset=dev-debug
```

- ✅ All features enabled
- ✅ Debug symbols
- ✅ CUDA debug mode (-G)
- ❌ No optimizations
- 📁 Installs to: `install-debug/`
- 🐛 Use with gdb/cuda-gdb

#### `dev-asan`
**Best for: Finding memory leaks, buffer overflows**

```bash
cmake --preset=dev-asan
cmake --build --preset=dev-asan
```

- ✅ OpenCL benchmarks (ASan doesn't work well with CUDA)
- ✅ AddressSanitizer enabled
- 📁 Installs to: `install-asan/`
- 🔍 Detects memory errors at runtime

### GPU-Specific Builds

#### `cuda-arch-ampere`
**For: RTX 3050/3060/3070/3080/3090, RTX A4000/A5000/A6000, A100**

```bash
cmake --preset=cuda-arch-ampere
cmake --build --preset=cuda-arch-ampere
```

- Compute capability: 8.0, 8.6
- ⚡ Fastest compilation for Ampere GPUs
- ✅ Optimized code for your GPU

#### `cuda-arch-turing`
**For: RTX 2060/2070/2080, Quadro RTX 4000/5000/6000/8000, Tesla T4**

```bash
cmake --preset=cuda-arch-turing
cmake --build --preset=cuda-arch-turing
```

- Compute capability: 7.5
- ⚡ Optimized for Turing architecture

#### `cuda-arch-volta`
**For: Tesla V100, Titan V, Quadro GV100**

```bash
cmake --preset=cuda-arch-volta
cmake --build --preset=cuda-arch-volta
```

- Compute capability: 7.0
- ⚡ Optimized for Volta architecture

#### `cuda-arch-pascal`
**For: GTX 1050/1060/1070/1080, Tesla P100, Quadro P4000/P5000/P6000**

```bash
cmake --preset=cuda-arch-pascal
cmake --build --preset=cuda-arch-pascal
```

- Compute capability: 6.0, 6.1
- ⚡ Optimized for Pascal architecture

## 🎯 Common Use Cases

### "I have an NVIDIA GPU and want to run benchmarks quickly"

```bash
cmake --preset=cuda-only
cmake --build --preset=cuda-only -j$(nproc)
cmake --build build-cuda-only --target install
./install-cuda/bin/Serial/CUDA/MaxFlops
```

### "I have an AMD GPU"

```bash
cmake --preset=opencl-only
cmake --build --preset=opencl-only -j$(nproc)
cmake --build build-opencl-only --target install
./install-opencl/bin/Serial/OpenCL/MaxFlops
```

### "I have multiple GPUs and want MPI benchmarks"

```bash
cmake --preset=cuda-mpi
cmake --build --preset=cuda-mpi -j$(nproc)
cmake --build build-cuda-mpi --target install
mpirun -np 2 ./install-cuda-mpi/bin/EP/CUDA/BusSpeedDownload
```

### "I want to benchmark CUDA vs OpenCL on my system"

```bash
# Build both
cmake --preset=full
cmake --build --preset=full -j$(nproc)
cmake --build build-full --target install

# Run CUDA version
./install-full/bin/Serial/CUDA/FFT -s 2

# Run OpenCL version
./install-full/bin/Serial/OpenCL/FFT -s 2

# Compare results!
```

### "I'm developing SHOC and need to debug a crash"

```bash
cmake --preset=dev-debug
cmake --build --preset=dev-debug
gdb ./build-dev-debug/bin/Serial/CUDA/BFS
```

### "I want the fastest possible build for my RTX 3080"

```bash
cmake --preset=cuda-arch-ampere
cmake --build --preset=cuda-arch-ampere -j$(nproc)
# This only compiles for Ampere architecture (sm_86)
# Much faster than compiling for all architectures!
```

## 🔧 Advanced Usage

### Customize a Preset

You can override any cache variable:

```bash
# Use cuda-only preset but install elsewhere
cmake --preset=cuda-only \
  -DCMAKE_INSTALL_PREFIX=/opt/shoc

# Use cuda-only but add specific architecture
cmake --preset=cuda-only \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
```

### Build Multiple Configurations

Presets create separate build directories, so you can have multiple configurations:

```bash
# Build CUDA version
cmake --preset=cuda-only
cmake --build --preset=cuda-only

# Build OpenCL version (in parallel!)
cmake --preset=opencl-only
cmake --build --preset=opencl-only

# Both build directories coexist:
# - build-cuda-only/
# - build-opencl-only/
```

### Clean a Preset Build

```bash
# Remove the build directory
rm -rf build-cuda-only

# Reconfigure from scratch
cmake --preset=cuda-only
```

## 📊 Comparison: Old Way vs New Way

### Old Way (Manual CMake)
```bash
mkdir build
cd build
cmake .. \
  -DSHOC_BUILD_CUDA=ON \
  -DSHOC_BUILD_OPENCL=OFF \
  -DSHOC_BUILD_MPI=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80"
make -j$(nproc)
make install
cd ..
./install/bin/Serial/CUDA/MaxFlops
```

### New Way (Presets)
```bash
cmake --preset=cuda-only
cmake --build --preset=cuda-only -j
./install-cuda/bin/Serial/CUDA/MaxFlops
```

**Result: 80% fewer keystrokes, no mistakes!**

## 🆘 Troubleshooting

### "cmake: unrecognized option '--preset'"

You need CMake 3.23+. Check your version:

```bash
cmake --version

# Update if needed
# Ubuntu/Debian: use snap or build from source
# macOS: brew upgrade cmake
```

### "Preset 'xyz' not found"

Make sure you're in the SHOC root directory (where `CMakePresets.json` is located):

```bash
cd /path/to/shoc
cmake --list-presets
```

### "CUDA not found" when using cuda presets

Add CUDA to your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
# Then try again
cmake --preset=cuda-only
```

### "I want to see what a preset does"

Check the `CMakePresets.json` file - it's human-readable JSON showing exactly what each preset configures.

## 💡 Tips & Tricks

### 1. Use Tab Completion

If your shell supports it:
```bash
cmake --preset=cu<TAB>  # Autocompletes to cuda presets
```

### 2. IDE Integration

Modern IDEs (CLion, VSCode with CMake Tools) automatically detect presets and show them in a dropdown menu!

### 3. CI/CD

Presets are perfect for CI:
```yaml
# GitHub Actions example
- name: Build SHOC
  run: |
    cmake --preset=minimal
    cmake --build --preset=minimal
```

### 4. Documentation

The preset names are self-documenting. No need to remember flags!

## 📚 Further Reading

- [CMake Presets Documentation](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
- [SHOC Build Guide](CMAKE_BUILD.md)
- [SHOC Migration Plan](CMAKE_MIGRATION_PLAN.md)

---

**Questions or issues?** Check out the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide or file an issue.
