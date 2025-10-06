# Building SHOC with CMake

This document describes how to build the SHOC Benchmark Suite using CMake (Phase 1 implementation).

## Requirements

- CMake 3.18 or later (3.24+ recommended)
- C++ compiler with C++11 support
- At least one of:
  - CUDA Toolkit (for CUDA benchmarks)
  - OpenCL SDK (for OpenCL benchmarks)
- MPI implementation (optional, for parallel benchmarks)

## Quick Start

### Basic Build (In-Source)

```bash
cd shoc
mkdir build
cd build
cmake ..
make
make install
```

### Out-of-Source Build (Recommended)

```bash
cd shoc
mkdir ../shoc-build
cd ../shoc-build
cmake ../shoc
make
make install
```

## Configuration Options

CMake options can be set using `-D` flags:

```bash
cmake -DSHOC_BUILD_CUDA=ON -DSHOC_BUILD_OPENCL=OFF ..
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `SHOC_BUILD_CUDA` | ON | Build CUDA benchmarks |
| `SHOC_BUILD_OPENCL` | ON | Build OpenCL benchmarks |
| `SHOC_BUILD_MPI` | ON | Build MPI parallel benchmarks |
| `SHOC_BUILD_STABILITY` | ON | Build stability tests (requires CUDA) |
| `SHOC_BUILD_DOC` | OFF | Build documentation (requires pdflatex, bibtex, latexmk) |
| `CMAKE_INSTALL_PREFIX` | `${BUILD_DIR}` | Installation directory |
| `CMAKE_CUDA_ARCHITECTURES` | `50;60;70;75;80` | CUDA architectures to target |

## Example Configurations

### CUDA Only

```bash
cmake -DSHOC_BUILD_CUDA=ON \
      -DSHOC_BUILD_OPENCL=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/shoc \
      ..
make -j$(nproc)
make install
```

### OpenCL Only

```bash
cmake -DSHOC_BUILD_CUDA=OFF \
      -DSHOC_BUILD_OPENCL=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/shoc \
      ..
make -j$(nproc)
make install
```

### Both CUDA and OpenCL with MPI

```bash
cmake -DSHOC_BUILD_CUDA=ON \
      -DSHOC_BUILD_OPENCL=ON \
      -DSHOC_BUILD_MPI=ON \
      -DCMAKE_INSTALL_PREFIX=$HOME/shoc \
      ..
make -j$(nproc)
make install
```

### Specific CUDA Architectures

For faster compilation or specific GPU targets:

```bash
# For a single GPU architecture (e.g., RTX 3090, compute capability 8.6)
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..

# For multiple architectures
cmake -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" ..

# Let CMake detect your GPU (CMake 3.24+)
cmake -DCMAKE_CUDA_ARCHITECTURES=native ..
```

## Installation Structure

After `make install`, benchmarks are organized as:

```
${CMAKE_INSTALL_PREFIX}/
├── bin/
│   ├── Serial/
│   │   ├── CUDA/          # Serial CUDA benchmarks
│   │   └── OpenCL/        # Serial OpenCL benchmarks
│   ├── EP/                # Embarrassingly Parallel (MPI)
│   │   ├── CUDA/
│   │   └── OpenCL/
│   ├── TP/                # Truly Parallel (MPI)
│   │   ├── CUDA/
│   │   └── OpenCL/
│   └── shocdriver         # Driver script
├── lib/
│   ├── libSHOCCommon.a
│   └── libSHOCCommonOpenCL.a
└── doc/
    └── shoc-manual.pdf
```

## Running Benchmarks

### Serial Benchmarks

```bash
# CUDA
./bin/Serial/CUDA/BusSpeedDownload
./bin/Serial/CUDA/MaxFlops

# OpenCL
./bin/Serial/OpenCL/Sort
```

### MPI Benchmarks

```bash
# EP (Embarrassingly Parallel) - 1 rank per device
mpirun -np 2 ./bin/EP/CUDA/BusSpeedDownload

# TP (Truly Parallel) - devices communicate
mpirun -np 4 ./bin/TP/CUDA/Reduction
```

### Using shocdriver

```bash
# CUDA benchmarks, size 2
./bin/shocdriver -s 2 -cuda

# OpenCL benchmarks, size 2
./bin/shocdriver -s 2 -opencl

# Parallel benchmarks on 4 nodes with 2 devices per node
./bin/shocdriver -n 4 -d 2 -s 1 -cuda
```

## Troubleshooting

### CUDA Not Found

Ensure `nvcc` is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Or specify CUDA location:
```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

### OpenCL Not Found

Set OpenCL paths:
```bash
cmake -DOpenCL_INCLUDE_DIR=/path/to/opencl/include \
      -DOpenCL_LIBRARY=/path/to/libOpenCL.so \
      ..
```

### MPI Not Found

Ensure MPI is in PATH:
```bash
export PATH=/usr/local/mpi/bin:$PATH
```

Or specify MPI compiler:
```bash
cmake -DMPI_CXX_COMPILER=/usr/local/mpi/bin/mpicxx ..
```

## Phase 1 Status

The current CMake implementation (Phase 1) includes:

✅ **Implemented:**
- Root CMake infrastructure with options
- Common libraries (SHOCCommon, SHOCCommonOpenCL)
- CUDA Level 0 benchmarks (Serial and EP/MPI)
- OpenCL Sort benchmark (Serial)
- OpenCL kernel embedding system
- Utility functions for adding benchmarks

⚠️ **Placeholders (Phase 2):**
- Most CUDA Level 1 and Level 2 benchmarks
- Most OpenCL Level 1 and Level 2 benchmarks
- TP (Truly Parallel) MPI benchmarks
- MPI contention tests
- Stability tests
- Documentation build

## Comparison with Autotools

| Feature | Autotools | CMake |
|---------|-----------|-------|
| Minimum version | autoconf 2.65 | CMake 3.18 |
| Configure | `./configure --with-cuda` | `cmake -DSHOC_BUILD_CUDA=ON` |
| Out-of-tree builds | Supported | Recommended |
| IDE support | Limited | Excellent (CLion, VS, etc.) |
| Windows support | Cygwin only | Native |
| CUDA detection | Custom scripts | Built-in (CMake 3.8+) |
| OpenCL detection | Custom scripts | FindOpenCL module |
| Installation prefix | `--prefix=DIR` | `-DCMAKE_INSTALL_PREFIX=DIR` |

## Migration from Autotools

If you were using autotools, here's the equivalent CMake configuration:

```bash
# Autotools
./configure --with-cuda --with-opencl --with-mpi \
            --prefix=/opt/shoc \
            CUDA_CPPFLAGS="-gencode=arch=compute_70,code=sm_70"

# CMake equivalent
cmake -DSHOC_BUILD_CUDA=ON \
      -DSHOC_BUILD_OPENCL=ON \
      -DSHOC_BUILD_MPI=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/shoc \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      ..
```

## Contributing

To add a new benchmark to the CMake build:

1. Create a `CMakeLists.txt` in the benchmark directory
2. Use the helper functions from `cmake/SHOCUtilities.cmake`:
   - `shoc_add_cuda_serial_executable()`
   - `shoc_add_opencl_serial_executable()`
   - `shoc_add_cuda_ep_executable()` (for MPI EP)
   - `shoc_add_opencl_ep_executable()` (for MPI EP)
3. For OpenCL kernels, use `shoc_embed_opencl_kernel()`
4. Add the subdirectory to the parent `CMakeLists.txt`

See [CMAKE_MIGRATION_PLAN.md](CMAKE_MIGRATION_PLAN.md) for details.

## Support

For issues specific to the CMake build system, please check:
- [CMAKE_MIGRATION_PLAN.md](CMAKE_MIGRATION_PLAN.md) - Migration strategy
- SHOC user manual in `doc/` directory
- SHOC GitHub repository

For general SHOC questions, see README.txt and INSTALL.txt.
