# SHOC CMake Migration - Phase 1 Summary

## Completed Work

Phase 1 of the CMake migration has been completed. The new CMake build system is now in place alongside the existing autotools build system.

### Files Created

#### Root Configuration
- **CMakeLists.txt** - Main project configuration with options and framework detection
- **CMAKE_BUILD.md** - Complete build documentation for CMake users
- **CMAKE_MIGRATION_PLAN.md** - Detailed migration strategy and implementation plan

#### CMake Modules (cmake/)
- **config.h.in** - Configuration header template
- **EmbedOpenCLKernel.cmake** - Script to embed OpenCL kernels as C++ strings
- **SHOCUtilities.cmake** - Utility functions for adding benchmarks consistently

#### Source Tree (src/)
- **src/CMakeLists.txt** - Top-level source directory configuration
- **src/common/CMakeLists.txt** - SHOCCommon library (core utilities)
- **src/opencl/CMakeLists.txt** - OpenCL benchmark tree root
- **src/opencl/common/CMakeLists.txt** - SHOCCommonOpenCL library
- **src/opencl/level0/CMakeLists.txt** - OpenCL Level 0 (placeholder)
- **src/opencl/level1/CMakeLists.txt** - OpenCL Level 1 directory
- **src/opencl/level1/sort/CMakeLists.txt** - **Complete OpenCL Sort benchmark** ✅
- **src/opencl/level2/CMakeLists.txt** - OpenCL Level 2 (placeholder)
- **src/cuda/CMakeLists.txt** - CUDA benchmark tree root
- **src/cuda/level0/CMakeLists.txt** - **Complete CUDA Level 0 Serial benchmarks** ✅
- **src/cuda/level0/epmpi/CMakeLists.txt** - **Complete CUDA Level 0 EP/MPI benchmarks** ✅
- **src/cuda/level1/CMakeLists.txt** - CUDA Level 1 directory
- **src/cuda/level1/*/CMakeLists.txt** - Placeholders for 12 Level 1 benchmarks
- **src/cuda/level2/CMakeLists.txt** - CUDA Level 2 directory
- **src/cuda/level2/*/CMakeLists.txt** - Placeholders for 2 Level 2 benchmarks
- **src/mpi/CMakeLists.txt** - MPI contention tests (placeholder)
- **src/stability/CMakeLists.txt** - Stability tests (placeholder)

#### Other Directories
- **tools/CMakeLists.txt** - Tools installation
- **doc/CMakeLists.txt** - Documentation build/install

## Key Features Implemented

### 1. Modern CMake Practices
- Minimum CMake 3.18 (for native CUDA support)
- First-class CUDA language support (no deprecated FindCUDA module)
- OpenCL via imported targets (OpenCL::OpenCL)
- Target-based library linking and include propagation
- Generator expressions for build/install paths

### 2. Configuration Options
All autotools options have CMake equivalents:

| Autotools | CMake |
|-----------|-------|
| `--with-cuda` | `-DSHOC_BUILD_CUDA=ON` |
| `--with-opencl` | `-DSHOC_BUILD_OPENCL=ON` |
| `--with-mpi` | `-DSHOC_BUILD_MPI=ON` |
| `--enable-builddoc` | `-DSHOC_BUILD_DOC=ON` |
| `--disable-stability` | `-DSHOC_BUILD_STABILITY=OFF` |
| `--prefix=DIR` | `-DCMAKE_INSTALL_PREFIX=DIR` |
| `CUDA_CPPFLAGS` | `-DCMAKE_CUDA_ARCHITECTURES` |

### 3. Installation Structure
Preserves the existing bin/ structure:
```
bin/
├── Serial/
│   ├── CUDA/
│   └── OpenCL/
├── EP/
│   ├── CUDA/
│   └── OpenCL/
├── TP/
│   ├── CUDA/
│   └── OpenCL/
└── shocdriver
```

### 4. Utility Functions
Helper functions in `SHOCUtilities.cmake`:
- `shoc_add_cuda_serial_executable()` - CUDA serial benchmarks
- `shoc_add_opencl_serial_executable()` - OpenCL serial benchmarks
- `shoc_add_cuda_ep_executable()` - CUDA EP/MPI benchmarks
- `shoc_add_opencl_ep_executable()` - OpenCL EP/MPI benchmarks
- `shoc_add_cuda_tp_executable()` - CUDA TP/MPI benchmarks (for Phase 2)
- `shoc_add_opencl_tp_executable()` - OpenCL TP/MPI benchmarks (for Phase 2)
- `shoc_embed_opencl_kernel()` - Embed .cl files as C++ strings

### 5. OpenCL Kernel Embedding
Replicated the autotools sed-based workflow:
- Input: `.cl` file
- Output: `_cl.cpp` file with embedded kernel string
- Custom CMake script for portability
- Automatic dependency tracking

## Fully Implemented Benchmarks

The following benchmarks are **complete and buildable** in Phase 1:

### CUDA Serial (bin/Serial/CUDA/)
- BusSpeedDownload
- BusSpeedReadback
- DeviceMemory
- MaxFlops

### CUDA EP/MPI (bin/EP/CUDA/)
- BusSpeedDownload
- BusSpeedReadback
- DeviceMemory
- MaxFlops

### OpenCL Serial (bin/Serial/OpenCL/)
- Sort

## Phase 2 Next Steps

To complete the migration, the following work remains:

### Immediate Priorities
1. **Complete remaining CUDA Level 1 benchmarks** (12 benchmarks)
   - BFS, FFT, GEMM, MD, MD5Hash, NeuralNet, Reduction, Scan, Sort, SpMV, Stencil2D, Triad
   - Each needs:
     - Identify source files (.cu, .cpp)
     - Create CMakeLists.txt using utility functions
     - Add EP/MPI versions where applicable
     - Add TP/MPI versions where applicable

2. **Complete remaining OpenCL Level 1 benchmarks** (similar to Sort)
   - BFS, FFT, GEMM, MD, MD5Hash, Reduction, Scan, SpMV, Stencil2D, Triad
   - Embed OpenCL kernels (.cl files)
   - Add EP/MPI versions

3. **Level 2 benchmarks**
   - CUDA: S3D, QTClustering
   - OpenCL: S3D

4. **MPI contention tests**
   - src/mpi/contention/
   - src/mpi/contention-mt/

5. **Stability tests**
   - src/stability/

### Documentation
- Add CMake instructions to SHOC user manual (LaTeX)
- Update INSTALL.txt to mention CMake option
- Create migration guide for existing users

### Testing & Validation
- Test builds on multiple platforms (Linux, macOS, Windows)
- Verify output matches autotools builds
- Test various configuration combinations
- Benchmark performance comparison

## How to Use (Quick Start)

```bash
cd shoc
mkdir build
cd build

# Configure (auto-detect CUDA/OpenCL)
cmake ..

# Or configure with specific options
cmake -DSHOC_BUILD_CUDA=ON \
      -DSHOC_BUILD_OPENCL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_INSTALL_PREFIX=$HOME/shoc \
      ..

# Build
make -j$(nproc)

# Install
make install

# Run benchmarks
./bin/Serial/CUDA/BusSpeedDownload
./bin/Serial/OpenCL/Sort
mpirun -np 2 ./bin/EP/CUDA/MaxFlops
```

See **CMAKE_BUILD.md** for complete documentation.

## Adding New Benchmarks (Template)

For developers adding Phase 2 benchmarks, here's the pattern:

### CUDA Benchmark Example
```cmake
# src/cuda/level1/mybench/CMakeLists.txt
include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Serial version
shoc_add_cuda_serial_executable(MyBench
    SOURCES
        MyBench.cu
        MyBenchHelpers.cpp
        ${CUDA_COMMON_DIR}/main.cpp
)

# EP/MPI version
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
```

### OpenCL Benchmark Example
```cmake
# src/opencl/level1/mybench/CMakeLists.txt
include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed kernel
shoc_embed_opencl_kernel(
    mybench.cl
    ${CMAKE_CURRENT_BINARY_DIR}/mybench_cl.cpp
    mybench
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# Serial version
shoc_add_opencl_serial_executable(MyBench
    SOURCES
        MyBench.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/mybench_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# EP/MPI version
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
```

## Benefits Delivered

✅ Modern, maintainable build system
✅ Better IDE integration (CLion, VSCode, Visual Studio)
✅ Native Windows support (future)
✅ Cleaner dependency management
✅ Faster configuration (no autotools overhead)
✅ Better out-of-tree build support
✅ Automatic CUDA architecture detection
✅ Modern OpenCL package detection
✅ Reusable utility functions reduce duplication
✅ Comprehensive documentation

## Status

**Phase 1: COMPLETE** ✅

The foundation is in place. Users can now build:
- 4 CUDA Serial benchmarks
- 4 CUDA EP/MPI benchmarks
- 1 OpenCL Serial benchmark

The infrastructure is ready for rapid expansion in Phase 2.
