# SHOC CMake Migration Plan

## Current Build System Analysis

### Key Configuration Options from configure.ac:
- `--with-opencl`: Build OpenCL versions (default: check)
- `--with-cuda`: Build CUDA versions (default: check)
- `--with-mpi`: Build MPI-based parallel versions (default: check)
- `--enable-builddoc`: Build documentation (requires pdflatex, bibtex, latexmk)
- `--enable-data-unzip`: Unzip data files (default: enabled)
- `--disable-stability`: Disable CUDA stability tests
- `CUDA_CPPFLAGS`: Custom CUDA gencode flags for GPU architectures
- `NVCXXFLAGS`: NVCC compiler flags

### Project Structure:
- 3 benchmark levels: level0, level1, level2
- 3 parallelism categories: Serial, EP (Embarrassingly Parallel), TP (Truly Parallel)
- 2 compute frameworks: CUDA and OpenCL
- Shared libraries: libSHOCCommon, libSHOCCommonOpenCL
- 80+ Makefile.am files across deeply nested directories

## Recommended CMake Migration Strategy

### 1. Use Modern CMake (3.18+ minimum, 3.24+ recommended)
- CMake 3.18+ has mature CUDA language support
- CMake 3.24+ has improved FindOpenCL
- Avoid deprecated `FindCUDA` module entirely

### 2. Hierarchical CMakeLists.txt Structure
```
CMakeLists.txt (root)
├── src/CMakeLists.txt
│   ├── common/CMakeLists.txt
│   ├── cuda/CMakeLists.txt
│   │   ├── common/CMakeLists.txt
│   │   ├── level0/CMakeLists.txt
│   │   ├── level1/CMakeLists.txt (with benchmark subdirs)
│   │   └── level2/CMakeLists.txt
│   ├── opencl/CMakeLists.txt
│   │   ├── common/CMakeLists.txt
│   │   ├── level0/CMakeLists.txt
│   │   └── ...
│   ├── mpi/CMakeLists.txt
│   └── stability/CMakeLists.txt
├── tools/CMakeLists.txt
└── doc/CMakeLists.txt
```

### 3. Key CMake Features to Leverage

#### Option-based Configuration:
```cmake
option(SHOC_BUILD_CUDA "Build CUDA benchmarks" ON)
option(SHOC_BUILD_OPENCL "Build OpenCL benchmarks" ON)
option(SHOC_BUILD_MPI "Build MPI parallel benchmarks" ON)
option(SHOC_BUILD_STABILITY "Build stability tests" ON)
option(SHOC_BUILD_DOC "Build documentation" OFF)
```

#### Modern CUDA Support:
```cmake
# In root CMakeLists.txt
project(SHOC VERSION 1.1.5 LANGUAGES CXX)
if(SHOC_BUILD_CUDA)
    enable_language(CUDA)
endif()
```

#### OpenCL via Imported Targets:
```cmake
find_package(OpenCL)
target_link_libraries(myapp PRIVATE OpenCL::OpenCL)
```

#### MPI Detection:
```cmake
find_package(MPI COMPONENTS CXX)
target_link_libraries(myapp PRIVATE MPI::MPI_CXX)
```

### 4. Handle OpenCL Kernel Embedding
Current system converts `.cl` files to C++ string literals via sed. CMake approach:
```cmake
# Custom command to embed OpenCL kernels
function(embed_opencl_kernel input_cl output_cpp kernel_name)
    add_custom_command(
        OUTPUT ${output_cpp}
        COMMAND ${CMAKE_COMMAND} -P embed_cl.cmake ${input_cl} ${output_cpp} ${kernel_name}
        DEPENDS ${input_cl}
    )
endfunction()
```

### 5. Shared Library Strategy
Create interface/static libraries for common code:
```cmake
add_library(SHOCCommon STATIC
    Timer.cpp ResultDatabase.cpp ...)
target_include_directories(SHOCCommon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

add_library(SHOCCommonCUDA STATIC ...)
target_link_libraries(SHOCCommonCUDA PUBLIC SHOCCommon)
```

### 6. CUDA Architecture Handling
```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
endif()
```

### 7. Installation Layout Preservation
```cmake
# Maintain existing bin/Serial/CUDA structure
install(TARGETS BusSpeedDownload
    DESTINATION bin/Serial/CUDA)
install(TARGETS Sort_EP
    DESTINATION bin/EP/OpenCL)
```

### 8. Migration Benefits

#### Advantages over autotools:
- ✅ Better Windows support (current VS solution is separate)
- ✅ Integrated IDE support (CLion, VSCode, Visual Studio)
- ✅ Simpler out-of-tree builds
- ✅ Modern CUDA/OpenCL package detection
- ✅ Better dependency tracking
- ✅ Cleaner cross-compilation support
- ✅ More maintainable (~20-30 CMakeLists vs 80+ Makefile.am files)

#### Challenges:
- ⚠️ Need to carefully handle conditional MPI builds (epmpi/tpmpi subdirs)
- ⚠️ VPATH functionality needs CMake equivalents
- ⚠️ Custom CUDA_CPPFLAGS gencode handling
- ⚠️ OpenCL kernel embedding workflow

### 9. Recommended Migration Phases

#### Phase 1: Core infrastructure
- Root CMakeLists.txt with options
- Common library builds
- One benchmark from each type (CUDA Serial, OpenCL Serial, CUDA EP)

#### Phase 2: Expand coverage
- All Serial benchmarks
- All EP benchmarks
- Documentation build

#### Phase 3: Complete migration
- TP benchmarks
- Stability tests
- CI/testing integration

### 10. Compatibility Considerations
- Keep autotools in parallel initially (deprecate in next major release)
- Provide migration guide for users
- Maintain same installation directory structure
- Ensure `shocdriver` script compatibility

## Implementation Notes

### Modern CMake Best Practices (2025)
Based on research of current CMake ecosystem:

1. **CUDA**: Use first-class CUDA language support (CMake 3.8+), not FindCUDA module (deprecated since 3.10)
2. **OpenCL**: Use imported targets `OpenCL::OpenCL` (available since CMake 3.7)
3. **CUDA Toolkit Libraries**: Use `FindCUDAToolkit` module for finding CUDA libraries
4. **Usage Requirements**: Leverage modern CMake's target-based propagation of include directories, defines, and options

### Configuration Migration Map

| Autoconf Option | CMake Equivalent |
|----------------|------------------|
| `--with-opencl` | `SHOC_BUILD_OPENCL=ON/OFF` |
| `--with-cuda` | `SHOC_BUILD_CUDA=ON/OFF` |
| `--with-mpi` | `SHOC_BUILD_MPI=ON/OFF` |
| `--enable-builddoc` | `SHOC_BUILD_DOC=ON/OFF` |
| `--disable-stability` | `SHOC_BUILD_STABILITY=OFF` |
| `--prefix=DIR` | `CMAKE_INSTALL_PREFIX=DIR` |
| `CPPFLAGS` | `CMAKE_CXX_FLAGS` / target properties |
| `CUDA_CPPFLAGS` | `CMAKE_CUDA_ARCHITECTURES` |
| `NVCXXFLAGS` | `CMAKE_CUDA_FLAGS` |
| `MPICXX=compiler` | Auto-detected via FindMPI |

## Testing Strategy

After implementing each phase:
1. Build with CUDA only
2. Build with OpenCL only
3. Build with both CUDA and OpenCL
4. Build with MPI enabled
5. Test out-of-tree builds
6. Verify installation directory structure matches autotools output
7. Run sample benchmarks to ensure functionality

## Documentation Updates Needed

- Update INSTALL.txt with CMake instructions
- Update README.txt
- Update user manual (doc/shoc-manual.tex)
- Create CMAKE_BUILD.md with detailed CMake build instructions
