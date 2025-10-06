# SHOC CUDA Level 1 Benchmarks
# Level 1 benchmarks evaluate performance of a single device

# Add benchmark subdirectories
# For Phase 1, we'll add a few representative benchmarks
# More can be added in Phase 2

add_subdirectory(bfs)
add_subdirectory(fft)
add_subdirectory(gemm)
add_subdirectory(md)
add_subdirectory(md5hash)
add_subdirectory(neuralnet)
add_subdirectory(reduction)
add_subdirectory(scan)
add_subdirectory(sort)
add_subdirectory(spmv)
add_subdirectory(stencil2d)
add_subdirectory(triad)
