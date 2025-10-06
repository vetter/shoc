# SHOC CUDA benchmarks

# CUDA common library is just headers, no compiled library needed
# The main.cpp is linked into each benchmark

# Add benchmark levels
add_subdirectory(level0)
add_subdirectory(level1)
add_subdirectory(level2)
