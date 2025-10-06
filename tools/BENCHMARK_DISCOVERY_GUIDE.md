# SHOC Benchmark Discovery Tool Guide

The `list_benchmarks.py` tool helps you discover what benchmarks are available in your SHOC installation.

## Quick Start

```bash
# Auto-detect installation and list all benchmarks
./tools/list_benchmarks.py

# Specify installation directory
./tools/list_benchmarks.py install-cuda
```

## Features

### 1. **Automatic Discovery**
Scans your SHOC installation and finds all benchmark executables across:
- Backends: CUDA, OpenCL
- Categories: Serial, EP (Embarrassingly Parallel), TP (Truly Parallel)

### 2. **Rich Descriptions**
Every benchmark includes a description:
```
CUDA
----
  Serial:
    • BusSpeedDownload    Measures PCIe bus bandwidth from host to device
    • FFT                 Fast Fourier Transform (1D, forward and inverse)
    • GEMM                General Matrix Multiply (dense linear algebra)
```

### 3. **Filtering Options**

Filter by backend:
```bash
./tools/list_benchmarks.py --backend cuda
./tools/list_benchmarks.py --backend opencl
```

Filter by category:
```bash
./tools/list_benchmarks.py --category Serial
./tools/list_benchmarks.py --category EP
./tools/list_benchmarks.py --category TP
```

Search by keyword:
```bash
./tools/list_benchmarks.py --search fft
./tools/list_benchmarks.py --search memory
./tools/list_benchmarks.py --search matrix
```

Combine filters:
```bash
./tools/list_benchmarks.py --backend cuda --category Serial --search bandwidth
```

### 4. **Multiple Output Formats**

**Human-readable (default)** - Colored, organized by backend and category:
```bash
./tools/list_benchmarks.py
```

**Markdown table** - Perfect for documentation:
```bash
./tools/list_benchmarks.py --format markdown > BENCHMARKS.md
```

**JSON** - For scripting and automation:
```bash
./tools/list_benchmarks.py --format json > benchmarks.json
```

### 5. **Statistics**

Quick stats mode:
```bash
./tools/list_benchmarks.py --stats-only
# Output:
# Total: 24
# Backends: {'CUDA': 18, 'OpenCL': 6}
# Categories: {'Serial': 20, 'EP': 3, 'TP': 1}
```

## Common Use Cases

### New User: Explore What's Available
```bash
# See everything with descriptions
./tools/list_benchmarks.py

# Focus on CUDA benchmarks
./tools/list_benchmarks.py --backend cuda
```

### Find Specific Tests
```bash
# Find all FFT-related benchmarks
./tools/list_benchmarks.py --search fft

# Find memory bandwidth tests
./tools/list_benchmarks.py --search bandwidth

# Find graph algorithm tests
./tools/list_benchmarks.py --search graph
```

### Generate Documentation
```bash
# Create a Markdown table of all benchmarks
./tools/list_benchmarks.py --format markdown > docs/AVAILABLE_BENCHMARKS.md

# Create JSON metadata for your website
./tools/list_benchmarks.py --format json > website/benchmarks.json
```

### CI/CD Integration
```bash
# Get list of all Serial CUDA benchmarks as JSON
./tools/list_benchmarks.py --backend cuda --category Serial --format json > tests.json

# Parse in your CI script
python3 << EOF
import json
with open('tests.json') as f:
    data = json.load(f)
    for bench in data['benchmarks']:
        print(f"Running {bench['path']}...")
EOF
```

### Compare Installations
```bash
# Compare CUDA-only vs full installation
./tools/list_benchmarks.py install-cuda --stats-only
./tools/list_benchmarks.py install-full --stats-only
```

## Benchmark Levels

SHOC organizes benchmarks into three levels:

### Level 0: Device Capabilities
Measure fundamental device characteristics:
- **BusSpeedDownload/Readback** - PCIe bandwidth
- **DeviceMemory** - Memory bandwidth and latency
- **MaxFlops** - Peak floating-point performance
- **QueueDelay** - Command queue latency

### Level 1: Algorithm Performance
Common parallel algorithms:
- **BFS** - Graph traversal
- **FFT** - Signal processing
- **GEMM** - Dense linear algebra
- **MD** - N-body simulation
- **Reduction/Scan** - Parallel primitives
- **Sort** - Sorting algorithms
- **Spmv** - Sparse linear algebra
- **Stencil2D** - Structured grid computation
- **Triad** - Memory bandwidth (STREAM)

### Level 2: Application Kernels
Real scientific application kernels:
- **S3D** - Combustion chemistry
- **QTC** - Quantum chemistry

## Auto-Detection

The tool automatically searches these locations:
1. `./install`
2. `./install-full`
3. `./install-cuda`
4. `./install-opencl`
5. `../install`
6. `/usr/local/shoc`
7. `~/shoc/install`

If your installation is elsewhere, provide the path explicitly:
```bash
./tools/list_benchmarks.py /path/to/my/shoc/installation
```

## Example Output

### Human-Readable Format
```
================================================================================
SHOC Benchmarks - /home/user/shoc/install-full
================================================================================

CUDA
----

  Serial:
    • BusSpeedDownload    Measures PCIe bus bandwidth from host to device
    • BusSpeedReadback    Measures PCIe bus bandwidth from device to host
    • DeviceMemory        Tests device memory bandwidth and latency
    • FFT                 Fast Fourier Transform (1D, forward and inverse)
    ...

  EP:
    • BusSpeedDownload    Measures PCIe bus bandwidth from host to device
    ...

OpenCL
------

  Serial:
    • BusSpeedDownload    Measures PCIe bus bandwidth from host to device
    ...

================================================================================
Statistics:
  Total benchmarks: 75
  By backend: CUDA=50, OpenCL=25
  By category: EP=25, Serial=40, TP=10
================================================================================
```

### Markdown Format
```markdown
# SHOC Benchmarks - /home/user/shoc/install-full

| Benchmark | Backend | Category | Description |
|-----------|---------|----------|-------------|
| BFS | CUDA | Serial | Breadth-First Search graph traversal algorithm |
| FFT | CUDA | Serial | Fast Fourier Transform (1D, forward and inverse) |
...

**Total: 75 benchmarks**
```

### JSON Format
```json
{
  "install_dir": "/home/user/shoc/install-full",
  "total_count": 75,
  "benchmarks": [
    {
      "name": "BFS",
      "backend": "CUDA",
      "category": "Serial",
      "path": "/home/user/shoc/install-full/bin/Serial/CUDA/BFS",
      "description": "Breadth-First Search graph traversal algorithm"
    },
    ...
  ]
}
```

## Tips

1. **Pipe to less** for easier browsing of long lists:
   ```bash
   ./tools/list_benchmarks.py | less -R
   ```

2. **Count benchmarks** by backend:
   ```bash
   ./tools/list_benchmarks.py --backend cuda --stats-only
   ```

3. **Create a simple benchmark list** for documentation:
   ```bash
   ./tools/list_benchmarks.py --format markdown | grep "^|" > simple_list.txt
   ```

4. **Check if a specific benchmark exists**:
   ```bash
   ./tools/list_benchmarks.py --search "GEMM" --format json | jq '.total_count'
   ```

## Extending

To add descriptions for new benchmarks, edit the `BENCHMARK_DESCRIPTIONS` dictionary in `list_benchmarks.py`:

```python
BENCHMARK_DESCRIPTIONS = {
    'MyNewBenchmark': 'Description of what it does',
    ...
}
```

## Related Tools

- **validate_install.sh** - Verify installation is correct
- **CMake Presets** - Configure and build SHOC
- **QUICKSTART.md** - Get started with SHOC
