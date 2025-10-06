# SHOC Driver Guide

Modern Python replacement for the legacy Perl `driver.pl` script.

## What It Does

The SHOC Driver:
- Runs all SHOC benchmarks (or a subset)
- Supports CUDA and OpenCL backends
- Handles single GPU, multi-GPU, and multi-node (MPI) configurations
- Parses benchmark output and extracts results
- Generates CSV reports and optional JSON reports
- Provides progress bars and colored output
- Supports parallel benchmark execution to save time

## Quick Start

### Command Line Usage

**Run CUDA benchmarks (size class 2):**
```bash
./tools/shocdriver.py --backend cuda --size 2
```

**Run OpenCL benchmarks:**
```bash
./tools/shocdriver.py --backend opencl --size 2
```

**Run single benchmark:**
```bash
./tools/shocdriver.py --backend cuda --size 1 --benchmark FFT
```

**Multi-GPU:**
```bash
./tools/shocdriver.py --backend cuda --size 2 --device 0,1
```

**Fast parallel execution:**
```bash
./tools/shocdriver.py --backend cuda --size 2 --parallel --max-workers 8
```

### Configuration File Usage

**Using YAML config:**
```bash
./tools/shocdriver.py --config config/examples/cuda-quick.yaml
```

**Using JSON config:**
```bash
./tools/shocdriver.py --config config/examples/baseline.json
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--config FILE` | Load configuration from YAML/JSON file |
| `--backend {cuda,opencl}` | Backend to use (required if not using --config) |
| `--size {1,2,3,4}` | Problem size class (required if not using --config) |
| `--device IDS` | Device IDs, comma-separated (default: 0) |
| `--platform ID` | OpenCL platform ID (default: 0) |
| `--num-nodes N` | Number of nodes for MPI runs (default: 1) |
| `--hostfile FILE` | MPI hostfile for multi-node runs |
| `--benchmark NAME` | Run only specific benchmark (e.g., FFT) |
| `--bin-dir DIR` | SHOC binary directory (default: ./bin) |
| `--log-dir DIR` | Log output directory (default: ./Logs) |
| `--output FILE` | CSV output file (default: ./results.csv) |
| `--json-report FILE` | Save detailed JSON report |
| `--parallel` | Run benchmarks in parallel (faster) |
| `--max-workers N` | Max parallel workers (default: 4) |

## Configuration Files

Configuration files (YAML or JSON) provide a cleaner way to specify complex configurations.

### YAML Example

```yaml
# config/my-experiment.yaml
backend: cuda
size: 3
devices: "0,1"
num_nodes: 1
bin_dir: ./install-cuda/bin
log_dir: ./Logs/experiment-1
parallel_execution: true
max_workers: 8
```

Run with:
```bash
./tools/shocdriver.py --config config/my-experiment.yaml
```

### JSON Example

```json
{
  "backend": "opencl",
  "size": 2,
  "devices": "0",
  "platform": "0",
  "num_nodes": 1,
  "bin_dir": "./install-opencl/bin",
  "log_dir": "./Logs/baseline",
  "parallel_execution": false
}
```

### Available Example Configs

Located in [`config/examples/`](../config/examples/):

- **cuda-quick.yaml** - Quick CUDA test (size 1, single GPU)
- **cuda-full.yaml** - Full CUDA suite (size 3, parallel execution)
- **opencl-multi-device.yaml** - OpenCL with multiple GPUs
- **mpi-cluster.yaml** - Multi-node MPI cluster configuration
- **baseline.json** - JSON format baseline configuration

## Size Classes

SHOC provides 4 size classes for problem sizes:

| Size Class | Description | Use Case |
|------------|-------------|----------|
| 1 | Tiny | Quick testing, debugging |
| 2 | Small | Standard testing |
| 3 | Medium | Performance evaluation |
| 4 | Large | Large-scale benchmarking |

## Output Files

### CSV Results (`results.csv`)

Default output file containing benchmark results in CSV format:

```csv
maxspflops,maxdpflops,fft_sp,fft_dp,gemm_n,...
1234.56,567.89,890.12,345.67,...
```

Specify custom output:
```bash
./tools/shocdriver.py --backend cuda --size 2 --output my-results.csv
```

### JSON Report (optional)

Detailed JSON report with configuration, results, log files, and timestamps:

```bash
./tools/shocdriver.py --backend cuda --size 2 --json-report report.json
```

Example JSON output:
```json
{
  "timestamp": "2025-10-06T14:30:00",
  "config": {
    "backend": "cuda",
    "size": 2,
    ...
  },
  "results": [
    {
      "benchmark": "FFT",
      "status": "success",
      "runtime": 12.34,
      "results": {
        "fft_sp": [890.12, "GFLOPS"],
        "fft_dp": [345.67, "GFLOPS"]
      },
      "log_file": "./Logs/dev0_FFT.log"
    },
    ...
  ]
}
```

### Log Files

Individual log files for each benchmark in `Logs/` (or custom directory):

- `dev0_BenchmarkName.log` - Benchmark output
- `dev0_BenchmarkName.err` - Error output

## Parallel Execution

Speed up benchmark runs by executing multiple benchmarks concurrently:

```bash
./tools/shocdriver.py --backend cuda --size 2 --parallel --max-workers 8
```

**Benefits:**
- Significantly faster for large benchmark suites
- Utilizes multiple CPU cores
- Safe for independent benchmarks

**Caution:**
- Don't use `--parallel` with MPI multi-node runs (use only for serial/single-node)
- Adjust `--max-workers` based on available CPU cores and GPU contention

## Common Use Cases

### 1. Quick Sanity Check

Test that everything works with small problem size:

```bash
./tools/shocdriver.py --backend cuda --size 1
```

### 2. Performance Baseline

Establish baseline performance with standard size:

```bash
./tools/shocdriver.py --backend cuda --size 2 \
  --output baseline.csv \
  --json-report baseline.json
```

### 3. Debug Single Benchmark

Focus on one benchmark for debugging:

```bash
./tools/shocdriver.py --backend cuda --size 1 --benchmark FFT
```

### 4. Multi-GPU Testing

Test with multiple GPUs:

```bash
./tools/shocdriver.py --backend cuda --size 2 --device 0,1,2,3
```

### 5. Cluster Run

Multi-node MPI cluster:

```bash
# Create hostfile
cat > hostfile.txt << EOF
node01
node02
node03
node04
EOF

# Run with config
./tools/shocdriver.py --config config/examples/mpi-cluster.yaml
```

### 6. Automated Testing

Run regularly with config file for reproducibility:

```bash
#!/bin/bash
# nightly-test.sh
./tools/shocdriver.py --config config/nightly.yaml \
  --output "results-$(date +%Y%m%d).csv" \
  --json-report "report-$(date +%Y%m%d).json"
```

### 7. Compare Backends

Compare CUDA vs OpenCL:

```bash
./tools/shocdriver.py --backend cuda --size 2 --output cuda-results.csv
./tools/shocdriver.py --backend opencl --size 2 --output opencl-results.csv
# Then compare CSV files
```

## Dependencies

**Required:**
- Python 3.6+
- PyYAML (for YAML configs): `pip install pyyaml`

**Optional:**
- tqdm (progress bars): `pip install tqdm`

Install all dependencies:
```bash
pip install pyyaml tqdm
```

## Progress Output

With `tqdm` installed:
```
Running benchmarks: 45%|████████▌         | 9/20 [02:15<02:30, 0.07bench/s]
```

Without `tqdm`:
```
Running benchmark BusSpeedDownload
Running benchmark MaxFlops
...
```

## Result Extraction

The driver automatically parses SHOC's ResultDatabase output format and extracts:

- **Maximum values** - For throughput metrics (GFLOPS, GB/s)
- **Minimum values** - For latency metrics (ms, compile time)
- **Mean values** - For parallel runs (average across MPI ranks)

Extraction strategies are defined per-benchmark based on metric type.

## Comparison with Legacy driver.pl

| Feature | driver.pl (Perl) | shocdriver.py (Python) |
|---------|------------------|------------------------|
| Language | Perl | Python 3 |
| Config | Command-line only | YAML/JSON + CLI |
| Parallel execution | No | Yes (--parallel) |
| Progress bars | No | Yes (tqdm) |
| Colored output | No | Yes |
| JSON reports | No | Yes |
| Error handling | Basic | Enhanced |
| Cross-platform | Unix-like | Windows/Mac/Linux |
| Timeout handling | No | Yes (600s default) |
| Concurrent benchmarks | No | Yes |

## Troubleshooting

### "Backend and size required"

You must specify `--backend` and `--size` or use `--config`:

```bash
# Wrong
./tools/shocdriver.py

# Correct
./tools/shocdriver.py --backend cuda --size 2
```

### "Binary directory not found"

The `bin_dir` path is incorrect. Either:
- Run from SHOC install root, or
- Use `--bin-dir /path/to/install/bin`

### "SHOC benchmarks not found"

SHOC hasn't been built/installed. Run CMake build first:

```bash
cmake --build build --target install
```

### Benchmarks timeout

Increase timeout in code or reduce problem size with `--size 1`.

### ImportError: No module named yaml

Install PyYAML:

```bash
pip install pyyaml
```

### MPI errors

For multi-node runs:
- Ensure MPI is properly configured
- Check hostfile format
- Verify SSH access to all nodes
- Test with simple `mpirun hostname` first

## Advanced: Custom Configurations

Create custom configs for specific scenarios:

**GPU Architecture Comparison:**
```yaml
# config/a100-vs-v100.yaml
backend: cuda
size: 3
devices: "0"  # Run same test on different machines
bin_dir: ./install-cuda/bin
log_dir: ./Logs/gpu-comparison
parallel_execution: true
max_workers: 8
```

Run on different systems:
```bash
# On A100 system
./tools/shocdriver.py --config config/a100-vs-v100.yaml --output a100.csv

# On V100 system
./tools/shocdriver.py --config config/a100-vs-v100.yaml --output v100.csv
```

**Regression Testing:**
```yaml
# config/regression.yaml
backend: cuda
size: 2
devices: "0"
bin_dir: ./install-cuda/bin
log_dir: ./Logs/regression
parallel_execution: true
max_workers: 4
```

Integrate into CI:
```bash
#!/bin/bash
# ci-test.sh
./tools/shocdriver.py --config config/regression.yaml \
  --output current-results.csv

# Compare with baseline
python scripts/compare_results.py baseline.csv current-results.csv
```

## Integration with Other Tools

**With validate_install.sh:**
```bash
# First validate installation
./tools/validate_install.sh install-cuda

# Then run benchmarks
./tools/shocdriver.py --backend cuda --size 2
```

**With list_benchmarks.py:**
```bash
# Discover what's available
./tools/list_benchmarks.py install-cuda

# Run specific benchmark
./tools/shocdriver.py --backend cuda --size 2 --benchmark FFT
```

**With CMake Presets:**
```bash
# Build with preset
cmake --preset cuda-only
cmake --build build-cuda-only --target install

# Run benchmarks
./tools/shocdriver.py --backend cuda --size 2 --bin-dir install-cuda/bin
```

## Related Documentation

- [QUICKSTART.md](../QUICKSTART.md) - Getting started guide
- [CMAKE_PRESETS_GUIDE.md](../CMAKE_PRESETS_GUIDE.md) - CMake configuration
- [BENCHMARK_DISCOVERY_GUIDE.md](./BENCHMARK_DISCOVERY_GUIDE.md) - List benchmarks
- [RECOMMENDATIONS.md](../RECOMMENDATIONS.md) - All improvement recommendations

## Tips

1. **Start small**: Use `--size 1` for initial testing
2. **Use configs**: Create YAML files for reproducible experiments
3. **Enable parallel**: Use `--parallel` to save time on large runs
4. **JSON reports**: Use `--json-report` for detailed analysis
5. **Single benchmark**: Use `--benchmark` when debugging
6. **Log organization**: Use custom `--log-dir` for different experiments
7. **Automation**: Script regular runs with cron/CI using config files
