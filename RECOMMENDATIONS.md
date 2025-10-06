# SHOC Benchmark Suite - Recommendations for Improved Usability & Flexibility

Based on the comprehensive analysis during the CMake migration, here are recommendations to modernize and improve the SHOC benchmark suite.

## 🎯 High Priority Recommendations

### 1. Add CMake Presets (CMake 3.19+)

**Problem**: Users must remember complex CMake command-line options.

**Solution**: Create `CMakePresets.json` for common configurations.

```json
{
  "version": 3,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 19,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "cuda-only",
      "displayName": "CUDA Only Build",
      "description": "Build only CUDA benchmarks",
      "binaryDir": "${sourceDir}/build-cuda",
      "cacheVariables": {
        "SHOC_BUILD_CUDA": "ON",
        "SHOC_BUILD_OPENCL": "OFF",
        "CMAKE_BUILD_TYPE": "Release"
      }
    },
    {
      "name": "opencl-only",
      "displayName": "OpenCL Only Build",
      "description": "Build only OpenCL benchmarks",
      "binaryDir": "${sourceDir}/build-opencl",
      "cacheVariables": {
        "SHOC_BUILD_CUDA": "OFF",
        "SHOC_BUILD_OPENCL": "ON",
        "CMAKE_BUILD_TYPE": "Release"
      }
    },
    {
      "name": "full-mpi",
      "displayName": "Full Build with MPI",
      "description": "Build all benchmarks with MPI support",
      "binaryDir": "${sourceDir}/build-full",
      "cacheVariables": {
        "SHOC_BUILD_CUDA": "ON",
        "SHOC_BUILD_OPENCL": "ON",
        "SHOC_BUILD_MPI": "ON",
        "CMAKE_BUILD_TYPE": "Release"
      }
    },
    {
      "name": "dev-debug",
      "displayName": "Development Debug Build",
      "description": "Debug build for development",
      "binaryDir": "${sourceDir}/build-debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "cuda-only",
      "configurePreset": "cuda-only"
    },
    {
      "name": "opencl-only",
      "configurePreset": "opencl-only"
    },
    {
      "name": "full-mpi",
      "configurePreset": "full-mpi"
    }
  ]
}
```

**Usage**:
```bash
# List available presets
cmake --list-presets

# Configure with preset
cmake --preset=cuda-only

# Build
cmake --build --preset=cuda-only
```

### 2. Modernize shocdriver Script

**Problem**: The Perl-based `shocdriver` script is dated and not user-friendly.

**Solution**: Create a Python version with better UX.

**File**: `tools/shocdriver.py`
```python
#!/usr/bin/env python3
"""
SHOC Benchmark Driver - Modern Python version
Runs SHOC benchmarks and generates reports in multiple formats
"""

import argparse
import subprocess
import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class SHOCDriver:
    def __init__(self, install_dir: Path):
        self.install_dir = install_dir
        self.results = []

    def find_benchmarks(self, category: str, backend: str) -> List[Path]:
        """Find all benchmark executables for a category and backend."""
        bench_dir = self.install_dir / "bin" / category / backend
        if not bench_dir.exists():
            return []
        return [f for f in bench_dir.iterdir() if f.is_file() and f.stat().st_mode & 0o111]

    def run_benchmark(self, executable: Path, size: int, device: int = 0) -> Dict:
        """Run a single benchmark and capture results."""
        try:
            cmd = [str(executable), "-s", str(size), "-d", str(device)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return {
                "name": executable.name,
                "category": executable.parent.parent.name,
                "backend": executable.parent.name,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
        except subprocess.TimeoutExpired:
            return {"name": executable.name, "error": "timeout"}
        except Exception as e:
            return {"name": executable.name, "error": str(e)}

    def run_suite(self, backend: str, size: int, categories: List[str]):
        """Run full benchmark suite."""
        print(f"Running {backend} benchmarks (size {size})...")

        for category in categories:
            benchmarks = self.find_benchmarks(category, backend)
            print(f"\n{category}/{backend}: {len(benchmarks)} benchmarks")

            for bench in benchmarks:
                print(f"  Running {bench.name}...", end=" ", flush=True)
                result = self.run_benchmark(bench, size)
                self.results.append(result)

                if result.get("returncode") == 0:
                    print("✓")
                else:
                    print("✗")

    def export_json(self, output: Path):
        """Export results as JSON."""
        with output.open("w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results exported to {output}")

    def export_csv(self, output: Path):
        """Export results as CSV."""
        if not self.results:
            return

        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        print(f"Results exported to {output}")

    def export_markdown(self, output: Path):
        """Export results as Markdown report."""
        with output.open("w") as f:
            f.write(f"# SHOC Benchmark Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Group by category and backend
            by_backend = {}
            for r in self.results:
                key = f"{r.get('category', 'Unknown')}/{r.get('backend', 'Unknown')}"
                if key not in by_backend:
                    by_backend[key] = []
                by_backend[key].append(r)

            for backend, results in by_backend.items():
                f.write(f"\n## {backend}\n\n")
                f.write("| Benchmark | Status | Notes |\n")
                f.write("|-----------|--------|-------|\n")

                for r in results:
                    status = "✓ Pass" if r.get("returncode") == 0 else "✗ Fail"
                    error = r.get("error", "")
                    f.write(f"| {r['name']} | {status} | {error} |\n")

        print(f"Markdown report exported to {output}")

def main():
    parser = argparse.ArgumentParser(
        description="SHOC Benchmark Suite Driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --cuda -s 2                    # Run CUDA benchmarks, size 2
  %(prog)s --opencl -s 1 --output json    # Run OpenCL, export JSON
  %(prog)s --cuda --mpi -n 4 -d 2 -s 3    # Run MPI benchmarks
  %(prog)s --all --format md              # Run everything, Markdown report
        """
    )

    # Backend selection
    parser.add_argument("--cuda", action="store_true", help="Run CUDA benchmarks")
    parser.add_argument("--opencl", action="store_true", help="Run OpenCL benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all available benchmarks")

    # MPI options
    parser.add_argument("--mpi", action="store_true", help="Run MPI parallel benchmarks")
    parser.add_argument("-n", "--nodes", type=int, default=1, help="Number of MPI nodes")
    parser.add_argument("-d", "--devices", type=int, default=1, help="Devices per node")

    # Benchmark options
    parser.add_argument("-s", "--size", type=int, default=1, choices=[1,2,3,4],
                        help="Problem size (1-4)")
    parser.add_argument("--device", type=int, default=0, help="Device ID to use")

    # Output options
    parser.add_argument("--format", choices=["csv", "json", "md"], default="csv",
                        help="Output format")
    parser.add_argument("-o", "--output", type=Path, help="Output file")
    parser.add_argument("--install-dir", type=Path, default=Path.cwd(),
                        help="SHOC installation directory")

    args = parser.parse_args()

    # Validate
    if not args.cuda and not args.opencl and not args.all:
        parser.error("Must specify --cuda, --opencl, or --all")

    if not args.install_dir.exists():
        print(f"Error: Installation directory not found: {args.install_dir}")
        sys.exit(1)

    # Determine output file
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"shoc_results_{timestamp}.{args.format}")

    # Run benchmarks
    driver = SHOCDriver(args.install_dir)

    categories = ["Serial"]
    if args.mpi:
        categories.extend(["EP", "TP"])

    backends = []
    if args.cuda or args.all:
        backends.append("CUDA")
    if args.opencl or args.all:
        backends.append("OpenCL")

    for backend in backends:
        driver.run_suite(backend, args.size, categories)

    # Export results
    if args.format == "json":
        driver.export_json(args.output)
    elif args.format == "csv":
        driver.export_csv(args.output)
    elif args.format == "md":
        driver.export_markdown(args.output)

    print(f"\n✓ Completed {len(driver.results)} benchmark runs")

if __name__ == "__main__":
    main()
```

**Benefits**:
- Modern Python 3 (widely available)
- JSON/CSV/Markdown output formats
- Better error handling
- Progress indication
- Timestamp tracking
- Easy to extend

### 3. Add Result Visualization

**Problem**: CSV output is hard to interpret.

**Solution**: Create visualization tools.

**File**: `tools/visualize_results.py`
```python
#!/usr/bin/env python3
"""
SHOC Results Visualizer
Creates plots from benchmark results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def plot_performance_comparison(data: pd.DataFrame, output: Path):
    """Create performance comparison plots."""
    sns.set_style("whitegrid")

    # Group by backend
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CUDA benchmarks
    cuda_data = data[data['backend'] == 'CUDA']
    if not cuda_data.empty:
        cuda_data.plot(x='name', y='time', kind='bar', ax=axes[0], color='green')
        axes[0].set_title('CUDA Benchmark Performance')
        axes[0].set_ylabel('Time (ms)')
        axes[0].tick_params(axis='x', rotation=45)

    # OpenCL benchmarks
    opencl_data = data[data['backend'] == 'OpenCL']
    if not opencl_data.empty:
        opencl_data.plot(x='name', y='time', kind='bar', ax=axes[1], color='blue')
        axes[1].set_title('OpenCL Benchmark Performance')
        axes[1].set_ylabel('Time (ms)')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Plot saved to {output}")

def main():
    parser = argparse.ArgumentParser(description="Visualize SHOC benchmark results")
    parser.add_argument("input", type=Path, help="Input JSON file from shocdriver")
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmark_results.png"),
                        help="Output image file")

    args = parser.parse_args()

    # Load data
    with args.input.open() as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    plot_performance_comparison(df, args.output)

if __name__ == "__main__":
    main()
```

### 4. Add Docker/Container Support

**Problem**: CUDA/OpenCL setup is complex and varies by system.

**Solution**: Provide containerized environments.

**File**: `docker/Dockerfile.cuda`
```dockerfile
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install SHOC
WORKDIR /opt
COPY . /opt/shoc
WORKDIR /opt/shoc/build

RUN cmake -GNinja \
    -DSHOC_BUILD_CUDA=ON \
    -DSHOC_BUILD_OPENCL=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/shoc/install \
    .. && \
    ninja && \
    ninja install

ENV PATH="/opt/shoc/install/bin:${PATH}"

WORKDIR /workspace
CMD ["/bin/bash"]
```

**File**: `docker/Dockerfile.opencl`
```dockerfile
FROM ubuntu:22.04

# Install OpenCL and build tools
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    ocl-icd-opencl-dev \
    opencl-headers \
    python3 \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/shoc
WORKDIR /opt/shoc/build

RUN cmake -GNinja \
    -DSHOC_BUILD_CUDA=OFF \
    -DSHOC_BUILD_OPENCL=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/shoc/install \
    .. && \
    ninja && \
    ninja install

ENV PATH="/opt/shoc/install/bin:${PATH}"

WORKDIR /workspace
CMD ["/bin/bash"]
```

**File**: `docker/docker-compose.yml`
```yaml
version: '3.8'

services:
  shoc-cuda:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cuda
    image: shoc:cuda
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ../results:/workspace/results

  shoc-opencl:
    build:
      context: ..
      dockerfile: docker/Dockerfile.opencl
    image: shoc:opencl
    devices:
      - /dev/dri:/dev/dri
    volumes:
      - ../results:/workspace/results
```

### 5. Add CTest Integration

**Problem**: No automated testing infrastructure.

**Solution**: Integrate with CTest for CI/CD.

**Add to root CMakeLists.txt**:
```cmake
# Enable testing
enable_testing()

# Add a function to register benchmark tests
function(shoc_add_benchmark_test TARGET CATEGORY BACKEND)
    add_test(
        NAME ${CATEGORY}_${BACKEND}_${TARGET}
        COMMAND ${TARGET} -s 1 -n 1
        WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin/${CATEGORY}/${BACKEND}
    )

    # Set timeout (2 minutes per test)
    set_tests_properties(${CATEGORY}_${BACKEND}_${TARGET} PROPERTIES
        TIMEOUT 120
        LABELS "${CATEGORY};${BACKEND}"
    )
endfunction()
```

**Usage**:
```bash
cmake --build build --target install
ctest --test-dir build --output-on-failure

# Run only CUDA tests
ctest --test-dir build -L CUDA

# Run only Serial tests
ctest --test-dir build -L Serial

# Parallel testing
ctest --test-dir build -j 4
```

### 6. Add GitHub Actions CI/CD

**Problem**: No continuous integration.

**Solution**: Automated builds and testing.

**File**: `.github/workflows/ci.yml`
```yaml
name: SHOC CI

on:
  push:
    branches: [ master-2025, develop ]
  pull_request:
    branches: [ master-2025 ]

jobs:
  build-cuda:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.0.0-devel-ubuntu22.04

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        apt-get update
        apt-get install -y cmake ninja-build

    - name: Configure
      run: |
        cmake -B build -GNinja \
          -DSHOC_BUILD_CUDA=ON \
          -DSHOC_BUILD_OPENCL=OFF \
          -DCMAKE_BUILD_TYPE=Release

    - name: Build
      run: cmake --build build

    - name: Install
      run: cmake --install build --prefix install

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: shoc-cuda
        path: install/

  build-opencl:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake ninja-build \
          ocl-icd-opencl-dev opencl-headers

    - name: Configure
      run: |
        cmake -B build -GNinja \
          -DSHOC_BUILD_CUDA=OFF \
          -DSHOC_BUILD_OPENCL=ON \
          -DCMAKE_BUILD_TYPE=Release

    - name: Build
      run: cmake --build build

    - name: Install
      run: cmake --install build --prefix install

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: shoc-opencl
        path: install/

  documentation:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Build documentation
      run: |
        # Generate API docs, user guide, etc.
        echo "Documentation build placeholder"

    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/master-2025'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/build
```

## 🔧 Medium Priority Recommendations

### 7. Add Benchmark Configuration Files

**Problem**: Benchmark parameters hardcoded or command-line only.

**Solution**: YAML/JSON configuration files.

**File**: `configs/example.yaml`
```yaml
# SHOC Benchmark Configuration

benchmarks:
  - name: BFS
    backend: CUDA
    category: Serial
    size: 2
    device: 0
    parameters:
      iterations: 10
      warmup: 2

  - name: FFT
    backend: OpenCL
    category: Serial
    size: 3
    device: 0
    parameters:
      iterations: 5
      warmup: 1

mpi:
  enabled: false
  nodes: 4
  devices_per_node: 2

output:
  format: json
  file: results/benchmark_run_{timestamp}.json
  plot: true
```

### 8. Add Performance Regression Detection

**Problem**: No way to track performance over time.

**Solution**: Store historical results and detect regressions.

**File**: `tools/regression_check.py`
```python
#!/usr/bin/env python3
"""
Performance Regression Checker
Compares current results against baseline
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

class RegressionChecker:
    def __init__(self, baseline_file: Path, threshold: float = 0.10):
        self.baseline = self.load_baseline(baseline_file)
        self.threshold = threshold  # 10% regression threshold

    def load_baseline(self, file: Path) -> Dict:
        with file.open() as f:
            return json.load(f)

    def check_regression(self, current_results: Dict) -> List[Dict]:
        """Compare current results against baseline."""
        regressions = []

        for bench, current_time in current_results.items():
            if bench in self.baseline:
                baseline_time = self.baseline[bench]
                diff_pct = (current_time - baseline_time) / baseline_time * 100

                if diff_pct > self.threshold * 100:
                    regressions.append({
                        "benchmark": bench,
                        "baseline": baseline_time,
                        "current": current_time,
                        "regression_pct": diff_pct
                    })

        return regressions

    def report(self, regressions: List[Dict]):
        """Print regression report."""
        if not regressions:
            print("✓ No performance regressions detected")
            return 0

        print(f"⚠ {len(regressions)} performance regressions detected:\n")
        for r in regressions:
            print(f"  {r['benchmark']}")
            print(f"    Baseline: {r['baseline']:.2f}ms")
            print(f"    Current:  {r['current']:.2f}ms")
            print(f"    Regression: +{r['regression_pct']:.1f}%\n")

        return 1

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path, help="Baseline results JSON")
    parser.add_argument("current", type=Path, help="Current results JSON")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Regression threshold (default: 0.10 = 10%%)")

    args = parser.parse_args()

    # Load current results
    with args.current.open() as f:
        current = json.load(f)

    # Check for regressions
    checker = RegressionChecker(args.baseline, args.threshold)
    regressions = checker.check_regression(current)

    sys.exit(checker.report(regressions))

if __name__ == "__main__":
    main()
```

### 9. Add JSON Output Support to Benchmarks

**Problem**: Benchmarks output text, hard to parse programmatically.

**Solution**: Add `--json` flag to all benchmarks.

**Modify benchmark main.cpp template**:
```cpp
// In src/common/main.cpp (or similar)
if (op.getOptionBool("json")) {
    // Output JSON format
    std::cout << "{" << std::endl;
    std::cout << "  \"benchmark\": \"" << benchmarkName << "\"," << std::endl;
    std::cout << "  \"backend\": \"" << backend << "\"," << std::endl;
    std::cout << "  \"results\": [" << std::endl;

    for (const auto& result : results) {
        std::cout << "    {" << std::endl;
        std::cout << "      \"test\": \"" << result.test << "\"," << std::endl;
        std::cout << "      \"value\": " << result.value << "," << std::endl;
        std::cout << "      \"unit\": \"" << result.unit << "\"" << std::endl;
        std::cout << "    }," << std::endl;
    }

    std::cout << "  ]" << std::endl;
    std::cout << "}" << std::endl;
} else {
    // Traditional text output
    resultDB.DumpSummary(std::cout);
}
```

### 10. Add Benchmark Discovery Tool

**Problem**: Users don't know what benchmarks are available.

**Solution**: List/describe all available benchmarks.

**File**: `tools/list_benchmarks.py`
```python
#!/usr/bin/env python3
"""
SHOC Benchmark Discovery
Lists all available benchmarks with descriptions
"""

from pathlib import Path
import subprocess

DESCRIPTIONS = {
    "BFS": "Breadth-First Search graph traversal",
    "FFT": "Fast Fourier Transform",
    "GEMM": "General Matrix Multiply (BLAS)",
    "MD": "Molecular Dynamics n-body simulation",
    "MD5Hash": "MD5 hash computation",
    "NeuralNet": "Neural network training",
    "Reduction": "Parallel reduction operation",
    "Scan": "Parallel prefix sum (scan)",
    "Sort": "Radix sort algorithm",
    "Spmv": "Sparse Matrix-Vector multiply",
    "Stencil2D": "2D stencil computation",
    "Triad": "STREAM Triad memory benchmark",
    "S3D": "S3D combustion chemistry",
    "QTC": "Quality Threshold Clustering",
    "BusSpeedDownload": "PCIe bus bandwidth (host to device)",
    "BusSpeedReadback": "PCIe bus bandwidth (device to host)",
    "DeviceMemory": "Device memory bandwidth",
    "MaxFlops": "Peak FLOPS measurement",
    "KernelCompile": "OpenCL kernel compilation time",
    "QueueDelay": "OpenCL queue latency"
}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="List SHOC benchmarks")
    parser.add_argument("--install-dir", type=Path, default=Path.cwd(),
                        help="SHOC installation directory")
    parser.add_argument("--backend", choices=["CUDA", "OpenCL", "all"],
                        default="all", help="Filter by backend")
    parser.add_argument("--category", choices=["Serial", "EP", "TP", "all"],
                        default="all", help="Filter by category")
    parser.add_argument("--markdown", action="store_true",
                        help="Output as Markdown table")

    args = parser.parse_args()

    bin_dir = args.install_dir / "bin"
    if not bin_dir.exists():
        print(f"Error: Binary directory not found: {bin_dir}")
        return 1

    # Collect all benchmarks
    benchmarks = []
    for category_dir in bin_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        if args.category != "all" and category != args.category:
            continue

        for backend_dir in category_dir.iterdir():
            if not backend_dir.is_dir():
                continue

            backend = backend_dir.name
            if args.backend != "all" and backend != args.backend:
                continue

            for bench in backend_dir.iterdir():
                if bench.is_file() and bench.stat().st_mode & 0o111:
                    benchmarks.append({
                        "name": bench.name,
                        "category": category,
                        "backend": backend,
                        "path": bench,
                        "description": DESCRIPTIONS.get(bench.name, "No description")
                    })

    # Output
    if args.markdown:
        print("| Benchmark | Category | Backend | Description |")
        print("|-----------|----------|---------|-------------|")
        for b in sorted(benchmarks, key=lambda x: (x['category'], x['backend'], x['name'])):
            print(f"| {b['name']} | {b['category']} | {b['backend']} | {b['description']} |")
    else:
        print(f"Found {len(benchmarks)} benchmarks:\n")

        current_category = None
        current_backend = None

        for b in sorted(benchmarks, key=lambda x: (x['category'], x['backend'], x['name'])):
            if b['category'] != current_category or b['backend'] != current_backend:
                print(f"\n{b['category']}/{b['backend']}:")
                current_category = b['category']
                current_backend = b['backend']

            print(f"  • {b['name']:<20} - {b['description']}")

if __name__ == "__main__":
    main()
```

## 📊 Low Priority (Nice to Have)

### 11. Add Benchmark Comparison Tool

Compare results across different GPUs, configurations, or versions:

```python
#!/usr/bin/env python3
"""Compare SHOC results from different runs"""

import json
import pandas as pd
from pathlib import Path

def compare_results(file1: Path, file2: Path):
    with file1.open() as f:
        results1 = json.load(f)
    with file2.open() as f:
        results2 = json.load(f)

    df1 = pd.DataFrame(results1).set_index('name')
    df2 = pd.DataFrame(results2).set_index('name')

    comparison = pd.DataFrame({
        'config1': df1['time'],
        'config2': df2['time'],
        'speedup': df1['time'] / df2['time']
    })

    print(comparison.to_string())
    print(f"\nAverage speedup: {comparison['speedup'].mean():.2f}x")
```

### 12. Add Web Dashboard

Create a simple web interface for viewing results:

**File**: `tools/dashboard/app.py`
```python
#!/usr/bin/env python3
from flask import Flask, render_template, jsonify
from pathlib import Path
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/results')
def get_results():
    results_dir = Path('results')
    all_results = []
    for result_file in results_dir.glob('*.json'):
        with result_file.open() as f:
            all_results.append(json.load(f))
    return jsonify(all_results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 13. Add Installation Validation Script

**File**: `tools/validate_install.sh`
```bash
#!/bin/bash
# Validate SHOC installation

set -e

INSTALL_DIR=${1:-.}

echo "=== SHOC Installation Validator ==="
echo "Checking installation at: $INSTALL_DIR"
echo

# Check directory structure
echo "Checking directory structure..."
for dir in bin bin/Serial bin/EP bin/TP lib; do
    if [ -d "$INSTALL_DIR/$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir (missing)"
        exit 1
    fi
done

# Check libraries
echo -e "\nChecking libraries..."
for lib in libSHOCCommon.a libSHOCCommonOpenCL.a; do
    if [ -f "$INSTALL_DIR/lib/$lib" ]; then
        echo "  ✓ $lib"
    else
        echo "  ⚠ $lib (not found - might be optional)"
    fi
done

# Count executables
echo -e "\nCounting executables..."
CUDA_COUNT=$(find "$INSTALL_DIR/bin" -type f -path "*/CUDA/*" 2>/dev/null | wc -l)
OPENCL_COUNT=$(find "$INSTALL_DIR/bin" -type f -path "*/OpenCL/*" 2>/dev/null | wc -l)

echo "  CUDA benchmarks: $CUDA_COUNT"
echo "  OpenCL benchmarks: $OPENCL_COUNT"
echo "  Total: $((CUDA_COUNT + OPENCL_COUNT))"

# Try running a simple benchmark
echo -e "\nTrying to run a test benchmark..."
if [ $CUDA_COUNT -gt 0 ]; then
    BENCH=$(find "$INSTALL_DIR/bin/Serial/CUDA" -type f | head -1)
    echo "  Running: $(basename $BENCH)"
    if timeout 10 "$BENCH" -s 1 -n 1 > /dev/null 2>&1; then
        echo "  ✓ Benchmark executed successfully"
    else
        echo "  ⚠ Benchmark failed (might need GPU)"
    fi
fi

echo -e "\n✓ Installation validation complete"
```

### 14. Add Power/Energy Measurement Support

For modern GPU benchmarking, energy efficiency is important:

```cpp
// In benchmark code, add NVML support for NVIDIA GPUs
#ifdef USE_NVML
#include <nvml.h>

class PowerMeter {
public:
    PowerMeter(int device_id) {
        nvmlInit();
        nvmlDeviceGetHandleByIndex(device_id, &device);
        nvmlDeviceGetPowerUsage(device, &initial_power);
    }

    ~PowerMeter() {
        nvmlShutdown();
    }

    unsigned int get_average_power() {
        unsigned int current_power;
        nvmlDeviceGetPowerUsage(device, &current_power);
        return (current_power + initial_power) / 2;
    }

private:
    nvmlDevice_t device;
    unsigned int initial_power;
};
#endif
```

### 15. Add Multi-GPU Benchmark Support

**Problem**: Single GPU benchmarks only.

**Solution**: Add multi-GPU variants.

```cmake
# In CMakeLists.txt
option(SHOC_BUILD_MULTIGPU "Build multi-GPU benchmarks" OFF)

if(SHOC_BUILD_MULTIGPU)
    add_subdirectory(src/multigpu)
endif()
```

## 📝 Documentation Improvements

### 16. Create Quick Start Guide

**File**: `QUICKSTART.md`
```markdown
# SHOC Quick Start Guide

## Installation (5 minutes)

```bash
# Clone
git clone https://github.com/your-org/shoc.git
cd shoc

# Build (CUDA)
cmake -B build -DSHOC_BUILD_CUDA=ON -DSHOC_BUILD_OPENCL=OFF
cmake --build build -j
cmake --install build --prefix ./install

# Run
./install/bin/Serial/CUDA/MaxFlops
```

## Running Your First Benchmark

```bash
# Single benchmark
./install/bin/Serial/CUDA/BFS -s 2

# Full suite
python3 tools/shocdriver.py --cuda -s 2

# View results
cat shoc_results_*.csv
```

## Next Steps

- Read [CMAKE_BUILD.md](CMAKE_BUILD.md) for configuration options
- See [examples/](examples/) for advanced usage
- Run `tools/list_benchmarks.py` to see all benchmarks
```

### 17. Add Troubleshooting Guide

**File**: `TROUBLESHOOTING.md`
```markdown
# SHOC Troubleshooting Guide

## Build Issues

### "CUDA not found"
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Or specify directly
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

### "OpenCL headers not found"
```bash
# Ubuntu/Debian
sudo apt-get install ocl-icd-opencl-dev opencl-headers

# Then specify location
cmake -DOpenCL_INCLUDE_DIR=/usr/include ..
```

## Runtime Issues

### "Illegal memory access"
- Check CUDA architecture matches your GPU
- Rebuild with correct `-DCMAKE_CUDA_ARCHITECTURES`

### "CL_DEVICE_NOT_FOUND"
- Install GPU drivers
- Check: `clinfo` (OpenCL) or `nvidia-smi` (CUDA)

### Benchmark crashes
- Try smaller problem size: `-s 1`
- Reduce iterations: `-n 1`
- Check available memory

## Performance Issues

### Low performance
- Check GPU isn't throttling (temperature)
- Ensure exclusive GPU access
- Disable power saving modes
```

## 🎯 Summary & Priorities

### Immediate (Do First)
1. **CMake Presets** - Huge usability win, minimal effort
2. **Modern shocdriver.py** - Much better UX
3. **CTest integration** - Enables CI/CD
4. **Installation validator** - Helps users verify setup

### Short Term (Next Sprint)
5. **Docker containers** - Reproducible environments
6. **GitHub Actions CI** - Automated quality assurance
7. **Result visualization** - Better insights
8. **Quick start guide** - Lower barrier to entry

### Medium Term (Next Quarter)
9. **JSON output** - Machine-readable results
10. **Regression detection** - Track performance over time
11. **Benchmark discovery** - Improved documentation
12. **Configuration files** - Easier batch runs

### Long Term (Future)
13. **Web dashboard** - Enterprise-friendly
14. **Power measurement** - Modern GPU metrics
15. **Multi-GPU support** - Scales to larger systems

## Implementation Order

I recommend implementing in this order for maximum impact:

1. **Week 1**: CMake presets, validation script, quick start guide
2. **Week 2**: Modern shocdriver.py, result visualization
3. **Week 3**: Docker support, CTest integration
4. **Week 4**: GitHub Actions, documentation improvements

This gives you a professional, modern benchmark suite in one month while maintaining backward compatibility.
