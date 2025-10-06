# SHOC Visualization Guide

Generate publication-quality charts and interactive visualizations from SHOC benchmark results.

## What It Does

The SHOC Visualization Tool (`shocviz.py`):
- Loads results from CSV (from driver) or JSON reports (from shocdriver.py)
- Generates bar charts, comparison charts, and speedup analysis
- Supports static images (PNG, SVG, PDF) and interactive HTML
- Automatically categorizes metrics (throughput, bandwidth, latency)
- Filters benchmarks by name patterns
- Compares multiple runs side-by-side

## Quick Start

### Single Run Visualization

**Generate bar chart from CSV:**
```bash
./tools/shocviz.py results.csv --output overview.png
```

**From JSON report:**
```bash
./tools/shocviz.py report.json --output overview.png
```

### Comparison Visualization

**Compare two runs:**
```bash
./tools/shocviz.py baseline.csv optimized.csv --compare --output comparison.png
```

**Speedup analysis:**
```bash
./tools/shocviz.py baseline.csv optimized.csv --speedup --output speedup.png
```

### Interactive Charts

**Generate interactive HTML (requires plotly):**
```bash
./tools/shocviz.py results.json --interactive --output report.html
```

## Installation

### Required

```bash
pip install matplotlib
```

### Optional (for interactive charts)

```bash
pip install plotly
```

Install both:
```bash
pip install matplotlib plotly
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `input` | Input file(s): CSV or JSON (1+ files) |
| `--output`, `-o` | Output file (.png, .svg, .pdf, .html) |
| `--chart {bar,line}` | Chart type (default: bar) |
| `--compare` | Compare multiple runs (2+ inputs) |
| `--speedup` | Generate speedup chart (exactly 2 inputs) |
| `--filter REGEX` | Filter benchmarks by regex pattern |
| `--interactive` | Generate interactive plot (requires plotly) |

## Chart Types

### 1. Bar Chart (Single Run)

Visualizes all benchmarks from a single run, grouped by category.

```bash
./tools/shocviz.py results.csv --chart bar --output report.png
```

**Features:**
- Automatic grouping by metric type (throughput, bandwidth, latency)
- Color-coded by category
- Value labels on bars
- Separate subplots for different metric types

**Output:**
- Multiple horizontal sections, one per category
- Easy to identify best/worst performers

### 2. Comparison Chart

Side-by-side comparison of multiple runs.

```bash
./tools/shocviz.py run1.csv run2.csv run3.csv --compare --output comparison.png
```

**Features:**
- Shows only common benchmarks across all runs
- Grouped bars for easy comparison
- Handles different run names automatically
- Works with 2+ input files

**Use Cases:**
- Compare different GPUs
- Before/after optimization
- CUDA vs OpenCL comparison
- Different compiler versions

### 3. Speedup Chart

Shows relative performance improvement (or regression).

```bash
./tools/shocviz.py baseline.csv improved.csv --speedup --output speedup.png
```

**Features:**
- Speedup = improved / baseline (for throughput)
- Speedup = baseline / improved (for latency - inverted)
- Green bars = improvement (>1.0×)
- Red bars = regression (<1.0×)
- Speedup values labeled on bars

**Use Cases:**
- Quantify optimization impact
- Identify which benchmarks improved most
- Performance regression testing

### 4. Interactive Charts

HTML charts with zoom, pan, hover tooltips.

```bash
./tools/shocviz.py results.json --interactive --output report.html
```

**Features:**
- Hover for exact values
- Zoom/pan controls
- Toggle series on/off
- Export to PNG from browser
- Responsive design

**Requirements:**
- `pip install plotly`
- Web browser to view

## Filtering Benchmarks

Use `--filter` with regex patterns to show subset of benchmarks:

**FFT benchmarks only:**
```bash
./tools/shocviz.py results.csv --filter "fft" --output fft-only.png
```

**GEMM benchmarks:**
```bash
./tools/shocviz.py results.csv --filter "gemm" --output gemm.png
```

**Multiple patterns (FFT or GEMM):**
```bash
./tools/shocviz.py results.csv --filter "fft|gemm" --output linear-algebra.png
```

**All double-precision:**
```bash
./tools/shocviz.py results.csv --filter "dp|dgemm|dgemv" --output double-precision.png
```

**PCIe bandwidth:**
```bash
./tools/shocviz.py results.csv --filter "pcie|bspeed" --output pcie.png
```

## Output Formats

### PNG (default)
```bash
./tools/shocviz.py results.csv --output report.png
```
- High resolution (300 DPI)
- Good for presentations
- Smaller file size

### SVG (vector)
```bash
./tools/shocviz.py results.csv --output report.svg
```
- Scalable vector graphics
- Perfect for publications
- Editable in Inkscape/Illustrator

### PDF
```bash
./tools/shocviz.py results.csv --output report.pdf
```
- Vector format
- Good for LaTeX documents
- Multi-page support

### HTML (interactive)
```bash
./tools/shocviz.py results.json --interactive --output report.html
```
- Requires `--interactive` flag
- Needs plotly installed
- Best for exploration

## Workflow Examples

### 1. Basic Performance Report

```bash
# Run benchmarks
./tools/shocdriver.py --backend cuda --size 2 --output results.csv

# Generate visualization
./tools/shocviz.py results.csv --output performance-report.png
```

### 2. Before/After Optimization

```bash
# Baseline
./tools/shocdriver.py --backend cuda --size 2 --output baseline.csv

# ... make optimizations ...

# After optimization
./tools/shocdriver.py --backend cuda --size 2 --output optimized.csv

# Compare
./tools/shocviz.py baseline.csv optimized.csv --compare --output comparison.png
./tools/shocviz.py baseline.csv optimized.csv --speedup --output speedup.png
```

### 3. Multi-GPU Comparison

```bash
# Single GPU
./tools/shocdriver.py --backend cuda --size 2 --device 0 --output gpu0.csv

# Dual GPU
./tools/shocdriver.py --backend cuda --size 2 --device 0,1 --output gpu01.csv

# Quad GPU
./tools/shocdriver.py --backend cuda --size 2 --device 0,1,2,3 --output gpu0123.csv

# Compare scaling
./tools/shocviz.py gpu0.csv gpu01.csv gpu0123.csv --compare --output scaling.png
```

### 4. CUDA vs OpenCL

```bash
# CUDA
./tools/shocdriver.py --backend cuda --size 2 --output cuda.csv

# OpenCL
./tools/shocdriver.py --backend opencl --size 2 --output opencl.csv

# Compare
./tools/shocviz.py cuda.csv opencl.csv --compare --output cuda-vs-opencl.png
```

### 5. Continuous Monitoring

```bash
#!/bin/bash
# weekly-benchmark.sh

DATE=$(date +%Y%m%d)

# Run benchmarks
./tools/shocdriver.py --config config/baseline.yaml \
  --output "results-${DATE}.csv" \
  --json-report "report-${DATE}.json"

# Generate visualizations
./tools/shocviz.py "results-${DATE}.csv" \
  --output "reports/weekly-${DATE}.png"

# Compare with last week
if [ -f results-lastweek.csv ]; then
  ./tools/shocviz.py results-lastweek.csv "results-${DATE}.csv" \
    --speedup \
    --output "reports/week-over-week-${DATE}.png"
fi

# Update baseline
cp "results-${DATE}.csv" results-lastweek.csv
```

### 6. Publication-Quality Figures

```bash
# Generate SVG for papers
./tools/shocviz.py baseline.csv optimized.csv \
  --speedup \
  --filter "fft|gemm|reduction|scan" \
  --output figure1-speedup.svg

# Or PDF for LaTeX
./tools/shocviz.py results.csv \
  --filter "gmem|lmem|bspeed" \
  --output figure2-bandwidth.pdf
```

### 7. Interactive Exploration

```bash
# Generate interactive HTML report
./tools/shocdriver.py --backend cuda --size 2 \
  --json-report detailed-report.json

./tools/shocviz.py detailed-report.json \
  --interactive \
  --output interactive-report.html

# Open in browser
xdg-open interactive-report.html  # Linux
open interactive-report.html      # macOS
start interactive-report.html     # Windows
```

## Benchmark Categories

The tool automatically categorizes benchmarks:

### Throughput (Higher is Better)
- Compute: `maxspflops`, `maxdpflops`
- FFT: `fft_sp`, `fft_dp`, `ifft_sp`, `ifft_dp`
- GEMM: `sgemm_n`, `dgemm_t`, etc.
- MD: `md_sp_flops`, `md_dp_flops`
- Reduction/Scan: `reduction`, `scan`
- Sort: `sort`
- BFS: `bfs`, `bfs_teps`
- S3D: `s3d`, `s3d_dp`
- SpMV: `spmv_*`
- Stencil: `stencil`, `stencil_dp`

### Bandwidth (Higher is Better)
- Memory: `gmem_readbw`, `lmem_writebw`, `tex_readbw`
- PCIe: `bspeed_download`, `bspeed_readback`
- Triad: `triad_bw`
- MD: `md_sp_bw`, `md_dp_bw`

### Latency (Lower is Better)
- OpenCL: `ocl_kernel`, `ocl_queue`

## Tips

1. **Use JSON input for best metadata**: JSON reports from shocdriver.py include timestamps and config
2. **Filter for focus**: Use `--filter` to highlight specific benchmark families
3. **SVG for publications**: Vector formats scale perfectly
4. **Interactive for exploration**: Use HTML output to explore data dynamically
5. **Speedup for optimization**: Speedup charts clearly show improvements
6. **Comparison for variants**: Compare different configurations side-by-side
7. **Automate with scripts**: Create shell scripts for regular reporting

## Customization

### Example: Create Custom Comparison Script

```bash
#!/bin/bash
# compare-configurations.sh

CONFIGS=("baseline" "O2" "O3" "fast-math")

# Run all configurations
for config in "${CONFIGS[@]}"; do
  ./tools/shocdriver.py --config "config/${config}.yaml" \
    --output "results-${config}.csv"
done

# Create comparison chart
./tools/shocviz.py results-*.csv \
  --compare \
  --output optimization-comparison.png

# Create individual speedup charts
for config in "${CONFIGS[@]:1}"; do
  ./tools/shocviz.py results-baseline.csv "results-${config}.csv" \
    --speedup \
    --output "speedup-${config}.png"
done

echo "All comparisons generated in current directory"
```

### Example: Focus on Memory Subsystem

```bash
# Only memory and bandwidth benchmarks
./tools/shocviz.py results.csv \
  --filter "mem|bw|triad|bspeed" \
  --output memory-subsystem.png
```

### Example: Kernel-Only Performance

```bash
# Exclude PCIe transfer benchmarks
./tools/shocviz.py results.csv \
  --filter "^(?!.*pcie).*$" \
  --output kernel-only.png
```

## Troubleshooting

### "matplotlib not available"

Install matplotlib:
```bash
pip install matplotlib
```

### "plotly not available"

Only needed for `--interactive`:
```bash
pip install plotly
```

### "No common benchmarks found"

When comparing runs, ensure they ran the same benchmarks:
- Use same `--backend` (cuda/opencl)
- Check CSV/JSON files have overlapping result names
- Use `--filter` to focus on common subset

### "No results to plot"

- Check input file has data
- Verify filter pattern doesn't exclude everything
- Ensure CSV has data row (not just header)

### Charts look cluttered

Use `--filter` to reduce number of benchmarks:
```bash
./tools/shocviz.py results.csv --filter "fft|gemm" --output clean.png
```

### Want different colors/styles

Edit `shocviz.py` and modify the `colors` dictionary in visualizer classes.

## Integration with Other Tools

**With shocdriver.py:**
```bash
# Generate results and visualize in one go
./tools/shocdriver.py --backend cuda --size 2 \
  --output results.csv \
  --json-report report.json

./tools/shocviz.py report.json --output chart.png
```

**With validation:**
```bash
# Validate, benchmark, visualize
./tools/validate_install.sh install-cuda
./tools/shocdriver.py --backend cuda --size 2 --output results.csv
./tools/shocviz.py results.csv --output report.png
```

**With CMake:**
```bash
# Build, install, benchmark, visualize
cmake --preset cuda-only
cmake --build build-cuda-only --target install
./tools/shocdriver.py --backend cuda --size 2 --bin-dir install-cuda/bin --output results.csv
./tools/shocviz.py results.csv --output performance.png
```

## Related Documentation

- [SHOCDRIVER_GUIDE.md](./SHOCDRIVER_GUIDE.md) - Running benchmarks
- [QUICKSTART.md](../QUICKSTART.md) - Getting started
- [RECOMMENDATIONS.md](../RECOMMENDATIONS.md) - All improvements

## Future Enhancements

Potential additions (not yet implemented):
- Heatmaps for parameter sweeps
- Time-series plots for monitoring
- Statistical confidence intervals
- Automated regression detection
- Performance profiles
- Multi-device scaling efficiency plots
