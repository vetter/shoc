#!/usr/bin/env python3
"""
SHOC Visualization Tool

Generate plots and charts from SHOC benchmark results.
Supports single-run visualizations and multi-run comparisons.

Usage:
    ./shocviz.py results.csv --output report.html
    ./shocviz.py baseline.json current.json --compare --output comparison.png
    ./shocviz.py results.csv --chart bar --filter fft,gemm
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

# Optional dependencies with graceful fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib", file=sys.stderr)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Note: plotly not available for interactive plots. Install with: pip install plotly", file=sys.stderr)

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    value: float
    unit: str
    category: str = "unknown"  # throughput, latency, bandwidth, etc.

@dataclass
class BenchmarkRun:
    """Complete benchmark run with metadata"""
    name: str
    timestamp: Optional[str]
    config: Dict
    results: List[BenchmarkResult]

# Benchmark categorization
BENCHMARK_METADATA = {
    # Throughput metrics (higher is better)
    'maxspflops': ('Compute', 'GFLOPS', 'throughput'),
    'maxdpflops': ('Compute', 'GFLOPS', 'throughput'),
    'fft_sp': ('FFT', 'GFLOPS', 'throughput'),
    'fft_dp': ('FFT', 'GFLOPS', 'throughput'),
    'ifft_sp': ('FFT', 'GFLOPS', 'throughput'),
    'ifft_dp': ('FFT', 'GFLOPS', 'throughput'),
    'sgemm_n': ('GEMM', 'GFLOPS', 'throughput'),
    'sgemm_t': ('GEMM', 'GFLOPS', 'throughput'),
    'dgemm_n': ('GEMM', 'GFLOPS', 'throughput'),
    'dgemm_t': ('GEMM', 'GFLOPS', 'throughput'),
    'md_sp_flops': ('MD', 'GFLOPS', 'throughput'),
    'md_dp_flops': ('MD', 'GFLOPS', 'throughput'),
    'reduction': ('Reduction', 'GB/s', 'throughput'),
    'reduction_dp': ('Reduction', 'GB/s', 'throughput'),
    'scan': ('Scan', 'GB/s', 'throughput'),
    'scan_dp': ('Scan', 'GB/s', 'throughput'),
    'sort': ('Sort', 'MKeys/s', 'throughput'),
    'md5hash': ('MD5Hash', 'MHash/s', 'throughput'),
    'bfs': ('BFS', 'MTEPS', 'throughput'),
    'bfs_teps': ('BFS', 'MTEPS', 'throughput'),
    's3d': ('S3D', 'GFLOPS', 'throughput'),
    's3d_dp': ('S3D', 'GFLOPS', 'throughput'),

    # Bandwidth metrics (higher is better)
    'gmem_readbw': ('Memory', 'GB/s', 'bandwidth'),
    'gmem_writebw': ('Memory', 'GB/s', 'bandwidth'),
    'gmem_readbw_strided': ('Memory', 'GB/s', 'bandwidth'),
    'gmem_writebw_strided': ('Memory', 'GB/s', 'bandwidth'),
    'lmem_readbw': ('Memory', 'GB/s', 'bandwidth'),
    'lmem_writebw': ('Memory', 'GB/s', 'bandwidth'),
    'tex_readbw': ('Memory', 'GB/s', 'bandwidth'),
    'bspeed_download': ('PCIe', 'GB/s', 'bandwidth'),
    'bspeed_readback': ('PCIe', 'GB/s', 'bandwidth'),
    'triad_bw': ('Memory', 'GB/s', 'bandwidth'),
    'md_sp_bw': ('MD', 'GB/s', 'bandwidth'),
    'md_dp_bw': ('MD', 'GB/s', 'bandwidth'),

    # Latency metrics (lower is better)
    'ocl_kernel': ('OpenCL', 'ms', 'latency'),
    'ocl_queue': ('OpenCL', 'us', 'latency'),

    # Sparse matrix
    'spmv_csr_scalar_sp': ('SpMV', 'GFLOPS', 'throughput'),
    'spmv_csr_scalar_dp': ('SpMV', 'GFLOPS', 'throughput'),
    'spmv_csr_vector_sp': ('SpMV', 'GFLOPS', 'throughput'),
    'spmv_csr_vector_dp': ('SpMV', 'GFLOPS', 'throughput'),
    'spmv_ellpackr_sp': ('SpMV', 'GFLOPS', 'throughput'),
    'spmv_ellpackr_dp': ('SpMV', 'GFLOPS', 'throughput'),

    # Stencil
    'stencil': ('Stencil', 'GCell/s', 'throughput'),
    'stencil_dp': ('Stencil', 'GCell/s', 'throughput'),
}

class DataLoader:
    """Load benchmark results from various formats"""

    @staticmethod
    def load_csv(csv_file: Path) -> BenchmarkRun:
        """Load results from CSV file"""
        results = []

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)  # Only one data row in SHOC CSV

            for name, value_str in row.items():
                if not value_str or value_str == '':
                    continue

                try:
                    value = float(value_str)
                    category, unit, metric_type = BENCHMARK_METADATA.get(
                        name, ('Unknown', '', 'throughput')
                    )
                    results.append(BenchmarkResult(name, value, unit, metric_type))
                except ValueError:
                    pass

        return BenchmarkRun(
            name=csv_file.stem,
            timestamp=None,
            config={},
            results=results
        )

    @staticmethod
    def load_json(json_file: Path) -> BenchmarkRun:
        """Load results from JSON report (from shocdriver.py)"""
        with open(json_file) as f:
            data = json.load(f)

        results = []
        for bench_result in data.get('results', []):
            if bench_result['status'] != 'success':
                continue

            for name, (value, unit) in bench_result.get('results', {}).items():
                category, default_unit, metric_type = BENCHMARK_METADATA.get(
                    name, ('Unknown', unit, 'throughput')
                )
                results.append(BenchmarkResult(name, value, unit or default_unit, metric_type))

        return BenchmarkRun(
            name=json_file.stem,
            timestamp=data.get('timestamp'),
            config=data.get('config', {}),
            results=results
        )

    @staticmethod
    def load(file_path: Path) -> BenchmarkRun:
        """Auto-detect format and load"""
        if file_path.suffix == '.json':
            return DataLoader.load_json(file_path)
        elif file_path.suffix == '.csv':
            return DataLoader.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

class MatplotlibVisualizer:
    """Generate plots using matplotlib"""

    @staticmethod
    def bar_chart(run: BenchmarkRun, output: Path, filter_pattern: Optional[str] = None):
        """Generate bar chart of all results"""
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib required for this chart type", file=sys.stderr)
            return

        results = run.results
        if filter_pattern:
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            results = [r for r in results if pattern.search(r.name)]

        if not results:
            print("No results to plot", file=sys.stderr)
            return

        # Group by category
        grouped = defaultdict(list)
        for r in results:
            grouped[r.category].append(r)

        fig, axes = plt.subplots(len(grouped), 1, figsize=(12, 4 * len(grouped)))
        if len(grouped) == 1:
            axes = [axes]

        for idx, (category, category_results) in enumerate(sorted(grouped.items())):
            ax = axes[idx]

            names = [r.name for r in category_results]
            values = [r.value for r in category_results]

            colors = ['#2E86AB' if category == 'throughput' else
                     '#A23B72' if category == 'bandwidth' else '#F18F01'
                     for _ in category_results]

            bars = ax.bar(range(len(names)), values, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel(category_results[0].unit if category_results else '')
            ax.set_title(f'{category.capitalize()} Metrics - {run.name}')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to {output}")

    @staticmethod
    def comparison_chart(runs: List[BenchmarkRun], output: Path, filter_pattern: Optional[str] = None):
        """Generate comparison chart for multiple runs"""
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib required for this chart type", file=sys.stderr)
            return

        # Find common benchmarks across all runs
        common_names = set(r.name for r in runs[0].results)
        for run in runs[1:]:
            common_names &= set(r.name for r in run.results)

        if filter_pattern:
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            common_names = {n for n in common_names if pattern.search(n)}

        common_names = sorted(common_names)

        if not common_names:
            print("No common benchmarks found across runs", file=sys.stderr)
            return

        # Prepare data
        fig, ax = plt.subplots(figsize=(14, max(6, len(common_names) * 0.4)))

        x = range(len(common_names))
        width = 0.8 / len(runs)
        colors = plt.cm.Set2(range(len(runs)))

        for run_idx, run in enumerate(runs):
            result_dict = {r.name: r.value for r in run.results}
            values = [result_dict.get(name, 0) for name in common_names]

            offset = (run_idx - len(runs)/2 + 0.5) * width
            ax.barh([i + offset for i in x], values, width,
                   label=run.name, color=colors[run_idx], alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(common_names)
        ax.set_xlabel('Performance')
        ax.set_title('Benchmark Comparison')
        ax.legend(loc='best')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to {output}")

    @staticmethod
    def speedup_chart(baseline: BenchmarkRun, comparison: BenchmarkRun, output: Path,
                     filter_pattern: Optional[str] = None):
        """Generate speedup chart (comparison / baseline)"""
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib required for this chart type", file=sys.stderr)
            return

        baseline_dict = {r.name: r for r in baseline.results}
        comparison_dict = {r.name: r for r in comparison.results}

        common_names = set(baseline_dict.keys()) & set(comparison_dict.keys())

        if filter_pattern:
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            common_names = {n for n in common_names if pattern.search(n)}

        if not common_names:
            print("No common benchmarks found", file=sys.stderr)
            return

        speedups = []
        names = []
        categories = []

        for name in sorted(common_names):
            base_val = baseline_dict[name].value
            comp_val = comparison_dict[name].value
            category = baseline_dict[name].category

            if base_val > 0:
                # For latency (lower is better), invert the speedup
                if category == 'latency':
                    speedup = base_val / comp_val
                else:
                    speedup = comp_val / base_val

                speedups.append(speedup)
                names.append(name)
                categories.append(category)

        fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))

        colors = ['#27AE60' if s > 1.0 else '#E74C3C' if s < 1.0 else '#95A5A6'
                 for s in speedups]

        bars = ax.barh(range(len(names)), speedups, color=colors, alpha=0.7)
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Speedup (×)')
        ax.set_title(f'Speedup: {comparison.name} vs {baseline.name}')
        ax.grid(axis='x', alpha=0.3)

        # Add speedup labels
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            width = bar.get_width()
            label_x = width + (0.05 if width > 1 else -0.05)
            ha = 'left' if width > 1 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{speedup:.2f}×', ha=ha, va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Speedup chart saved to {output}")

class PlotlyVisualizer:
    """Generate interactive plots using plotly"""

    @staticmethod
    def interactive_bar(run: BenchmarkRun, output: Path, filter_pattern: Optional[str] = None):
        """Generate interactive bar chart"""
        if not HAS_PLOTLY:
            print("Error: plotly required for interactive charts", file=sys.stderr)
            return

        results = run.results
        if filter_pattern:
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            results = [r for r in results if pattern.search(r.name)]

        if not results:
            print("No results to plot", file=sys.stderr)
            return

        # Group by category
        grouped = defaultdict(list)
        for r in results:
            grouped[r.category].append(r)

        fig = make_subplots(
            rows=len(grouped), cols=1,
            subplot_titles=[f'{cat.capitalize()} Metrics' for cat in sorted(grouped.keys())]
        )

        colors = {
            'throughput': '#2E86AB',
            'bandwidth': '#A23B72',
            'latency': '#F18F01'
        }

        for idx, (category, category_results) in enumerate(sorted(grouped.items()), 1):
            names = [r.name for r in category_results]
            values = [r.value for r in category_results]

            fig.add_trace(
                go.Bar(
                    x=names,
                    y=values,
                    name=category,
                    marker_color=colors.get(category, '#999999'),
                    text=[f'{v:.2f}' for v in values],
                    textposition='outside'
                ),
                row=idx, col=1
            )

            fig.update_yaxes(title_text=category_results[0].unit if category_results else '',
                           row=idx, col=1)

        fig.update_layout(
            title_text=f'SHOC Results - {run.name}',
            showlegend=False,
            height=400 * len(grouped)
        )

        if output.suffix == '.html':
            fig.write_html(output)
            print(f"Interactive chart saved to {output}")
        else:
            fig.write_image(output)
            print(f"Chart saved to {output}")

    @staticmethod
    def interactive_comparison(runs: List[BenchmarkRun], output: Path, filter_pattern: Optional[str] = None):
        """Generate interactive comparison chart"""
        if not HAS_PLOTLY:
            print("Error: plotly required for interactive charts", file=sys.stderr)
            return

        # Find common benchmarks
        common_names = set(r.name for r in runs[0].results)
        for run in runs[1:]:
            common_names &= set(r.name for r in run.results)

        if filter_pattern:
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            common_names = {n for n in common_names if pattern.search(n)}

        common_names = sorted(common_names)

        if not common_names:
            print("No common benchmarks found", file=sys.stderr)
            return

        fig = go.Figure()

        for run in runs:
            result_dict = {r.name: r.value for r in run.results}
            values = [result_dict.get(name, 0) for name in common_names]

            fig.add_trace(go.Bar(
                name=run.name,
                x=common_names,
                y=values,
                text=[f'{v:.2f}' for v in values],
                textposition='outside'
            ))

        fig.update_layout(
            title='Benchmark Comparison',
            xaxis_title='Benchmark',
            yaxis_title='Performance',
            barmode='group',
            height=600
        )

        if output.suffix == '.html':
            fig.write_html(output)
            print(f"Interactive comparison saved to {output}")
        else:
            fig.write_image(output)
            print(f"Comparison saved to {output}")

def main():
    parser = argparse.ArgumentParser(
        description='SHOC Visualization Tool - Generate charts from benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run bar chart
  %(prog)s results.csv --chart bar --output report.png

  # Compare two runs
  %(prog)s baseline.csv current.csv --compare --output comparison.png

  # Speedup analysis
  %(prog)s baseline.json optimized.json --speedup --output speedup.png

  # Interactive HTML chart
  %(prog)s results.json --chart bar --output report.html --interactive

  # Filter specific benchmarks
  %(prog)s results.csv --chart bar --filter "fft|gemm" --output filtered.png
        """
    )

    parser.add_argument('input', nargs='+', type=Path,
                       help='Input file(s): CSV or JSON from shocdriver')
    parser.add_argument('--output', '-o', type=Path, required=True,
                       help='Output file (.png, .svg, .pdf, .html)')
    parser.add_argument('--chart', choices=['bar', 'line'],
                       default='bar', help='Chart type (default: bar)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple runs (requires 2+ inputs)')
    parser.add_argument('--speedup', action='store_true',
                       help='Generate speedup chart (requires exactly 2 inputs)')
    parser.add_argument('--filter', type=str,
                       help='Filter benchmarks by regex pattern')
    parser.add_argument('--interactive', action='store_true',
                       help='Generate interactive plot (requires plotly)')

    args = parser.parse_args()

    # Validate dependencies
    if args.interactive and not HAS_PLOTLY:
        print("Error: --interactive requires plotly. Install with: pip install plotly", file=sys.stderr)
        sys.exit(1)

    if not args.interactive and not HAS_MATPLOTLIB:
        print("Error: matplotlib required. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # Load data
    runs = [DataLoader.load(f) for f in args.input]

    # Generate visualization
    if args.speedup:
        if len(runs) != 2:
            print("Error: --speedup requires exactly 2 input files", file=sys.stderr)
            sys.exit(1)
        MatplotlibVisualizer.speedup_chart(runs[0], runs[1], args.output, args.filter)

    elif args.compare:
        if len(runs) < 2:
            print("Error: --compare requires 2+ input files", file=sys.stderr)
            sys.exit(1)

        if args.interactive:
            PlotlyVisualizer.interactive_comparison(runs, args.output, args.filter)
        else:
            MatplotlibVisualizer.comparison_chart(runs, args.output, args.filter)

    else:
        # Single run visualization
        if len(runs) > 1:
            print("Warning: Multiple inputs provided but no --compare. Using first file only.", file=sys.stderr)

        if args.interactive:
            PlotlyVisualizer.interactive_bar(runs[0], args.output, args.filter)
        else:
            MatplotlibVisualizer.bar_chart(runs[0], args.output, args.filter)

if __name__ == '__main__':
    main()
