#!/usr/bin/env python3
"""
SHOC Benchmark Discovery Tool

Scans SHOC installation directory and lists all available benchmarks
with descriptions, categorization, and filtering capabilities.

Usage:
    ./list_benchmarks.py [install_dir]
    ./list_benchmarks.py --backend cuda
    ./list_benchmarks.py --category Serial --format markdown
    ./list_benchmarks.py --search fft --format json
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

# Benchmark descriptions database
BENCHMARK_DESCRIPTIONS = {
    # Level 0 - Device Capabilities
    'BusSpeedDownload': 'Measures PCIe bus bandwidth from host to device',
    'BusSpeedReadback': 'Measures PCIe bus bandwidth from device to host',
    'DeviceMemory': 'Tests device memory bandwidth and latency characteristics',
    'KernelCompile': 'Measures kernel compilation time',
    'MaxFlops': 'Determines peak floating-point performance (GFLOPS)',
    'QueueDelay': 'Measures command queue latency overhead',

    # Level 1 - Algorithm Performance
    'BFS': 'Breadth-First Search graph traversal algorithm',
    'FFT': 'Fast Fourier Transform (1D, forward and inverse)',
    'GEMM': 'General Matrix Multiply (dense linear algebra)',
    'MD': 'Molecular Dynamics N-body simulation',
    'MD5Hash': 'MD5 cryptographic hash computation',
    'Reduction': 'Parallel reduction (sum) operation',
    'Scan': 'Parallel prefix sum (scan) operation',
    'Sort': 'Radix sort algorithm for integer arrays',
    'Spmv': 'Sparse Matrix-Vector multiplication',
    'Stencil2D': '2D stencil computation (9-point)',
    'Triad': 'STREAM Triad memory bandwidth test (A = B + s*C)',
    'NeuralNet': 'Neural network layer computation',

    # Level 2 - Application Kernels
    'S3D': 'S3D combustion chemistry kernel',
    'QTC': 'Quantum Chemistry CCSD(T) tensor contractions',
}

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def colorize(text, color):
    """Add color to text if stdout is a TTY"""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text

def find_install_dir(provided_path=None):
    """Find SHOC installation directory"""
    if provided_path:
        path = Path(provided_path)
        if path.exists() and path.is_dir():
            return path
        else:
            print(f"Error: Provided path '{provided_path}' does not exist or is not a directory")
            sys.exit(1)

    # Auto-detect common installation locations
    candidates = [
        Path('install'),
        Path('install-full'),
        Path('install-cuda'),
        Path('install-opencl'),
        Path('../install'),
        Path('/usr/local/shoc'),
        Path.home() / 'shoc' / 'install',
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / 'bin').exists():
            return candidate

    print("Error: Could not find SHOC installation directory.")
    print("Please specify installation directory as argument:")
    print("  ./list_benchmarks.py <install_dir>")
    print("\nSearched locations:")
    for candidate in candidates:
        print(f"  - {candidate}")
    sys.exit(1)

def scan_benchmarks(install_dir):
    """Scan installation directory for benchmarks"""
    bin_dir = install_dir / 'bin'
    if not bin_dir.exists():
        print(f"Error: bin directory not found in {install_dir}")
        sys.exit(1)

    benchmarks = []

    # Scan all category/backend combinations
    for category_dir in bin_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name  # Serial, EP, TP

        for backend_dir in category_dir.iterdir():
            if not backend_dir.is_dir():
                continue

            backend = backend_dir.name  # CUDA, OpenCL

            # Find all executables
            for executable in backend_dir.iterdir():
                if executable.is_file() and os.access(executable, os.X_OK):
                    name = executable.name
                    benchmarks.append({
                        'name': name,
                        'backend': backend,
                        'category': category,
                        'path': str(executable),
                        'description': BENCHMARK_DESCRIPTIONS.get(name, 'No description available')
                    })

    return sorted(benchmarks, key=lambda x: (x['backend'], x['category'], x['name']))

def filter_benchmarks(benchmarks, backend=None, category=None, search=None):
    """Filter benchmarks based on criteria"""
    filtered = benchmarks

    if backend:
        filtered = [b for b in filtered if b['backend'].lower() == backend.lower()]

    if category:
        filtered = [b for b in filtered if b['category'].lower() == category.lower()]

    if search:
        search_lower = search.lower()
        filtered = [b for b in filtered
                   if search_lower in b['name'].lower()
                   or search_lower in b['description'].lower()]

    return filtered

def format_human_readable(benchmarks, install_dir):
    """Format benchmarks as human-readable text"""
    if not benchmarks:
        return colorize("No benchmarks found matching criteria.", Colors.YELLOW)

    output = []
    output.append(colorize(f"\n{'='*80}", Colors.BOLD))
    output.append(colorize(f"SHOC Benchmarks - {install_dir}", Colors.HEADER + Colors.BOLD))
    output.append(colorize(f"{'='*80}\n", Colors.BOLD))

    # Group by backend and category
    grouped = defaultdict(lambda: defaultdict(list))
    for bench in benchmarks:
        grouped[bench['backend']][bench['category']].append(bench)

    for backend in sorted(grouped.keys()):
        output.append(colorize(f"\n{backend}", Colors.BLUE + Colors.BOLD))
        output.append(colorize("-" * len(backend), Colors.BLUE))

        for category in sorted(grouped[backend].keys()):
            output.append(colorize(f"\n  {category}:", Colors.CYAN + Colors.BOLD))

            for bench in grouped[backend][category]:
                name = colorize(f"    • {bench['name']:<20}", Colors.GREEN)
                desc = bench['description']
                output.append(f"{name} {desc}")

    # Statistics
    total = len(benchmarks)
    by_backend = defaultdict(int)
    by_category = defaultdict(int)
    for bench in benchmarks:
        by_backend[bench['backend']] += 1
        by_category[bench['category']] += 1

    output.append(colorize(f"\n{'='*80}", Colors.BOLD))
    output.append(colorize("Statistics:", Colors.HEADER + Colors.BOLD))
    output.append(f"  Total benchmarks: {colorize(str(total), Colors.BOLD)}")
    output.append(f"  By backend: {', '.join(f'{k}={colorize(str(v), Colors.BOLD)}' for k, v in sorted(by_backend.items()))}")
    output.append(f"  By category: {', '.join(f'{k}={colorize(str(v), Colors.BOLD)}' for k, v in sorted(by_category.items()))}")
    output.append(colorize(f"{'='*80}\n", Colors.BOLD))

    return '\n'.join(output)

def format_markdown(benchmarks, install_dir):
    """Format benchmarks as Markdown table"""
    if not benchmarks:
        return "No benchmarks found matching criteria."

    output = []
    output.append(f"# SHOC Benchmarks - {install_dir}\n")
    output.append("| Benchmark | Backend | Category | Description |")
    output.append("|-----------|---------|----------|-------------|")

    for bench in benchmarks:
        output.append(f"| {bench['name']} | {bench['backend']} | {bench['category']} | {bench['description']} |")

    output.append(f"\n**Total: {len(benchmarks)} benchmarks**")

    return '\n'.join(output)

def format_json(benchmarks, install_dir):
    """Format benchmarks as JSON"""
    data = {
        'install_dir': str(install_dir),
        'total_count': len(benchmarks),
        'benchmarks': benchmarks
    }
    return json.dumps(data, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description='SHOC Benchmark Discovery Tool - List and describe all SHOC benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Auto-detect installation and list all
  %(prog)s install-cuda                 # Scan specific installation
  %(prog)s --backend cuda               # Show only CUDA benchmarks
  %(prog)s --category Serial            # Show only Serial benchmarks
  %(prog)s --search fft                 # Search for FFT-related benchmarks
  %(prog)s --format markdown > list.md  # Export as Markdown table
  %(prog)s --format json > list.json    # Export as JSON
        """
    )

    parser.add_argument('install_dir', nargs='?', default=None,
                       help='SHOC installation directory (auto-detects if not provided)')
    parser.add_argument('--backend', choices=['cuda', 'opencl', 'CUDA', 'OpenCL'],
                       help='Filter by backend (CUDA or OpenCL)')
    parser.add_argument('--category', choices=['serial', 'ep', 'tp', 'Serial', 'EP', 'TP'],
                       help='Filter by category (Serial, EP, or TP)')
    parser.add_argument('--search', type=str,
                       help='Search benchmarks by name or description')
    parser.add_argument('--format', choices=['human', 'markdown', 'json'], default='human',
                       help='Output format (default: human)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Show only statistics')

    args = parser.parse_args()

    # Find installation directory
    install_dir = find_install_dir(args.install_dir)

    # Scan for benchmarks
    benchmarks = scan_benchmarks(install_dir)

    # Filter
    benchmarks = filter_benchmarks(benchmarks, args.backend, args.category, args.search)

    # Stats only mode
    if args.stats_only:
        by_backend = defaultdict(int)
        by_category = defaultdict(int)
        for bench in benchmarks:
            by_backend[bench['backend']] += 1
            by_category[bench['category']] += 1

        print(f"Total: {len(benchmarks)}")
        print(f"Backends: {dict(by_backend)}")
        print(f"Categories: {dict(by_category)}")
        return

    # Format and output
    if args.format == 'human':
        print(format_human_readable(benchmarks, install_dir))
    elif args.format == 'markdown':
        print(format_markdown(benchmarks, install_dir))
    elif args.format == 'json':
        print(format_json(benchmarks, install_dir))

if __name__ == '__main__':
    main()
