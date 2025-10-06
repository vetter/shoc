#!/usr/bin/env python3
"""
SHOC Driver - Modern Python replacement for driver.pl

Runs SHOC benchmarks, collects results, and generates reports.
Supports YAML/JSON configuration files, parallel execution, and progress bars.

Usage:
    ./shocdriver.py --backend cuda --size 2
    ./shocdriver.py --config myconfig.yaml
    ./shocdriver.py --backend opencl --device 0,1 --parallel
"""

import argparse
import subprocess
import sys
import os
import re
import json
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Callable
import concurrent.futures
from enum import Enum

# Optional: Progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install 'tqdm' for progress bars: pip install tqdm", file=sys.stderr)

# ANSI colors
class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def colored(text, color):
    """Add color if stdout is a TTY"""
    if sys.stdout.isatty():
        return f"{color}{text}{Color.RESET}"
    return text

# Result extraction strategies
class ResultStrategy(Enum):
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    ANYMAX = "anymax"
    ANYMEAN = "anymean"

@dataclass
class ResultSpec:
    """Specification for extracting a result from benchmark output"""
    name: str
    strategy: ResultStrategy
    pattern: str

@dataclass
class BenchmarkSpec:
    """Specification for a single benchmark"""
    program: str
    cuda: bool = True
    opencl: bool = True
    tp: bool = False  # Truly Parallel (requires TP MPI)
    results: List[ResultSpec] = field(default_factory=list)

@dataclass
class BenchmarkResult:
    """Result from running a benchmark"""
    benchmark: str
    device: str
    status: str  # "success", "error", "skipped"
    results: Dict[str, Tuple[float, str]] = field(default_factory=dict)  # {name: (value, unit)}
    log_file: Optional[str] = None
    error_file: Optional[str] = None
    runtime: float = 0.0

class BenchmarkDatabase:
    """Database of all SHOC benchmarks with result extraction specifications"""

    @staticmethod
    def get_serial_benchmarks() -> List[BenchmarkSpec]:
        """Serial benchmark specifications"""
        return [
            BenchmarkSpec("BusSpeedDownload", True, True, False, [
                ResultSpec("bspeed_download", ResultStrategy.MAX, "DownloadSpeed")
            ]),
            BenchmarkSpec("BusSpeedReadback", True, True, False, [
                ResultSpec("bspeed_readback", ResultStrategy.MAX, "ReadbackSpeed")
            ]),
            BenchmarkSpec("MaxFlops", True, True, False, [
                ResultSpec("maxspflops", ResultStrategy.ANYMAX, "-SP"),
                ResultSpec("maxdpflops", ResultStrategy.ANYMAX, "-DP")
            ]),
            BenchmarkSpec("DeviceMemory", True, True, False, [
                ResultSpec("gmem_readbw", ResultStrategy.MAX, "readGlobalMemoryCoalesced"),
                ResultSpec("gmem_readbw_strided", ResultStrategy.MAX, "readGlobalMemoryUnit"),
                ResultSpec("gmem_writebw", ResultStrategy.MAX, "writeGlobalMemoryCoalesced"),
                ResultSpec("gmem_writebw_strided", ResultStrategy.MAX, "writeGlobalMemoryUnit"),
                ResultSpec("lmem_readbw", ResultStrategy.MAX, "readLocalMemory"),
                ResultSpec("lmem_writebw", ResultStrategy.MAX, "writeLocalMemory"),
                ResultSpec("tex_readbw", ResultStrategy.MAX, "TextureRepeatedRandomAccess")
            ]),
            BenchmarkSpec("KernelCompile", False, True, False, [
                ResultSpec("ocl_kernel", ResultStrategy.MIN, "BuildProgram")
            ]),
            BenchmarkSpec("QueueDelay", False, True, False, [
                ResultSpec("ocl_queue", ResultStrategy.MIN, "SSDelay")
            ]),
            BenchmarkSpec("BFS", True, True, False, [
                ResultSpec("bfs", ResultStrategy.MAX, "BFS"),
                ResultSpec("bfs_pcie", ResultStrategy.MAX, "BFS_PCIe"),
                ResultSpec("bfs_teps", ResultStrategy.MAX, "BFS_teps")
            ]),
            BenchmarkSpec("FFT", True, True, False, [
                ResultSpec("fft_sp", ResultStrategy.MAX, "SP-FFT"),
                ResultSpec("fft_sp_pcie", ResultStrategy.MAX, "SP-FFT_PCIe"),
                ResultSpec("ifft_sp", ResultStrategy.MAX, "SP-FFT-INV"),
                ResultSpec("ifft_sp_pcie", ResultStrategy.MAX, "SP-FFT-INV_PCIe"),
                ResultSpec("fft_dp", ResultStrategy.MAX, "DP-FFT"),
                ResultSpec("fft_dp_pcie", ResultStrategy.MAX, "DP-FFT_PCIe"),
                ResultSpec("ifft_dp", ResultStrategy.MAX, "DP-FFT-INV"),
                ResultSpec("ifft_dp_pcie", ResultStrategy.MAX, "DP-FFT-INV_PCIe")
            ]),
            BenchmarkSpec("GEMM", True, True, False, [
                ResultSpec("sgemm_n", ResultStrategy.MAX, "SGEMM-N"),
                ResultSpec("sgemm_t", ResultStrategy.MAX, "SGEMM-T"),
                ResultSpec("sgemm_n_pcie", ResultStrategy.MAX, "SGEMM-N_PCIe"),
                ResultSpec("sgemm_t_pcie", ResultStrategy.MAX, "SGEMM-T_PCIe"),
                ResultSpec("dgemm_n", ResultStrategy.MAX, "DGEMM-N"),
                ResultSpec("dgemm_t", ResultStrategy.MAX, "DGEMM-T"),
                ResultSpec("dgemm_n_pcie", ResultStrategy.MAX, "DGEMM-N_PCIe"),
                ResultSpec("dgemm_t_pcie", ResultStrategy.MAX, "DGEMM-T_PCIe")
            ]),
            BenchmarkSpec("MD", True, True, False, [
                ResultSpec("md_sp_flops", ResultStrategy.MAX, "MD-LJ"),
                ResultSpec("md_sp_bw", ResultStrategy.MAX, "MD-LJ-Bandwidth"),
                ResultSpec("md_sp_flops_pcie", ResultStrategy.MAX, "MD-LJ_PCIe"),
                ResultSpec("md_sp_bw_pcie", ResultStrategy.MAX, "MD-LJ-Bandwidth_PCIe"),
                ResultSpec("md_dp_flops", ResultStrategy.MAX, "MD-LJ-DP"),
                ResultSpec("md_dp_bw", ResultStrategy.MAX, "MD-LJ-DP-Bandwidth"),
                ResultSpec("md_dp_flops_pcie", ResultStrategy.MAX, "MD-LJ-DP_PCIe"),
                ResultSpec("md_dp_bw_pcie", ResultStrategy.MAX, "MD-LJ-DP-Bandwidth_PCIe")
            ]),
            BenchmarkSpec("MD5Hash", True, True, False, [
                ResultSpec("md5hash", ResultStrategy.MAX, "MD5Hash")
            ]),
            BenchmarkSpec("NeuralNet", True, False, False, [
                ResultSpec("nn_learning", ResultStrategy.MEAN, "Learning-Rate"),
                ResultSpec("nn_learning_pcie", ResultStrategy.MEAN, "Learning-Rate_PCIe")
            ]),
            BenchmarkSpec("Reduction", True, True, False, [
                ResultSpec("reduction", ResultStrategy.MAX, "Reduction"),
                ResultSpec("reduction_pcie", ResultStrategy.MAX, "Reduction_PCIe"),
                ResultSpec("reduction_dp", ResultStrategy.MAX, "Reduction-DP"),
                ResultSpec("reduction_dp_pcie", ResultStrategy.MAX, "Reduction-DP_PCIe")
            ]),
            BenchmarkSpec("Scan", True, True, False, [
                ResultSpec("scan", ResultStrategy.MAX, "Scan"),
                ResultSpec("scan_pcie", ResultStrategy.MAX, "Scan_PCIe"),
                ResultSpec("scan_dp", ResultStrategy.MAX, "Scan-DP"),
                ResultSpec("scan_dp_pcie", ResultStrategy.MAX, "Scan-DP_PCIe")
            ]),
            BenchmarkSpec("Sort", True, True, False, [
                ResultSpec("sort", ResultStrategy.MAX, "Sort-Rate"),
                ResultSpec("sort_pcie", ResultStrategy.MAX, "Sort-Rate_PCIe")
            ]),
            BenchmarkSpec("Spmv", True, True, False, [
                ResultSpec("spmv_csr_scalar_sp", ResultStrategy.MAX, "CSR-Scalar-SP"),
                ResultSpec("spmv_csr_vector_sp", ResultStrategy.MAX, "CSR-Vector-SP"),
                ResultSpec("spmv_ellpackr_sp", ResultStrategy.MAX, "ELLPACKR-SP"),
                ResultSpec("spmv_csr_scalar_dp", ResultStrategy.MAX, "CSR-Scalar-DP"),
                ResultSpec("spmv_csr_vector_dp", ResultStrategy.MAX, "CSR-Vector-DP"),
                ResultSpec("spmv_ellpackr_dp", ResultStrategy.MAX, "ELLPACKR-DP")
            ]),
            BenchmarkSpec("Stencil2D", True, True, False, [
                ResultSpec("stencil", ResultStrategy.MAX, "SP_Sten2D"),
                ResultSpec("stencil_dp", ResultStrategy.MAX, "DP_Sten2D")
            ]),
            BenchmarkSpec("Triad", True, True, False, [
                ResultSpec("triad_bw", ResultStrategy.MAX, "TriadBdwth")
            ]),
            BenchmarkSpec("S3D", True, True, False, [
                ResultSpec("s3d", ResultStrategy.MAX, "S3D-SP"),
                ResultSpec("s3d_pcie", ResultStrategy.MAX, "S3D-SP_PCIe"),
                ResultSpec("s3d_dp", ResultStrategy.MAX, "S3D-DP"),
                ResultSpec("s3d_dp_pcie", ResultStrategy.MAX, "S3D-DP_PCIe")
            ]),
        ]

    @staticmethod
    def get_parallel_benchmarks() -> List[BenchmarkSpec]:
        """Parallel (EP/TP) benchmark specifications"""
        # Similar to serial but with different result patterns (mean instead of max)
        benchmarks = BenchmarkDatabase.get_serial_benchmarks()
        # Adjust patterns for parallel execution (add "(max)" or "(min)" suffixes)
        for bench in benchmarks:
            for result in bench.results:
                if result.strategy == ResultStrategy.MAX:
                    result.strategy = ResultStrategy.MEAN
                    if not result.pattern.endswith("(max)"):
                        result.pattern = f"{result.pattern}(max)"
                elif result.strategy == ResultStrategy.MIN:
                    result.strategy = ResultStrategy.MEAN
                    if not result.pattern.endswith("(min)"):
                        result.pattern = f"{result.pattern}(min)"
                elif result.strategy == ResultStrategy.ANYMAX:
                    result.strategy = ResultStrategy.ANYMEAN
                    result.pattern = result.pattern.replace("-SP", "-SP\\\\(max\\\\)")
                    result.pattern = result.pattern.replace("-DP", "-DP\\\\(max\\\\)")

        # Add TP-specific benchmarks
        benchmarks.append(BenchmarkSpec("Stencil2D", True, True, True, [
            ResultSpec("stencil", ResultStrategy.MEAN, "SP_Sten2D(max)"),
            ResultSpec("stencil_dp", ResultStrategy.MEAN, "DP_Sten2D(max)")
        ]))
        benchmarks.append(BenchmarkSpec("QTC", True, False, True, [
            ResultSpec("qtc", ResultStrategy.MIN, "QTC+PCI_Trans.(min)"),
            ResultSpec("qtc_kernel", ResultStrategy.MIN, "QTC_Kernel(min)")
        ]))

        return benchmarks

class ResultExtractor:
    """Extracts results from benchmark log files using various strategies"""

    @staticmethod
    def extract(log_file: Path, result_spec: ResultSpec) -> Tuple[Optional[float], str]:
        """Extract a result from a log file"""
        if not log_file.exists():
            return None, ""

        strategy_map = {
            ResultStrategy.MAX: ResultExtractor.find_max,
            ResultStrategy.MIN: ResultExtractor.find_min,
            ResultStrategy.MEAN: ResultExtractor.find_mean,
            ResultStrategy.ANYMAX: ResultExtractor.find_anymax,
            ResultStrategy.ANYMEAN: ResultExtractor.find_anymean,
        }

        extractor = strategy_map[result_spec.strategy]
        return extractor(log_file, result_spec.pattern)

    @staticmethod
    def find_max(log_file: Path, pattern: str) -> Tuple[Optional[float], str]:
        """Find maximum value for a test"""
        best = -1.0
        unit = ""

        with open(log_file) as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens or tokens[0] != pattern:
                    continue

                if len(tokens) > 2:
                    unit = tokens[2]

                # Column 7+ contains trial values
                for i in range(7, len(tokens)):
                    try:
                        val = float(tokens[i])
                        if not (val != val or val == float('inf')):  # Skip nan/inf
                            best = max(best, val)
                    except (ValueError, IndexError):
                        pass

        return ResultExtractor._check_error(best), unit

    @staticmethod
    def find_min(log_file: Path, pattern: str) -> Tuple[Optional[float], str]:
        """Find minimum value for a test"""
        best = 1e37
        unit = ""

        with open(log_file) as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens or tokens[0] != pattern:
                    continue

                if len(tokens) > 2:
                    unit = tokens[2]

                # Column 6 contains min value
                if len(tokens) > 6:
                    try:
                        val = float(tokens[6])
                        best = min(best, val)
                    except ValueError:
                        pass

        return ResultExtractor._check_error(best), unit

    @staticmethod
    def find_mean(log_file: Path, pattern: str) -> Tuple[Optional[float], str]:
        """Find mean value for a test"""
        best = -1.0
        unit = ""

        with open(log_file) as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens or tokens[0] != pattern:
                    continue

                if len(tokens) > 2:
                    unit = tokens[2]

                # Column 4 contains mean value
                if len(tokens) > 4:
                    try:
                        val = float(tokens[4])
                        best = max(best, val)
                    except ValueError:
                        pass

        return ResultExtractor._check_error(best), unit

    @staticmethod
    def find_anymax(log_file: Path, pattern: str) -> Tuple[Optional[float], str]:
        """Find max value for any test matching pattern"""
        best = -1.0
        unit = ""
        header_found = False

        with open(log_file) as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue

                if tokens[0] == "test":
                    header_found = True
                    continue

                if header_found and re.search(pattern, tokens[0]):
                    if len(tokens) > 2:
                        unit = tokens[2]
                    if len(tokens) > 7:
                        try:
                            val = float(tokens[7])
                            best = max(best, val)
                        except ValueError:
                            pass

        return ResultExtractor._check_error(best), unit

    @staticmethod
    def find_anymean(log_file: Path, pattern: str) -> Tuple[Optional[float], str]:
        """Find mean value for any test matching pattern"""
        best = -1.0
        unit = ""
        header_found = False

        with open(log_file) as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue

                if tokens[0] == "test":
                    header_found = True
                    continue

                if header_found and re.search(pattern, tokens[0]):
                    if len(tokens) > 2:
                        unit = tokens[2]
                    if len(tokens) > 4:
                        try:
                            val = float(tokens[4])
                            best = max(best, val)
                        except ValueError:
                            pass

        return ResultExtractor._check_error(best), unit

    @staticmethod
    def _check_error(value: float) -> Optional[float]:
        """Check if value represents an error"""
        if value is None:
            return None
        if value == -1.0 or value == 0.0 or value >= 1e37 or value != value:
            return None
        return value

class SHOCDriver:
    """Main driver class for running SHOC benchmarks"""

    def __init__(self, config: Dict):
        self.config = config
        self.bin_dir = Path(config.get('bin_dir', './bin'))
        self.log_dir = Path(config.get('log_dir', './Logs'))
        self.backend = config['backend']  # 'cuda' or 'opencl'
        self.size_class = config.get('size', 1)
        self.devices = config.get('devices', '0')
        self.platform = config.get('platform', '0')
        self.num_nodes = config.get('num_nodes', 1)
        self.num_devices = len(str(self.devices).split(','))
        self.num_tasks = self.num_devices * self.num_nodes
        self.hostfile = config.get('hostfile', '')
        self.single_benchmark = config.get('benchmark', '')
        self.parallel_exec = config.get('parallel_execution', False)
        self.max_workers = config.get('max_workers', 4)

        self.log_dir.mkdir(exist_ok=True)

        # Choose benchmark set
        if self.num_tasks == 1:
            self.benchmarks = BenchmarkDatabase.get_serial_benchmarks()
            self.category = "Serial"
        else:
            self.benchmarks = BenchmarkDatabase.get_parallel_benchmarks()
            self.category = "EP"  # Default to EP, TP is special

    def validate(self):
        """Validate configuration"""
        if not self.bin_dir.exists():
            raise ValueError(f"Binary directory not found: {self.bin_dir}")

        serial_dir = self.bin_dir / "Serial"
        if not serial_dir.exists():
            raise ValueError(f"Not a SHOC binary directory: {self.bin_dir}")

        # Check for at least one benchmark
        backend_upper = self.backend.upper()
        test_bench = serial_dir / backend_upper / "Sort"
        if not test_bench.exists():
            raise ValueError(f"SHOC benchmarks not found in {self.bin_dir}")

    def build_command(self, benchmark: BenchmarkSpec) -> Tuple[str, Path, Path]:
        """Build command to run a benchmark"""
        log_base = self.log_dir / f"dev{self.devices}_{benchmark.program}"
        log_file = log_base.with_suffix('.log')
        err_file = log_base.with_suffix('.err')

        cmd_parts = []

        # MPI launcher for parallel
        if self.num_tasks > 1:
            hostname = subprocess.check_output(['hostname', '-A'], text=True).strip()
            if 'summit.olcf' in hostname:
                cmd_parts.append(f"jsrun -n {self.num_tasks} -a 1 -c 1 -g {self.num_devices}")
            else:
                cmd_parts.append(f"mpirun -np {self.num_tasks}")
                if self.hostfile:
                    cmd_parts.append(f"-hostfile {self.hostfile}")

        # Benchmark path
        if self.num_tasks == 1:
            bench_path = self.bin_dir / "Serial"
        else:
            if benchmark.tp:
                bench_path = self.bin_dir / "TP"
            else:
                bench_path = self.bin_dir / "EP"

        bench_path = bench_path / self.backend.upper() / benchmark.program
        cmd_parts.append(str(bench_path))

        # Arguments
        cmd_parts.append(f"-s {self.size_class}")
        if self.backend == "opencl" and self.platform:
            cmd_parts.append(f"-p {self.platform}")
        if self.devices:
            cmd_parts.append(f"-d {self.devices}")

        # Redirection
        cmd_parts.append(f"> {log_file}")
        cmd_parts.append(f"2> {err_file}")

        return ' '.join(cmd_parts), log_file, err_file

    def run_benchmark(self, benchmark: BenchmarkSpec) -> BenchmarkResult:
        """Run a single benchmark and collect results"""
        import time

        # Check if benchmark supports this backend
        if self.backend == "cuda" and not benchmark.cuda:
            return BenchmarkResult(benchmark.program, self.devices, "skipped")
        if self.backend == "opencl" and not benchmark.opencl:
            return BenchmarkResult(benchmark.program, self.devices, "skipped")

        # Check single benchmark filter
        if self.single_benchmark and benchmark.program != self.single_benchmark:
            return BenchmarkResult(benchmark.program, self.devices, "skipped")

        cmd, log_file, err_file = self.build_command(benchmark)

        start = time.time()
        try:
            result = subprocess.run(cmd, shell=True, timeout=600)
            runtime = time.time() - start

            if result.returncode != 0:
                return BenchmarkResult(
                    benchmark.program, self.devices, "error",
                    log_file=str(log_file), error_file=str(err_file),
                    runtime=runtime
                )

            # Extract results
            results = {}
            for result_spec in benchmark.results:
                value, unit = ResultExtractor.extract(log_file, result_spec)
                if value is not None:
                    results[result_spec.name] = (value, unit)

            return BenchmarkResult(
                benchmark.program, self.devices, "success",
                results=results, log_file=str(log_file),
                error_file=str(err_file), runtime=runtime
            )

        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                benchmark.program, self.devices, "timeout",
                log_file=str(log_file), error_file=str(err_file),
                runtime=time.time() - start
            )
        except Exception as e:
            return BenchmarkResult(
                benchmark.program, self.devices, "error",
                log_file=str(log_file), error_file=str(err_file),
                runtime=time.time() - start
            )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks"""
        results = []

        benchmarks_to_run = [b for b in self.benchmarks
                            if self.should_run_benchmark(b)]

        if HAS_TQDM:
            progress = tqdm(benchmarks_to_run, desc="Running benchmarks", unit="bench")
        else:
            progress = benchmarks_to_run

        if self.parallel_exec and len(benchmarks_to_run) > 1:
            # Run benchmarks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_bench = {executor.submit(self.run_benchmark, b): b
                                  for b in benchmarks_to_run}

                for future in concurrent.futures.as_completed(future_to_bench):
                    bench = future_to_bench[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if HAS_TQDM:
                            progress.update(1)
                        self._print_benchmark_result(result)
                    except Exception as e:
                        print(f"{colored('Error', Color.RED)} running {bench.program}: {e}")
        else:
            # Run sequentially
            for benchmark in progress:
                if HAS_TQDM:
                    progress.set_description(f"Running {benchmark.program}")
                else:
                    print(f"Running benchmark {colored(benchmark.program, Color.CYAN)}")

                result = self.run_benchmark(benchmark)
                results.append(result)
                self._print_benchmark_result(result)

        if HAS_TQDM:
            progress.close()

        return results

    def should_run_benchmark(self, benchmark: BenchmarkSpec) -> bool:
        """Check if benchmark should be run"""
        if self.single_benchmark and benchmark.program != self.single_benchmark:
            return False
        if self.backend == "cuda" and not benchmark.cuda:
            return False
        if self.backend == "opencl" and not benchmark.opencl:
            return False
        return True

    def _print_benchmark_result(self, result: BenchmarkResult):
        """Print result summary"""
        if result.status == "skipped":
            return

        status_colors = {
            "success": Color.GREEN,
            "error": Color.RED,
            "timeout": Color.YELLOW
        }

        status_str = colored(result.status.upper(), status_colors.get(result.status, Color.RESET))
        print(f"  [{status_str}] {result.benchmark} ({result.runtime:.2f}s)")

        for name, (value, unit) in result.results.items():
            print(f"    {name:<40} {value:>10.4f} {unit}")

    def save_results(self, results: List[BenchmarkResult], output_file: Path):
        """Save results to CSV file"""
        # Collect all result names in order
        all_result_names = set()
        for result in results:
            all_result_names.update(result.results.keys())
        result_names = sorted(all_result_names)

        with open(output_file, 'w') as f:
            # Header
            f.write(','.join(result_names) + '\n')

            # Data row
            values = []
            for name in result_names:
                found = False
                for result in results:
                    if name in result.results:
                        values.append(str(result.results[name][0]))
                        found = True
                        break
                if not found:
                    values.append('')
            f.write(','.join(values) + '\n')

        print(f"\nResults saved to {colored(str(output_file), Color.BOLD)}")

    def save_json_report(self, results: List[BenchmarkResult], output_file: Path):
        """Save detailed JSON report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': [asdict(r) for r in results]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Detailed report saved to {colored(str(output_file), Color.BOLD)}")

def load_config(config_file: Path) -> Dict:
    """Load configuration from YAML or JSON file"""
    with open(config_file) as f:
        if config_file.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_file.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")

def main():
    parser = argparse.ArgumentParser(
        description='SHOC Driver - Run SHOC benchmarks and collect results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --backend cuda --size 2
  %(prog)s --backend opencl --device 0,1 --size 3
  %(prog)s --config experiments/baseline.yaml
  %(prog)s --backend cuda --benchmark FFT --size 1
  %(prog)s --backend cuda --parallel --max-workers 8
        """
    )

    parser.add_argument('--config', type=Path, help='YAML/JSON configuration file')
    parser.add_argument('--backend', choices=['cuda', 'opencl'], help='Backend to use')
    parser.add_argument('--size', type=int, choices=[1,2,3,4], help='Problem size class')
    parser.add_argument('--device', '--devices', dest='devices', help='Device IDs (comma-separated)')
    parser.add_argument('--platform', help='OpenCL platform ID')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes (for MPI)')
    parser.add_argument('--hostfile', help='MPI hostfile')
    parser.add_argument('--benchmark', help='Run single benchmark')
    parser.add_argument('--bin-dir', type=Path, default=Path('./bin'), help='SHOC binary directory')
    parser.add_argument('--log-dir', type=Path, default=Path('./Logs'), help='Log output directory')
    parser.add_argument('--output', type=Path, default=Path('./results.csv'), help='Output CSV file')
    parser.add_argument('--json-report', type=Path, help='Save detailed JSON report')
    parser.add_argument('--parallel', dest='parallel_execution', action='store_true',
                       help='Run benchmarks in parallel')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Max parallel workers (default: 4)')

    args = parser.parse_args()

    # Load config from file or command line
    if args.config:
        config = load_config(args.config)
    else:
        if not args.backend or args.size is None:
            parser.error("--backend and --size required (or use --config)")

        config = {
            'backend': args.backend,
            'size': args.size,
            'devices': args.devices or '0',
            'platform': args.platform or '0',
            'num_nodes': args.num_nodes,
            'hostfile': args.hostfile or '',
            'benchmark': args.benchmark or '',
            'bin_dir': str(args.bin_dir),
            'log_dir': str(args.log_dir),
            'parallel_execution': args.parallel_execution,
            'max_workers': args.max_workers
        }

    # Run driver
    print(colored(f"{'='*80}", Color.BOLD))
    print(colored("SHOC Benchmark Driver", Color.HEADER + Color.BOLD))
    print(colored(f"{'='*80}", Color.BOLD))
    print()

    driver = SHOCDriver(config)
    driver.validate()

    print(f"Backend: {colored(driver.backend.upper(), Color.CYAN)}")
    print(f"Size class: {colored(str(driver.size_class), Color.CYAN)}")
    print(f"Devices: {colored(driver.devices, Color.CYAN)}")
    print(f"Tasks: {colored(str(driver.num_tasks), Color.CYAN)}")
    print()

    results = driver.run_all()

    # Save results
    driver.save_results(results, args.output)

    if args.json_report:
        driver.save_json_report(results, args.json_report)

    # Summary
    print()
    print(colored(f"{'='*80}", Color.BOLD))
    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    errors = sum(1 for r in results if r.status == "error")
    timeouts = sum(1 for r in results if r.status == "timeout")

    print(f"Summary: {colored(str(success), Color.GREEN)} succeeded, "
          f"{colored(str(errors), Color.RED)} errors, "
          f"{colored(str(timeouts), Color.YELLOW)} timeouts "
          f"(of {total} total)")
    print(colored(f"{'='*80}", Color.BOLD))

if __name__ == '__main__':
    main()
