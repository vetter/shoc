#!/usr/bin/env python3
"""
SHOC Configuration Runner

Advanced configuration-based benchmark orchestration with support for:
- Benchmark suites
- Parameter sweeps
- Test matrices
- Experiment workflows
- Automated result collection

Usage:
    ./shocrun.py --config experiments/baseline-suite.yaml
    ./shocrun.py --config experiments/parameter-sweep.yaml
    ./shocrun.py --config experiments/regression-tests.yaml
"""

import argparse
import yaml
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
import shutil

@dataclass
class BenchmarkConfig:
    """Single benchmark run configuration"""
    name: str
    backend: str
    size: int
    devices: str = "0"
    platform: str = "0"
    num_nodes: int = 1
    hostfile: str = ""
    benchmark: str = ""
    bin_dir: str = "./bin"
    log_dir: str = "./Logs"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    """Result from a benchmark experiment"""
    config: BenchmarkConfig
    success: bool
    csv_file: Optional[Path] = None
    json_file: Optional[Path] = None
    error: Optional[str] = None
    duration: float = 0.0

class ConfigurationRunner:
    """Execute benchmark configurations and manage results"""

    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.config = self._load_config(config_file)
        self.results: List[ExperimentResult] = []

    def _load_config(self, config_file: Path) -> Dict:
        """Load YAML configuration file"""
        with open(config_file) as f:
            return yaml.safe_load(f)

    def run(self):
        """Execute the configuration"""
        config_type = self.config.get('type', 'single')

        if config_type == 'single':
            self._run_single()
        elif config_type == 'suite':
            self._run_suite()
        elif config_type == 'sweep':
            self._run_sweep()
        elif config_type == 'matrix':
            self._run_matrix()
        elif config_type == 'experiment':
            self._run_experiment()
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")

        self._generate_summary()

    def _run_single(self):
        """Run a single benchmark configuration"""
        bench_config = self._parse_benchmark_config(self.config)
        result = self._execute_benchmark(bench_config)
        self.results.append(result)

    def _run_suite(self):
        """Run a suite of benchmarks"""
        suite_name = self.config.get('name', 'benchmark-suite')
        benchmarks = self.config.get('benchmarks', [])

        print(f"Running suite: {suite_name}")
        print(f"Benchmarks: {len(benchmarks)}")
        print()

        for idx, bench_def in enumerate(benchmarks, 1):
            print(f"[{idx}/{len(benchmarks)}] {bench_def.get('name', 'unnamed')}")

            # Merge suite defaults with benchmark-specific config
            merged_config = self._merge_configs(
                self.config.get('defaults', {}),
                bench_def
            )

            bench_config = self._parse_benchmark_config(merged_config)
            result = self._execute_benchmark(bench_config)
            self.results.append(result)
            print()

    def _run_sweep(self):
        """Run parameter sweep"""
        sweep_name = self.config.get('name', 'parameter-sweep')
        base_config = self.config.get('base', {})
        parameters = self.config.get('parameters', {})

        print(f"Running parameter sweep: {sweep_name}")
        print(f"Parameters: {list(parameters.keys())}")
        print()

        # Generate all combinations
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]
        combinations = list(product(*param_values))

        print(f"Total combinations: {len(combinations)}")
        print()

        for idx, combo in enumerate(combinations, 1):
            # Create config for this combination
            sweep_config = base_config.copy()
            sweep_metadata = {}

            for param_name, param_value in zip(param_names, combo):
                sweep_config[param_name] = param_value
                sweep_metadata[param_name] = param_value

            # Generate descriptive name
            combo_str = '_'.join(f"{k}={v}" for k, v in zip(param_names, combo))
            sweep_config['name'] = f"{sweep_name}_{combo_str}"
            sweep_config['metadata'] = sweep_metadata

            print(f"[{idx}/{len(combinations)}] {combo_str}")

            bench_config = self._parse_benchmark_config(sweep_config)
            bench_config.metadata = sweep_metadata
            result = self._execute_benchmark(bench_config)
            self.results.append(result)
            print()

    def _run_matrix(self):
        """Run test matrix (combinations of discrete options)"""
        matrix_name = self.config.get('name', 'test-matrix')
        base_config = self.config.get('base', {})
        matrix = self.config.get('matrix', {})

        print(f"Running test matrix: {matrix_name}")
        print()

        # Generate matrix combinations
        configs = []
        for matrix_item in matrix:
            test_config = base_config.copy()
            test_config.update(matrix_item)
            configs.append(test_config)

        print(f"Total configurations: {len(configs)}")
        print()

        for idx, test_config in enumerate(configs, 1):
            test_name = test_config.get('name', f'test-{idx}')
            print(f"[{idx}/{len(configs)}] {test_name}")

            bench_config = self._parse_benchmark_config(test_config)
            result = self._execute_benchmark(bench_config)
            self.results.append(result)
            print()

    def _run_experiment(self):
        """Run experiment workflow with phases"""
        exp_name = self.config.get('name', 'experiment')
        phases = self.config.get('phases', [])

        print(f"Running experiment: {exp_name}")
        print(f"Phases: {len(phases)}")
        print()

        for phase_idx, phase in enumerate(phases, 1):
            phase_name = phase.get('name', f'phase-{phase_idx}')
            print(f"\n{'='*80}")
            print(f"Phase {phase_idx}/{len(phases)}: {phase_name}")
            print(f"{'='*80}\n")

            phase_benchmarks = phase.get('benchmarks', [])
            phase_defaults = phase.get('defaults', {})

            for bench_idx, bench_def in enumerate(phase_benchmarks, 1):
                print(f"  [{bench_idx}/{len(phase_benchmarks)}] {bench_def.get('name', 'unnamed')}")

                merged_config = self._merge_configs(phase_defaults, bench_def)
                bench_config = self._parse_benchmark_config(merged_config)
                bench_config.metadata['phase'] = phase_name
                result = self._execute_benchmark(bench_config)
                self.results.append(result)
                print()

    def _parse_benchmark_config(self, config: Dict) -> BenchmarkConfig:
        """Parse configuration dict into BenchmarkConfig"""
        return BenchmarkConfig(
            name=config.get('name', 'unnamed'),
            backend=config.get('backend', 'cuda'),
            size=config.get('size', 1),
            devices=str(config.get('devices', '0')),
            platform=str(config.get('platform', '0')),
            num_nodes=config.get('num_nodes', 1),
            hostfile=config.get('hostfile', ''),
            benchmark=config.get('benchmark', ''),
            bin_dir=config.get('bin_dir', './bin'),
            log_dir=config.get('log_dir', './Logs'),
            metadata=config.get('metadata', {})
        )

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Merge two configuration dicts"""
        merged = base.copy()
        merged.update(override)
        return merged

    def _execute_benchmark(self, config: BenchmarkConfig) -> ExperimentResult:
        """Execute a single benchmark using shocdriver.py"""
        import time

        # Prepare output files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(self.config.get('output_dir', './results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = config.name.replace(' ', '_').replace('/', '_')
        csv_file = output_dir / f"{safe_name}_{timestamp}.csv"
        json_file = output_dir / f"{safe_name}_{timestamp}.json"

        # Build shocdriver.py command
        cmd = [
            sys.executable,
            'tools/shocdriver.py',
            '--backend', config.backend,
            '--size', str(config.size),
            '--device', config.devices,
            '--platform', config.platform,
            '--bin-dir', config.bin_dir,
            '--log-dir', config.log_dir,
            '--output', str(csv_file),
            '--json-report', str(json_file)
        ]

        if config.num_nodes > 1:
            cmd.extend(['--num-nodes', str(config.num_nodes)])

        if config.hostfile:
            cmd.extend(['--hostfile', config.hostfile])

        if config.benchmark:
            cmd.extend(['--benchmark', config.benchmark])

        # Execute
        print(f"  Running: {' '.join(cmd[2:])}")
        start = time.time()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            duration = time.time() - start

            if result.returncode == 0:
                return ExperimentResult(
                    config=config,
                    success=True,
                    csv_file=csv_file,
                    json_file=json_file,
                    duration=duration
                )
            else:
                return ExperimentResult(
                    config=config,
                    success=False,
                    error=result.stderr,
                    duration=duration
                )

        except subprocess.TimeoutExpired:
            return ExperimentResult(
                config=config,
                success=False,
                error="Timeout (3600s)",
                duration=3600.0
            )
        except Exception as e:
            return ExperimentResult(
                config=config,
                success=False,
                error=str(e),
                duration=time.time() - start
            )

    def _generate_summary(self):
        """Generate summary report"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80 + "\n")

        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        failed = total - success
        total_time = sum(r.duration for r in self.results)

        print(f"Total runs: {total}")
        print(f"Successful: {success}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print()

        # Save summary
        summary_file = Path(self.config.get('output_dir', './results')) / 'summary.json'
        summary = {
            'config_file': str(self.config_file),
            'timestamp': datetime.now().isoformat(),
            'total_runs': total,
            'successful': success,
            'failed': failed,
            'total_duration': total_time,
            'results': []
        }

        for result in self.results:
            summary['results'].append({
                'name': result.config.name,
                'success': result.success,
                'backend': result.config.backend,
                'size': result.config.size,
                'devices': result.config.devices,
                'metadata': result.config.metadata,
                'csv_file': str(result.csv_file) if result.csv_file else None,
                'json_file': str(result.json_file) if result.json_file else None,
                'error': result.error,
                'duration': result.duration
            })

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {summary_file}")

        # List failed runs
        if failed > 0:
            print("\nFailed runs:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.config.name}: {result.error}")

def main():
    parser = argparse.ArgumentParser(
        description='SHOC Configuration Runner - Advanced benchmark orchestration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Types:
  single     - Run a single benchmark
  suite      - Run a suite of benchmarks
  sweep      - Parameter sweep (all combinations)
  matrix     - Test matrix (explicit combinations)
  experiment - Multi-phase experiment workflow

Examples:
  %(prog)s --config experiments/quick-test.yaml
  %(prog)s --config experiments/gpu-comparison.yaml
  %(prog)s --config experiments/size-sweep.yaml
        """
    )

    parser.add_argument('--config', type=Path, required=True,
                       help='Configuration file (YAML)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    runner = ConfigurationRunner(args.config)

    if args.dry_run:
        print("Dry run mode - would execute configuration:")
        print(yaml.dump(runner.config, default_flow_style=False))
        sys.exit(0)

    runner.run()

if __name__ == '__main__':
    main()
