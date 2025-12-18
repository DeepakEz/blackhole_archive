"""
Experiment Runner
=================

PHASE 5: Unified experiment execution framework.

Provides:
- Parallel experiment execution
- Progress tracking
- Result aggregation
- Automatic figure generation

Usage:
    >>> runner = ExperimentRunner()
    >>> results = runner.run_full_suite()
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .metrics import MetricsCollector, AggregatedMetrics
from .ablation import AblationExperiment, AblationConfig, run_ablation_matrix
from .stress import StressExperiment, StressConfig, run_stress_tests, RecoveryMetrics
from .scaling import ScalingExperiment, ScalingConfig, ScalingResults, run_scaling_experiment
from .analysis import (
    ExperimentAnalyzer,
    plot_ablation_results,
    plot_stress_results,
    plot_scaling_results,
    generate_paper_figures,
)


@dataclass
class ExperimentConfig:
    """Master configuration for all experiments"""
    # What to run
    run_ablation: bool = True
    run_stress: bool = True
    run_scaling: bool = True

    # Common parameters
    n_seeds: int = 5
    steps_per_episode: int = 500
    n_beavers: int = 20
    grid_size: int = 32

    # Ablation specific
    ablation_config: Optional[AblationConfig] = None

    # Stress specific
    stress_config: Optional[StressConfig] = None

    # Scaling specific
    scaling_config: Optional[ScalingConfig] = None
    agent_counts: List[int] = field(default_factory=lambda: [10, 25, 50, 100])

    # Execution
    n_workers: int = 1  # Parallel workers (1 = sequential)
    output_dir: Path = field(default_factory=lambda: Path("results"))

    # Output options
    save_detailed: bool = True
    generate_figures: bool = True
    save_summary: bool = True


@dataclass
class ExperimentSuiteResults:
    """Results from running the full experiment suite"""
    timestamp: str
    config: ExperimentConfig
    duration_seconds: float

    # Results by experiment type
    ablation_results: Optional[Dict[str, AggregatedMetrics]] = None
    stress_results: Optional[Dict[str, List[RecoveryMetrics]]] = None
    scaling_results: Optional[ScalingResults] = None

    # Generated artifacts
    figure_paths: List[Path] = field(default_factory=list)
    summary_path: Optional[Path] = None


class ExperimentRunner:
    """
    Unified runner for the MycoBeaver experiment suite.

    Coordinates ablation, stress, and scaling experiments,
    with parallel execution and result aggregation.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.results: Optional[ExperimentSuiteResults] = None

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_suite(self, policy_fn: Optional[Callable] = None,
                       verbose: bool = True) -> ExperimentSuiteResults:
        """
        Run the complete experiment suite.

        Args:
            policy_fn: Optional trained policy for action selection
            verbose: Print progress

        Returns:
            ExperimentSuiteResults with all results
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if verbose:
            print("\n" + "=" * 70)
            print("MYCOBEAVER EXPERIMENT SUITE")
            print("=" * 70)
            print(f"Timestamp: {timestamp}")
            print(f"Output directory: {self.config.output_dir}")
            print(f"Experiments: "
                  f"{'Ablation ' if self.config.run_ablation else ''}"
                  f"{'Stress ' if self.config.run_stress else ''}"
                  f"{'Scaling ' if self.config.run_scaling else ''}")
            print("=" * 70)

        # Initialize results
        ablation_results = None
        stress_results = None
        scaling_results = None

        # Run ablation experiments
        if self.config.run_ablation:
            if verbose:
                print("\n[1/3] Running ABLATION experiments...")

            ablation_config = self.config.ablation_config or AblationConfig(
                n_seeds=self.config.n_seeds,
                steps_per_episode=self.config.steps_per_episode,
                n_beavers=self.config.n_beavers,
                grid_size=self.config.grid_size,
                output_dir=self.config.output_dir / "ablation",
            )

            experiment = AblationExperiment(ablation_config)
            ablation_results = experiment.run_all(policy_fn=policy_fn, verbose=verbose)

        # Run stress tests
        if self.config.run_stress:
            if verbose:
                print("\n[2/3] Running STRESS experiments...")

            stress_config = self.config.stress_config or StressConfig(
                n_seeds=self.config.n_seeds,
                steps_per_episode=self.config.steps_per_episode,
                n_beavers=self.config.n_beavers,
                grid_size=self.config.grid_size,
                output_dir=self.config.output_dir / "stress",
            )

            experiment = StressExperiment(stress_config)
            stress_results = experiment.run_all(policy_fn=policy_fn, verbose=verbose)

        # Run scaling experiments
        if self.config.run_scaling:
            if verbose:
                print("\n[3/3] Running SCALING experiments...")

            scaling_config = self.config.scaling_config or ScalingConfig(
                agent_counts=self.config.agent_counts,
                n_seeds=self.config.n_seeds,
                steps_per_episode=self.config.steps_per_episode,
                output_dir=self.config.output_dir / "scaling",
            )

            experiment = ScalingExperiment(scaling_config)
            scaling_results = experiment.run_all(policy_fn=policy_fn, verbose=verbose)

        # Create results object
        duration = time.time() - start_time
        self.results = ExperimentSuiteResults(
            timestamp=timestamp,
            config=self.config,
            duration_seconds=duration,
            ablation_results=ablation_results,
            stress_results=stress_results,
            scaling_results=scaling_results,
        )

        # Generate figures
        if self.config.generate_figures:
            if verbose:
                print("\nGenerating figures...")
            self._generate_figures(verbose)

        # Save summary
        if self.config.save_summary:
            if verbose:
                print("\nSaving summary...")
            self._save_summary()

        if verbose:
            print("\n" + "=" * 70)
            print(f"EXPERIMENT SUITE COMPLETE")
            print(f"Total duration: {duration / 60:.1f} minutes")
            print(f"Results saved to: {self.config.output_dir}")
            print("=" * 70)

        return self.results

    def _generate_figures(self, verbose: bool = True):
        """Generate all figures from results"""
        figures_dir = self.config.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        analyzer = ExperimentAnalyzer()

        # Ablation figures
        if self.results.ablation_results:
            path = figures_dir / "ablation_matrix.pdf"
            fig = analyzer.plot_ablation_matrix(self.results.ablation_results, path)
            self.results.figure_paths.append(path)
            if verbose:
                print(f"  Saved: {path}")

        # Stress figures
        if self.results.stress_results:
            path = figures_dir / "stress_recovery.pdf"
            fig = analyzer.plot_stress_recovery(self.results.stress_results, path)
            self.results.figure_paths.append(path)
            if verbose:
                print(f"  Saved: {path}")

        # Scaling figures
        if self.results.scaling_results:
            path = figures_dir / "scaling_laws.pdf"
            fig = analyzer.plot_scaling_laws(self.results.scaling_results, path)
            self.results.figure_paths.append(path)
            if verbose:
                print(f"  Saved: {path}")

    def _save_summary(self):
        """Save experiment summary as JSON"""
        summary_path = self.config.output_dir / f"summary_{self.results.timestamp}.json"

        summary = {
            "timestamp": self.results.timestamp,
            "duration_seconds": self.results.duration_seconds,
            "config": {
                "n_seeds": self.config.n_seeds,
                "steps_per_episode": self.config.steps_per_episode,
                "n_beavers": self.config.n_beavers,
            },
        }

        # Ablation summary
        if self.results.ablation_results:
            summary["ablation"] = {
                name: {
                    "survival_rate": f"{agg.survival_rate_mean:.3f} ± {agg.survival_rate_std:.3f}",
                    "time_to_stable": f"{agg.time_to_stable_mean:.1f}",
                    "efficiency": f"{agg.infrastructure_efficiency_mean:.4f}",
                }
                for name, agg in self.results.ablation_results.items()
            }

        # Stress summary
        if self.results.stress_results:
            summary["stress"] = {
                name: {
                    "n_runs": len(metrics),
                    "avg_recovery_time": float(np.mean([m.recovery_time for m in metrics])),
                    "avg_reuse_rate": float(np.mean([m.reuse_rate for m in metrics])),
                }
                for name, metrics in self.results.stress_results.items()
            }

        # Scaling summary
        if self.results.scaling_results:
            summary["scaling"] = {
                "infrastructure_exponent": self.results.scaling_results.infrastructure_exponent,
                "info_cost_exponent": self.results.scaling_results.info_cost_exponent,
                "efficiency_exponent": self.results.scaling_results.efficiency_exponent,
                "agent_counts": [p.n_agents for p in self.results.scaling_results.scaling_points],
            }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.results.summary_path = summary_path

    def run_quick_validation(self, verbose: bool = True) -> bool:
        """
        Run a quick validation to check everything works.

        Uses minimal parameters for fast execution.
        """
        if verbose:
            print("\nRunning quick validation...")

        try:
            # Minimal ablation
            config = AblationConfig(
                n_seeds=1,
                steps_per_episode=50,
                n_beavers=5,
                grid_size=16,
            )
            config.conditions = config.conditions[:2]  # Just 2 conditions
            exp = AblationExperiment(config)
            exp.run_all(verbose=False)

            if verbose:
                print("  ✓ Ablation OK")

            # Minimal stress
            stress_config = StressConfig(
                n_seeds=1,
                steps_per_episode=50,
                n_beavers=5,
            )
            stress_config.scenarios = stress_config.scenarios[:1]
            exp = StressExperiment(stress_config)
            exp.run_all(verbose=False)

            if verbose:
                print("  ✓ Stress OK")

            # Minimal scaling
            scaling_config = ScalingConfig(
                agent_counts=[5, 10],
                n_seeds=1,
                steps_per_episode=50,
            )
            exp = ScalingExperiment(scaling_config)
            exp.run_all(verbose=False)

            if verbose:
                print("  ✓ Scaling OK")
                print("\nValidation PASSED!")

            return True

        except Exception as e:
            if verbose:
                print(f"\nValidation FAILED: {e}")
            return False


def run_full_experiment_suite(n_seeds: int = 5,
                              steps_per_episode: int = 500,
                              n_beavers: int = 20,
                              output_dir: str = "results",
                              policy_fn: Optional[Callable] = None,
                              verbose: bool = True) -> ExperimentSuiteResults:
    """
    Convenience function to run the complete experiment suite.

    This produces publishable results for the MycoBeaver system.

    Args:
        n_seeds: Random seeds per condition
        steps_per_episode: Steps per experiment
        n_beavers: Number of agents
        output_dir: Where to save results
        policy_fn: Optional trained policy
        verbose: Print progress

    Returns:
        Complete experiment results

    Example:
        >>> results = run_full_experiment_suite(n_seeds=5)
        >>> print(results.scaling_results.infrastructure_exponent)
        >>> # Infrastructure scales as n^{exponent}
    """
    config = ExperimentConfig(
        n_seeds=n_seeds,
        steps_per_episode=steps_per_episode,
        n_beavers=n_beavers,
        output_dir=Path(output_dir),
    )

    runner = ExperimentRunner(config)
    return runner.run_full_suite(policy_fn=policy_fn, verbose=verbose)


# Main entry point for command-line execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MycoBeaver experiments")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per condition")
    parser.add_argument("--steps", type=int, default=500, help="Steps per episode")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Run quick validation only")

    args = parser.parse_args()

    if args.validate:
        runner = ExperimentRunner()
        runner.run_quick_validation(verbose=True)
    else:
        results = run_full_experiment_suite(
            n_seeds=args.seeds,
            steps_per_episode=args.steps,
            n_beavers=args.agents,
            output_dir=args.output,
            verbose=True,
        )
