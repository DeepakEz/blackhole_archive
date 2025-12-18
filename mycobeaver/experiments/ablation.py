"""
Ablation Matrix Experiments
===========================

PHASE 5: Experiment 1 - Systematic ablation studies.

Run with:
- No pheromones
- No Physarum
- No Overmind
- No semantic memory
- Full system

Metrics:
- Time to stable infrastructure
- Water efficiency
- Survival rate
- Info cost

This quantifies the contribution of each component.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import copy
import time
from enum import Enum

from ..config import SimulationConfig, TrainingConfig
from ..environment import MycoBeaverEnv
from .metrics import MetricsCollector, aggregate_metrics, AggregatedMetrics


class AblationCondition(Enum):
    """Ablation conditions for systematic comparison"""
    FULL_SYSTEM = "full"
    NO_PHEROMONES = "no_pheromones"
    NO_PHYSARUM = "no_physarum"
    NO_OVERMIND = "no_overmind"
    NO_SEMANTIC = "no_semantic"
    NO_PROJECTS = "no_projects"
    MINIMAL = "minimal"  # Only basic environment, no coordination


@dataclass
class AblationConfig:
    """Configuration for ablation experiments"""
    # Conditions to test
    conditions: List[AblationCondition] = field(default_factory=lambda: [
        AblationCondition.FULL_SYSTEM,
        AblationCondition.NO_PHEROMONES,
        AblationCondition.NO_PHYSARUM,
        AblationCondition.NO_OVERMIND,
        AblationCondition.NO_SEMANTIC,
        AblationCondition.NO_PROJECTS,
        AblationCondition.MINIMAL,
    ])

    # Experiment parameters
    n_episodes: int = 10
    steps_per_episode: int = 500
    n_seeds: int = 5

    # Base configuration
    n_beavers: int = 20
    grid_size: int = 32

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/ablation"))
    save_detailed: bool = True


class AblationExperiment:
    """
    Systematic ablation study of MycoBeaver components.

    Tests the contribution of each component by running experiments
    with each component disabled and comparing to the full system.
    """

    def __init__(self, config: Optional[AblationConfig] = None):
        self.config = config or AblationConfig()
        self.results: Dict[str, List[MetricsCollector]] = {}

    def create_config_for_condition(self, condition: AblationCondition,
                                     base_config: Optional[SimulationConfig] = None
                                     ) -> SimulationConfig:
        """Create a simulation config with specific components disabled"""
        if base_config is None:
            base_config = SimulationConfig()

        # Deep copy to avoid modifying original
        config = copy.deepcopy(base_config)

        # Set base parameters
        config.n_beavers = self.config.n_beavers
        config.grid.grid_size = self.config.grid_size
        config.training.max_steps_per_episode = self.config.steps_per_episode

        # Apply ablation
        if condition == AblationCondition.FULL_SYSTEM:
            # Enable everything
            config.training.use_pheromones = True
            config.training.use_physarum = True
            config.training.use_overmind = True
            config.training.use_projects = True
            # Semantic is always on in full system

        elif condition == AblationCondition.NO_PHEROMONES:
            config.training.use_pheromones = False

        elif condition == AblationCondition.NO_PHYSARUM:
            config.training.use_physarum = False

        elif condition == AblationCondition.NO_OVERMIND:
            config.training.use_overmind = False

        elif condition == AblationCondition.NO_SEMANTIC:
            # Disable semantic by setting very high costs
            config.info_costs.cost_semantic_vertex = 1000.0
            config.info_costs.cost_semantic_edge = 1000.0
            config.info_costs.cost_semantic_query = 1000.0

        elif condition == AblationCondition.NO_PROJECTS:
            config.training.use_projects = False

        elif condition == AblationCondition.MINIMAL:
            # Disable all coordination mechanisms
            config.training.use_pheromones = False
            config.training.use_physarum = False
            config.training.use_overmind = False
            config.training.use_projects = False
            config.info_costs.cost_semantic_vertex = 1000.0
            config.info_costs.cost_semantic_edge = 1000.0

        return config

    def run_single_experiment(self, condition: AblationCondition,
                               seed: int,
                               policy_fn: Optional[Callable] = None,
                               verbose: bool = False) -> MetricsCollector:
        """
        Run a single ablation experiment.

        Args:
            condition: Which ablation condition to test
            seed: Random seed
            policy_fn: Optional policy function for actions (default: random)
            verbose: Print progress

        Returns:
            MetricsCollector with all metrics
        """
        # Create config for this condition
        config = self.create_config_for_condition(condition)

        # Create environment
        env = MycoBeaverEnv(config)
        obs, info = env.reset(seed=seed)

        # Create metrics collector
        experiment_id = f"ablation_{condition.value}_seed{seed}"
        collector = MetricsCollector(
            experiment_id=experiment_id,
            config_name=condition.value,
            seed=seed,
        )

        # Run episode
        for step in range(self.config.steps_per_episode):
            # Get actions
            if policy_fn is not None:
                actions = policy_fn(obs, env)
            else:
                # Random policy
                actions = {f"agent_{i}": np.random.randint(0, config.policy.n_actions)
                          for i in range(config.n_beavers)}

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Collect metrics
            collector.collect_step(env, info)

            if terminated or truncated:
                break

            if verbose and step % 100 == 0:
                print(f"  Step {step}: {collector.metrics.step_metrics[-1].n_structures} structures, "
                      f"{collector.metrics.step_metrics[-1].survival_rate:.2%} survival")

        # Finalize
        collector.finalize()
        env.close()

        return collector

    def run_condition(self, condition: AblationCondition,
                       policy_fn: Optional[Callable] = None,
                       verbose: bool = True) -> List[MetricsCollector]:
        """Run all seeds for a single ablation condition"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running ablation: {condition.value}")
            print(f"{'='*60}")

        collectors = []
        for seed in range(self.config.n_seeds):
            if verbose:
                print(f"\n  Seed {seed + 1}/{self.config.n_seeds}")

            collector = self.run_single_experiment(
                condition, seed, policy_fn, verbose=verbose
            )
            collectors.append(collector)

            if verbose:
                metrics = collector.to_dict()
                print(f"    Final survival: {metrics['final_survival_rate']:.2%}")
                print(f"    Peak infrastructure: {metrics['peak_infrastructure']}")
                print(f"    Time to stable: {metrics['time_to_stable']}")

        return collectors

    def run_all(self, policy_fn: Optional[Callable] = None,
                verbose: bool = True) -> Dict[str, AggregatedMetrics]:
        """
        Run the complete ablation matrix.

        Returns:
            Dict mapping condition name to aggregated metrics
        """
        if verbose:
            print("\n" + "=" * 60)
            print("ABLATION MATRIX EXPERIMENT")
            print("=" * 60)
            print(f"Conditions: {[c.value for c in self.config.conditions]}")
            print(f"Seeds per condition: {self.config.n_seeds}")
            print(f"Steps per episode: {self.config.steps_per_episode}")

        results = {}

        for condition in self.config.conditions:
            collectors = self.run_condition(condition, policy_fn, verbose)
            self.results[condition.value] = collectors

            # Aggregate
            aggregated = aggregate_metrics(collectors, condition.value)
            results[condition.value] = aggregated

            # Save if configured
            if self.config.save_detailed:
                self._save_condition_results(condition, collectors)

        # Print summary
        if verbose:
            self._print_summary(results)

        return results

    def _save_condition_results(self, condition: AblationCondition,
                                 collectors: List[MetricsCollector]):
        """Save results for a condition"""
        output_dir = self.config.output_dir / condition.value
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, collector in enumerate(collectors):
            collector.save(output_dir / f"seed_{i}")

    def _print_summary(self, results: Dict[str, AggregatedMetrics]):
        """Print summary comparison table"""
        print("\n" + "=" * 80)
        print("ABLATION RESULTS SUMMARY")
        print("=" * 80)

        # Header
        print(f"{'Condition':<20} {'Survival':<12} {'Stability':<12} "
              f"{'Efficiency':<12} {'Info Cost':<12}")
        print("-" * 80)

        # Full system as baseline
        if "full" in results:
            baseline = results["full"]

        for name, agg in results.items():
            survival = f"{agg.survival_rate_mean:.2%} ± {agg.survival_rate_std:.2%}"
            stability = f"{agg.time_to_stable_mean:.0f} ± {agg.time_to_stable_std:.0f}" \
                       if agg.time_to_stable_mean > 0 else "N/A"
            efficiency = f"{agg.infrastructure_efficiency_mean:.3f} ± {agg.infrastructure_efficiency_std:.3f}"
            info_cost = f"{agg.info_cost_mean:.1f} ± {agg.info_cost_std:.1f}"

            print(f"{name:<20} {survival:<12} {stability:<12} {efficiency:<12} {info_cost:<12}")

        print("=" * 80)


def run_ablation_matrix(n_seeds: int = 5,
                        steps_per_episode: int = 500,
                        n_beavers: int = 20,
                        output_dir: str = "results/ablation",
                        policy_fn: Optional[Callable] = None,
                        verbose: bool = True) -> Dict[str, AggregatedMetrics]:
    """
    Convenience function to run the full ablation matrix.

    Args:
        n_seeds: Number of random seeds per condition
        steps_per_episode: Steps per experiment run
        n_beavers: Number of agents
        output_dir: Where to save results
        policy_fn: Optional trained policy (default: random)
        verbose: Print progress

    Returns:
        Dict of aggregated metrics per condition

    Example:
        >>> results = run_ablation_matrix(n_seeds=3, verbose=True)
        >>> print(results["full"].survival_rate_mean)
    """
    config = AblationConfig(
        n_seeds=n_seeds,
        steps_per_episode=steps_per_episode,
        n_beavers=n_beavers,
        output_dir=Path(output_dir),
    )

    experiment = AblationExperiment(config)
    return experiment.run_all(policy_fn=policy_fn, verbose=verbose)
