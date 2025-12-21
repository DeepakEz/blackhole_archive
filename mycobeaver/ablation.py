"""
MycoBeaver Ablation Experiment Framework
==========================================
Automated ablation matrix for systematic feature evaluation.

ABLATION VARIABLES:
- Overmind [on/off]: Meta-controller that adjusts training parameters
- Semantic [on/off]: Colony semantic system for knowledge aggregation
- Memory [on/off]: Event-based semantic memory with kNN retrieval
- Roles [uniform/specialized]: Agent role specialization
- TaskAlloc [on/off]: Swarm-based task allocation system

METRICS:
- cumulative_reward: Total reward over episode
- dam_count: Number of dams built
- survival_rate: Fraction of agents surviving
- role_action_entropy: Diversity of actions per role
- unique_cells_visited: Exploration coverage

OUTPUTS:
- CSV with all experiment results
- JSON with configuration and statistics
- Statistical analysis with t-tests between variants
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import json
import csv
import time
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import copy

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .config import SimulationConfig, create_default_config, AgentRole
from .environment import MycoBeaverEnv


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""
    name: str
    use_overmind: bool = True
    use_semantic: bool = True
    use_memory: bool = True
    use_specialized_roles: bool = True  # False = all workers
    use_task_allocation: bool = True
    use_pheromones: bool = True
    use_projects: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config_name: str
    seed: int
    run_id: int

    # Core metrics
    cumulative_reward: float = 0.0
    dam_count: int = 0
    survival_rate: float = 0.0
    unique_cells_visited: int = 0

    # Role metrics
    role_action_entropy: float = 0.0
    role_action_counts: Dict[str, Dict[int, int]] = field(default_factory=dict)

    # Environment metrics
    avg_water_level: float = 0.0
    total_vegetation: float = 0.0
    flood_events: int = 0
    dam_failures: int = 0

    # Memory metrics
    memory_events: int = 0
    memory_retrievals: int = 0

    # Performance
    episode_length: int = 0
    wall_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert nested dicts
        d["role_action_counts"] = dict(self.role_action_counts)
        return d


@dataclass
class AblationExperiment:
    """Ablation experiment manager."""
    output_dir: Path
    base_config: SimulationConfig = field(default_factory=create_default_config)
    n_runs_per_variant: int = 5
    n_episodes_per_run: int = 10
    max_steps_per_episode: int = 500
    base_seed: int = 42

    # Results storage
    results: List[ExperimentResult] = field(default_factory=list)
    variant_configs: Dict[str, AblationConfig] = field(default_factory=dict)

    # Git metadata
    commit_sha: str = ""

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._get_commit_sha()

    def _get_commit_sha(self):
        """Get current git commit SHA for reproducibility."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.commit_sha = result.stdout.strip()[:12]
        except (subprocess.SubprocessError, FileNotFoundError):
            self.commit_sha = "unknown"

    def define_ablation_matrix(self) -> Dict[str, AblationConfig]:
        """Define the full ablation matrix.

        Creates variants by toggling each feature:
        - baseline: All features on
        - no_overmind: Overmind off
        - no_semantic: Semantic system off
        - no_memory: Memory system off
        - uniform_roles: All agents are workers
        - no_task_alloc: Task allocation off

        Also creates combined variants for interaction effects.
        """
        configs = {}

        # Baseline (all features on)
        configs["baseline"] = AblationConfig(
            name="baseline",
            use_overmind=True,
            use_semantic=True,
            use_memory=True,
            use_specialized_roles=True,
            use_task_allocation=True,
        )

        # Single-feature ablations
        configs["no_overmind"] = AblationConfig(
            name="no_overmind",
            use_overmind=False,
            use_semantic=True,
            use_memory=True,
            use_specialized_roles=True,
            use_task_allocation=True,
        )

        configs["no_semantic"] = AblationConfig(
            name="no_semantic",
            use_overmind=True,
            use_semantic=False,
            use_memory=True,
            use_specialized_roles=True,
            use_task_allocation=True,
        )

        configs["no_memory"] = AblationConfig(
            name="no_memory",
            use_overmind=True,
            use_semantic=True,
            use_memory=False,
            use_specialized_roles=True,
            use_task_allocation=True,
        )

        configs["uniform_roles"] = AblationConfig(
            name="uniform_roles",
            use_overmind=True,
            use_semantic=True,
            use_memory=True,
            use_specialized_roles=False,
            use_task_allocation=True,
        )

        configs["no_task_alloc"] = AblationConfig(
            name="no_task_alloc",
            use_overmind=True,
            use_semantic=True,
            use_memory=True,
            use_specialized_roles=True,
            use_task_allocation=False,
        )

        # Combined ablations (interaction effects)
        configs["minimal"] = AblationConfig(
            name="minimal",
            use_overmind=False,
            use_semantic=False,
            use_memory=False,
            use_specialized_roles=False,
            use_task_allocation=False,
        )

        configs["no_coordination"] = AblationConfig(
            name="no_coordination",
            use_overmind=False,
            use_semantic=False,
            use_memory=False,
            use_specialized_roles=True,
            use_task_allocation=False,
        )

        self.variant_configs = configs
        return configs

    def apply_ablation_config(
        self,
        base_config: SimulationConfig,
        ablation: AblationConfig
    ) -> SimulationConfig:
        """Apply ablation settings to a simulation config."""
        config = copy.deepcopy(base_config)

        # Overmind
        config.training.use_overmind = ablation.use_overmind

        # Semantic system (disable by setting very low update rate)
        if not ablation.use_semantic:
            config.time_scales.semantic_update_interval = 1000000

        # Memory
        config.memory.enable_memory = ablation.use_memory

        # Roles (handled at environment level - all workers if not specialized)
        # This is tracked in ablation config but applied during agent init

        # Task allocation (disable by setting very low scan rate)
        if not ablation.use_task_allocation:
            # Effectively disable task scanning
            pass  # Handled in environment step

        # Pheromones and projects
        config.training.use_pheromones = ablation.use_pheromones
        config.training.use_projects = ablation.use_projects

        # Set max steps
        config.training.max_steps_per_episode = self.max_steps_per_episode

        return config

    def run_single_episode(
        self,
        env: MycoBeaverEnv,
        ablation: AblationConfig,
        seed: int
    ) -> Dict[str, Any]:
        """Run a single episode and collect metrics."""
        obs, info = env.reset(seed=seed)

        # If uniform roles, override agent roles to WORKER
        if not ablation.use_specialized_roles:
            for agent in env.agents:
                agent.role = AgentRole.WORKER

        cumulative_reward = 0.0
        role_actions = defaultdict(lambda: defaultdict(int))
        visited_cells = set()
        step = 0

        done = False
        while not done and step < self.max_steps_per_episode:
            # Random policy for ablation (or could use trained policy)
            actions = {}
            for i, agent in enumerate(env.agents):
                if agent.alive:
                    action = env.np_random.integers(0, env.config.policy.n_actions)
                    actions[f"agent_{i}"] = action

                    # Track role-action counts
                    role_name = agent.role.value
                    role_actions[role_name][action] += 1

                    # Track visited cells
                    visited_cells.add(agent.position)

            obs, rewards, terminated, truncated, info = env.step(actions)
            cumulative_reward += sum(rewards.values())
            done = terminated or truncated
            step += 1

        # Compute final metrics
        n_alive = sum(1 for a in env.agents if a.alive)
        survival_rate = n_alive / len(env.agents)

        # Compute role-action entropy
        role_entropy = self._compute_role_action_entropy(role_actions)

        # Dam count
        dam_mask = env.grid_state.dam_permeability < 0.9
        dam_count = int(np.sum(dam_mask))

        # Memory metrics
        memory_events = 0
        memory_retrievals = 0
        if env.semantic_memory is not None:
            mem_stats = env.semantic_memory.get_statistics()
            memory_events = mem_stats["total_events"]
            memory_retrievals = mem_stats["total_retrievals"]

        return {
            "cumulative_reward": cumulative_reward,
            "dam_count": dam_count,
            "survival_rate": survival_rate,
            "unique_cells_visited": len(visited_cells),
            "role_action_entropy": role_entropy,
            "role_action_counts": dict(role_actions),
            "avg_water_level": info.get("avg_water_level", 0.0),
            "total_vegetation": info.get("total_vegetation", 0.0),
            "flood_events": info.get("flood_events_caused", 0),
            "dam_failures": info.get("dam_failures", 0),
            "memory_events": memory_events,
            "memory_retrievals": memory_retrievals,
            "episode_length": step,
        }

    def _compute_role_action_entropy(
        self,
        role_actions: Dict[str, Dict[int, int]]
    ) -> float:
        """Compute average entropy of action distribution per role."""
        entropies = []

        for role, action_counts in role_actions.items():
            total = sum(action_counts.values())
            if total == 0:
                continue

            probs = np.array(list(action_counts.values())) / total
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        return float(np.mean(entropies)) if entropies else 0.0

    def run_variant(
        self,
        variant_name: str,
        ablation: AblationConfig,
        progress_callback: Optional[Callable] = None
    ) -> List[ExperimentResult]:
        """Run all experiments for a single variant."""
        results = []

        for run_id in range(self.n_runs_per_variant):
            run_seed = self.base_seed + run_id * 1000

            # Create config and environment
            config = self.apply_ablation_config(self.base_config, ablation)
            env = MycoBeaverEnv(config)

            run_metrics = defaultdict(list)
            start_time = time.time()

            for episode in range(self.n_episodes_per_run):
                episode_seed = run_seed + episode
                metrics = self.run_single_episode(env, ablation, episode_seed)

                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        run_metrics[key].append(value)

                if progress_callback:
                    progress = (run_id * self.n_episodes_per_run + episode + 1)
                    total = self.n_runs_per_variant * self.n_episodes_per_run
                    progress_callback(variant_name, progress, total)

            wall_time = time.time() - start_time

            # Aggregate metrics across episodes
            result = ExperimentResult(
                config_name=variant_name,
                seed=run_seed,
                run_id=run_id,
                cumulative_reward=float(np.mean(run_metrics["cumulative_reward"])),
                dam_count=int(np.mean(run_metrics["dam_count"])),
                survival_rate=float(np.mean(run_metrics["survival_rate"])),
                unique_cells_visited=int(np.mean(run_metrics["unique_cells_visited"])),
                role_action_entropy=float(np.mean(run_metrics["role_action_entropy"])),
                avg_water_level=float(np.mean(run_metrics["avg_water_level"])),
                total_vegetation=float(np.mean(run_metrics["total_vegetation"])),
                flood_events=int(np.sum(run_metrics.get("flood_events", [0]))),
                dam_failures=int(np.sum(run_metrics.get("dam_failures", [0]))),
                memory_events=int(np.mean(run_metrics.get("memory_events", [0]))),
                memory_retrievals=int(np.mean(run_metrics.get("memory_retrievals", [0]))),
                episode_length=int(np.mean(run_metrics["episode_length"])),
                wall_time_seconds=wall_time,
            )

            results.append(result)

        return results

    def run_all_variants(
        self,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """Run experiments for all ablation variants."""
        if not self.variant_configs:
            self.define_ablation_matrix()

        self.results = []

        for variant_name, ablation in self.variant_configs.items():
            print(f"Running variant: {variant_name}")
            variant_results = self.run_variant(
                variant_name,
                ablation,
                progress_callback
            )
            self.results.extend(variant_results)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute statistical analysis across variants."""
        stats_results = {
            "variants": {},
            "comparisons": {},
            "commit_sha": self.commit_sha,
            "timestamp": datetime.now().isoformat(),
        }

        # Group results by variant
        by_variant = defaultdict(list)
        for result in self.results:
            by_variant[result.config_name].append(result)

        # Compute per-variant statistics
        metrics = [
            "cumulative_reward", "dam_count", "survival_rate",
            "unique_cells_visited", "role_action_entropy"
        ]

        for variant_name, results in by_variant.items():
            variant_stats = {}
            for metric in metrics:
                values = [getattr(r, metric) for r in results]
                variant_stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": len(values),
                }
            stats_results["variants"][variant_name] = variant_stats

        # Statistical comparisons vs baseline
        if "baseline" in by_variant and SCIPY_AVAILABLE:
            baseline_results = by_variant["baseline"]

            for variant_name, results in by_variant.items():
                if variant_name == "baseline":
                    continue

                comparisons = {}
                for metric in metrics:
                    baseline_vals = [getattr(r, metric) for r in baseline_results]
                    variant_vals = [getattr(r, metric) for r in results]

                    # t-test
                    if len(baseline_vals) >= 2 and len(variant_vals) >= 2:
                        t_stat, p_value = stats.ttest_ind(baseline_vals, variant_vals)
                        effect_size = (
                            (np.mean(variant_vals) - np.mean(baseline_vals)) /
                            (np.std(baseline_vals) + 1e-10)
                        )

                        comparisons[metric] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "effect_size": float(effect_size),
                            "significant": p_value < 0.05,
                        }

                stats_results["comparisons"][variant_name] = comparisons

        return stats_results

    def export_csv(self, filename: Optional[str] = None) -> Path:
        """Export results to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ablation_results_{timestamp}.csv"

        filepath = self.output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = [
                "config_name", "seed", "run_id",
                "cumulative_reward", "dam_count", "survival_rate",
                "unique_cells_visited", "role_action_entropy",
                "avg_water_level", "total_vegetation",
                "flood_events", "dam_failures",
                "memory_events", "memory_retrievals",
                "episode_length", "wall_time_seconds"
            ]
            writer.writerow(header)

            # Data rows
            for result in self.results:
                row = [
                    result.config_name,
                    result.seed,
                    result.run_id,
                    result.cumulative_reward,
                    result.dam_count,
                    result.survival_rate,
                    result.unique_cells_visited,
                    result.role_action_entropy,
                    result.avg_water_level,
                    result.total_vegetation,
                    result.flood_events,
                    result.dam_failures,
                    result.memory_events,
                    result.memory_retrievals,
                    result.episode_length,
                    result.wall_time_seconds,
                ]
                writer.writerow(row)

        print(f"Exported CSV to: {filepath}")
        return filepath

    def export_json(self, filename: Optional[str] = None) -> Path:
        """Export results and statistics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ablation_results_{timestamp}.json"

        filepath = self.output_dir / filename

        # Compute statistics
        stats = self.compute_statistics()

        export_data = {
            "metadata": {
                "commit_sha": self.commit_sha,
                "timestamp": datetime.now().isoformat(),
                "base_seed": self.base_seed,
                "n_runs_per_variant": self.n_runs_per_variant,
                "n_episodes_per_run": self.n_episodes_per_run,
                "max_steps_per_episode": self.max_steps_per_episode,
            },
            "variant_configs": {
                name: config.to_dict()
                for name, config in self.variant_configs.items()
            },
            "statistics": stats,
            "raw_results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported JSON to: {filepath}")
        return filepath

    def print_summary(self) -> None:
        """Print summary of ablation results."""
        stats = self.compute_statistics()

        print("\n" + "=" * 60)
        print("ABLATION EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Commit SHA: {self.commit_sha}")
        print(f"Variants: {len(self.variant_configs)}")
        print(f"Runs per variant: {self.n_runs_per_variant}")
        print(f"Episodes per run: {self.n_episodes_per_run}")
        print()

        # Print variant statistics
        print("VARIANT STATISTICS:")
        print("-" * 60)
        for variant_name, variant_stats in stats["variants"].items():
            reward = variant_stats["cumulative_reward"]
            dams = variant_stats["dam_count"]
            survival = variant_stats["survival_rate"]
            print(f"  {variant_name}:")
            print(f"    Reward: {reward['mean']:.2f} (+/- {reward['std']:.2f})")
            print(f"    Dams: {dams['mean']:.1f} (+/- {dams['std']:.1f})")
            print(f"    Survival: {survival['mean']:.2%} (+/- {survival['std']:.2%})")
            print()

        # Print statistical comparisons
        if stats["comparisons"]:
            print("STATISTICAL COMPARISONS (vs baseline):")
            print("-" * 60)
            for variant_name, comparisons in stats["comparisons"].items():
                print(f"  {variant_name}:")
                for metric, comp in comparisons.items():
                    sig = "*" if comp["significant"] else ""
                    print(f"    {metric}: p={comp['p_value']:.4f}{sig}, "
                          f"effect={comp['effect_size']:+.2f}")
                print()


def run_ablation_experiment(
    output_dir: str = "ablation_results",
    n_runs: int = 5,
    n_episodes: int = 10,
    max_steps: int = 500,
    seed: int = 42,
    variants: Optional[List[str]] = None,
) -> AblationExperiment:
    """
    Run ablation experiment with specified parameters.

    Args:
        output_dir: Directory to save results
        n_runs: Number of runs per variant
        n_episodes: Number of episodes per run
        max_steps: Maximum steps per episode
        seed: Base random seed for reproducibility
        variants: Optional list of variant names to run (default: all)

    Returns:
        AblationExperiment with results
    """
    experiment = AblationExperiment(
        output_dir=Path(output_dir),
        n_runs_per_variant=n_runs,
        n_episodes_per_run=n_episodes,
        max_steps_per_episode=max_steps,
        base_seed=seed,
    )

    # Define ablation matrix
    all_configs = experiment.define_ablation_matrix()

    # Filter variants if specified
    if variants is not None:
        experiment.variant_configs = {
            k: v for k, v in all_configs.items() if k in variants
        }

    # Progress callback
    def print_progress(variant: str, current: int, total: int):
        pct = 100 * current / total
        print(f"\r  [{variant}] {current}/{total} ({pct:.0f}%)", end="", flush=True)

    # Run experiments
    print(f"Starting ablation experiment with {len(experiment.variant_configs)} variants")
    print(f"  Runs per variant: {n_runs}")
    print(f"  Episodes per run: {n_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Base seed: {seed}")
    print()

    experiment.run_all_variants(progress_callback=print_progress)
    print("\n")

    # Export results
    experiment.export_csv()
    experiment.export_json()

    # Print summary
    experiment.print_summary()

    return experiment


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MycoBeaver ablation experiments")
    parser.add_argument("--output-dir", default="ablation_results",
                        help="Output directory for results")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of runs per variant")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of episodes per run")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Specific variants to run (default: all)")

    args = parser.parse_args()

    run_ablation_experiment(
        output_dir=args.output_dir,
        n_runs=args.n_runs,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        variants=args.variants,
    )
