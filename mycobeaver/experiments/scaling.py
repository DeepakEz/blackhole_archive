"""
Scaling Law Experiments
=======================

PHASE 5: Experiment 3 - How does the system scale?

Scale:
- 10 → 50 → 100 → 200 agents
- Fixed grid vs scaling grid
- Fixed resources vs scaling resources

Measure:
- Coordination efficiency (infrastructure per agent)
- Info cost per agent
- Infrastructure reuse
- Communication overhead
- Emergent specialization

This is publishable material - reveals fundamental scaling properties
of distributed cognition systems.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import copy

from ..config import SimulationConfig
from ..environment import MycoBeaverEnv
from .metrics import MetricsCollector, aggregate_metrics


@dataclass
class ScalingPoint:
    """A single point in the scaling experiment"""
    n_agents: int
    grid_size: int

    # Results (filled after experiment)
    survival_rate: float = 0.0
    infrastructure_per_agent: float = 0.0
    info_cost_per_agent: float = 0.0
    coordination_efficiency: float = 0.0
    communication_overhead: float = 0.0
    time_to_stable: float = 0.0

    # Standard deviations
    survival_std: float = 0.0
    infrastructure_std: float = 0.0
    info_cost_std: float = 0.0


@dataclass
class ScalingConfig:
    """Configuration for scaling experiments"""
    # Agent counts to test
    agent_counts: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 200])

    # Grid scaling mode
    fixed_grid_size: int = 64  # Used when scale_grid=False
    scale_grid: bool = False  # If True, grid scales with agents
    agents_per_grid_cell: float = 0.05  # Density when scaling

    # Experiment parameters
    steps_per_episode: int = 500
    n_seeds: int = 5

    # Resource scaling
    scale_resources: bool = True  # Scale vegetation with grid

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/scaling"))


@dataclass
class ScalingResults:
    """Results from scaling experiment"""
    config: ScalingConfig
    scaling_points: List[ScalingPoint]

    # Fitted scaling laws
    # Infrastructure scales as: I ~ n^alpha
    infrastructure_exponent: float = 0.0
    infrastructure_r2: float = 0.0

    # Info cost scales as: C ~ n^beta
    info_cost_exponent: float = 0.0
    info_cost_r2: float = 0.0

    # Efficiency scales as: E ~ n^gamma
    efficiency_exponent: float = 0.0
    efficiency_r2: float = 0.0


class ScalingExperiment:
    """
    Scaling law experiment for MycoBeaver.

    Tests how coordination and efficiency change as colony size increases.
    This reveals fundamental properties of the distributed cognition system.
    """

    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.results: List[ScalingPoint] = []

    def get_grid_size(self, n_agents: int) -> int:
        """Compute grid size based on scaling mode"""
        if self.config.scale_grid:
            # Scale grid to maintain density
            cells_needed = n_agents / self.config.agents_per_grid_cell
            return int(np.ceil(np.sqrt(cells_needed)))
        else:
            return self.config.fixed_grid_size

    def run_scaling_point(self, n_agents: int,
                           policy_fn: Optional[Callable] = None,
                           verbose: bool = False) -> ScalingPoint:
        """
        Run experiments for a single agent count.

        Returns aggregated ScalingPoint with means and stds.
        """
        grid_size = self.get_grid_size(n_agents)

        # Collect metrics across seeds
        survival_rates = []
        infrastructures = []
        info_costs = []
        stability_times = []

        for seed in range(self.config.n_seeds):
            if verbose:
                print(f"    Seed {seed + 1}/{self.config.n_seeds}")

            metrics = self._run_single(n_agents, grid_size, seed, policy_fn, verbose)

            survival_rates.append(metrics["survival_rate"])
            infrastructures.append(metrics["infrastructure"])
            info_costs.append(metrics["info_cost"])
            stability_times.append(metrics["time_to_stable"])

        # Compute per-agent metrics
        infra_per_agent = np.array(infrastructures) / n_agents
        info_per_agent = np.array(info_costs) / n_agents
        coordination_eff = np.array(infrastructures) / (np.array(info_costs) + 1)

        return ScalingPoint(
            n_agents=n_agents,
            grid_size=grid_size,
            survival_rate=float(np.mean(survival_rates)),
            survival_std=float(np.std(survival_rates)),
            infrastructure_per_agent=float(np.mean(infra_per_agent)),
            infrastructure_std=float(np.std(infra_per_agent)),
            info_cost_per_agent=float(np.mean(info_per_agent)),
            info_cost_std=float(np.std(info_per_agent)),
            coordination_efficiency=float(np.mean(coordination_eff)),
            time_to_stable=float(np.mean([t for t in stability_times if t > 0])) \
                          if any(t > 0 for t in stability_times) else -1,
        )

    def _run_single(self, n_agents: int, grid_size: int, seed: int,
                    policy_fn: Optional[Callable], verbose: bool) -> Dict[str, float]:
        """Run a single experiment"""
        # Create config
        config = SimulationConfig()
        config.n_beavers = n_agents
        config.grid.grid_size = grid_size
        config.training.max_steps_per_episode = self.config.steps_per_episode

        # Scale vegetation if configured
        if self.config.scale_resources:
            config.initial_vegetation_density = min(1.0, 0.3 * (grid_size / 32))

        # Create environment and run
        env = MycoBeaverEnv(config)
        obs, info = env.reset(seed=seed)

        collector = MetricsCollector(
            experiment_id=f"scaling_n{n_agents}_seed{seed}",
            config_name=f"n_agents={n_agents}",
            seed=seed,
        )

        for step in range(self.config.steps_per_episode):
            if policy_fn is not None:
                actions = policy_fn(obs, env)
            else:
                actions = {f"agent_{i}": np.random.randint(0, config.policy.n_actions)
                          for i in range(n_agents)}

            obs, rewards, terminated, truncated, info = env.step(actions)
            collector.collect_step(env, info)

            if terminated or truncated:
                break

        collector.finalize()
        env.close()

        return {
            "survival_rate": collector.metrics.final_survival_rate,
            "infrastructure": collector.metrics.peak_infrastructure,
            "info_cost": collector.metrics.total_info_cost,
            "time_to_stable": collector.metrics.time_to_stable_infrastructure,
        }

    def fit_scaling_law(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Fit power law: y = c * x^alpha

        Returns (alpha, r_squared)
        """
        # Log-log linear regression
        log_x = np.log(x)
        log_y = np.log(y + 1e-10)

        # Linear fit
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = coeffs[0]

        # R-squared
        y_pred = coeffs[0] * log_x + coeffs[1]
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return float(alpha), float(r2)

    def run_all(self, policy_fn: Optional[Callable] = None,
                verbose: bool = True) -> ScalingResults:
        """
        Run the complete scaling experiment.

        Returns ScalingResults with fitted scaling laws.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("SCALING LAW EXPERIMENT")
            print("=" * 60)
            print(f"Agent counts: {self.config.agent_counts}")
            print(f"Grid scaling: {'Yes' if self.config.scale_grid else 'No'}")
            print(f"Seeds: {self.config.n_seeds}")

        scaling_points = []

        for n_agents in self.config.agent_counts:
            if verbose:
                grid_size = self.get_grid_size(n_agents)
                print(f"\n{'='*40}")
                print(f"Testing n_agents={n_agents} (grid={grid_size}x{grid_size})")
                print(f"{'='*40}")

            point = self.run_scaling_point(n_agents, policy_fn, verbose)
            scaling_points.append(point)

            if verbose:
                print(f"  Results:")
                print(f"    Survival: {point.survival_rate:.2%}")
                print(f"    Infra/agent: {point.infrastructure_per_agent:.3f}")
                print(f"    Info cost/agent: {point.info_cost_per_agent:.2f}")
                print(f"    Efficiency: {point.coordination_efficiency:.4f}")

        # Fit scaling laws
        agent_counts = np.array([p.n_agents for p in scaling_points])
        infra_per_agent = np.array([p.infrastructure_per_agent for p in scaling_points])
        info_per_agent = np.array([p.info_cost_per_agent for p in scaling_points])
        efficiency = np.array([p.coordination_efficiency for p in scaling_points])

        infra_exp, infra_r2 = self.fit_scaling_law(agent_counts, infra_per_agent)
        info_exp, info_r2 = self.fit_scaling_law(agent_counts, info_per_agent)
        eff_exp, eff_r2 = self.fit_scaling_law(agent_counts, efficiency)

        results = ScalingResults(
            config=self.config,
            scaling_points=scaling_points,
            infrastructure_exponent=infra_exp,
            infrastructure_r2=infra_r2,
            info_cost_exponent=info_exp,
            info_cost_r2=info_r2,
            efficiency_exponent=eff_exp,
            efficiency_r2=eff_r2,
        )

        if verbose:
            self._print_summary(results)

        self.results = scaling_points
        return results

    def _print_summary(self, results: ScalingResults):
        """Print scaling law summary"""
        print("\n" + "=" * 70)
        print("SCALING LAW RESULTS")
        print("=" * 70)

        print("\nFitted Scaling Laws (y ~ n^α):")
        print("-" * 70)
        print(f"  Infrastructure/agent: α = {results.infrastructure_exponent:.3f} "
              f"(R² = {results.infrastructure_r2:.3f})")
        print(f"  Info cost/agent:      α = {results.info_cost_exponent:.3f} "
              f"(R² = {results.info_cost_r2:.3f})")
        print(f"  Coordination eff:     α = {results.efficiency_exponent:.3f} "
              f"(R² = {results.efficiency_r2:.3f})")

        print("\nInterpretation:")
        print("-" * 70)

        # Infrastructure scaling
        if results.infrastructure_exponent > 0:
            print(f"  - Infrastructure per agent INCREASES with colony size (superlinear coordination)")
        elif results.infrastructure_exponent < -0.1:
            print(f"  - Infrastructure per agent DECREASES with colony size (diminishing returns)")
        else:
            print(f"  - Infrastructure per agent is roughly CONSTANT (linear scaling)")

        # Info cost scaling
        if results.info_cost_exponent > 0.2:
            print(f"  - Communication overhead INCREASES with scale (coordination bottleneck)")
        elif results.info_cost_exponent < 0:
            print(f"  - Communication becomes MORE EFFICIENT at scale (emergent organization)")
        else:
            print(f"  - Communication cost scales linearly (no emergent savings)")

        # Efficiency
        if results.efficiency_exponent > 0:
            print(f"  - System becomes MORE efficient at larger scales (positive network effects)")
        else:
            print(f"  - System becomes LESS efficient at larger scales (coordination breakdown)")

        print("\n" + "=" * 70)

        # Data table
        print("\nDetailed Results:")
        print("-" * 70)
        print(f"{'N Agents':<10} {'Grid':<8} {'Survival':<10} {'Infra/N':<10} "
              f"{'Info/N':<10} {'Efficiency':<10}")
        print("-" * 70)

        for p in results.scaling_points:
            print(f"{p.n_agents:<10} {p.grid_size:<8} {p.survival_rate:<10.2%} "
                  f"{p.infrastructure_per_agent:<10.3f} {p.info_cost_per_agent:<10.2f} "
                  f"{p.coordination_efficiency:<10.4f}")

        print("=" * 70)


def run_scaling_experiment(agent_counts: Optional[List[int]] = None,
                           n_seeds: int = 5,
                           steps_per_episode: int = 500,
                           scale_grid: bool = False,
                           policy_fn: Optional[Callable] = None,
                           verbose: bool = True) -> ScalingResults:
    """
    Convenience function to run scaling experiment.

    Args:
        agent_counts: List of agent counts to test
        n_seeds: Seeds per count
        steps_per_episode: Episode length
        scale_grid: Whether to scale grid with agents
        policy_fn: Optional trained policy
        verbose: Print progress

    Returns:
        ScalingResults with fitted laws

    Example:
        >>> results = run_scaling_experiment(agent_counts=[10, 50, 100])
        >>> print(f"Infrastructure scales as n^{results.infrastructure_exponent:.2f}")
    """
    config = ScalingConfig(
        n_seeds=n_seeds,
        steps_per_episode=steps_per_episode,
        scale_grid=scale_grid,
    )

    if agent_counts:
        config.agent_counts = agent_counts

    experiment = ScalingExperiment(config)
    return experiment.run_all(policy_fn=policy_fn, verbose=verbose)
