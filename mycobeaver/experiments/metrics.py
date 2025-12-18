"""
Metrics Collection Framework
============================

PHASE 5: Comprehensive metrics for publishable experiments.

Collects emergent capability metrics, not just reward curves:
- Time to stable infrastructure
- Water efficiency
- Survival rate
- Info cost efficiency
- Knowledge coherence
- Coordination metrics
- Recovery dynamics
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time
from pathlib import Path

# Try to import pandas for data analysis
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class StepMetrics:
    """Metrics collected at each simulation step"""
    step: int
    timestamp: float

    # Colony health
    n_alive: int
    n_total: int
    survival_rate: float
    avg_energy: float
    min_energy: float
    avg_info_energy: float
    min_info_energy: float

    # Infrastructure
    n_structures: int
    total_dam_strength: float
    infrastructure_coverage: float  # Fraction of potential dam sites used

    # Hydrology
    avg_water_depth: float
    water_variance: float
    flood_cells: int
    drought_cells: int
    water_efficiency: float  # 1 - variance / mean

    # Vegetation
    total_vegetation: float
    avg_vegetation: float
    wetland_cells: int  # Cells with both water and vegetation

    # Information thermodynamics
    info_spent: float
    info_gained: float
    net_info: float
    semantic_entropy: float
    coherence: float
    n_contradictions: int

    # Coordination
    n_active_projects: int
    n_completed_projects: int
    n_abandoned_projects: int
    avg_project_duration: float

    # Performance
    episode_reward: float
    cumulative_reward: float


@dataclass
class ExperimentMetrics:
    """Complete metrics for a single experiment run"""
    experiment_id: str
    config_name: str
    seed: int
    start_time: float
    end_time: float = 0.0

    # Per-step history
    step_metrics: List[StepMetrics] = field(default_factory=list)

    # Derived metrics (computed at end)
    time_to_stable_infrastructure: int = -1  # Steps until infrastructure stabilizes
    final_survival_rate: float = 0.0
    total_info_cost: float = 0.0
    peak_infrastructure: int = 0
    infrastructure_efficiency: float = 0.0  # Structures / info spent
    recovery_events: List[Dict[str, Any]] = field(default_factory=list)

    # Summary statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    max_coherence: float = 0.0
    min_coherence: float = 1.0
    total_projects_completed: int = 0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple experiment runs"""
    experiment_name: str
    n_runs: int
    config_names: List[str]

    # Mean and std for key metrics across runs
    survival_rate_mean: float = 0.0
    survival_rate_std: float = 0.0

    time_to_stable_mean: float = 0.0
    time_to_stable_std: float = 0.0

    infrastructure_efficiency_mean: float = 0.0
    infrastructure_efficiency_std: float = 0.0

    info_cost_mean: float = 0.0
    info_cost_std: float = 0.0

    coherence_mean: float = 0.0
    coherence_std: float = 0.0

    reward_mean: float = 0.0
    reward_std: float = 0.0

    # Per-config breakdown
    per_config_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class MetricsCollector:
    """
    Comprehensive metrics collector for experiments.

    Tracks all relevant metrics during simulation and computes
    derived metrics for analysis.
    """

    def __init__(self, experiment_id: str, config_name: str, seed: int):
        self.experiment_id = experiment_id
        self.config_name = config_name
        self.seed = seed
        self.start_time = time.time()

        self.metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            config_name=config_name,
            seed=seed,
            start_time=self.start_time,
        )

        # Tracking for derived metrics
        self._infrastructure_history: List[int] = []
        self._stability_window = 50  # Steps to check for stability
        self._stability_threshold = 0.1  # Max variance for "stable"

        # Recovery tracking
        self._last_survival_rate = 1.0
        self._in_crisis = False
        self._crisis_start = 0

        # Project tracking
        self._project_start_times: Dict[int, int] = {}
        self._project_durations: List[int] = []

    def collect_step(self, env: 'MycoBeaverEnv', info: Dict[str, Any]):
        """Collect metrics for current step"""
        step = env.current_step
        gs = env.grid_state

        # Colony health
        alive_agents = [a for a in env.agents if a.alive]
        n_alive = len(alive_agents)
        n_total = len(env.agents)

        energies = [a.energy for a in alive_agents] if alive_agents else [0]
        info_energies = [a.info_energy for a in alive_agents] if alive_agents else [0]

        # Infrastructure
        n_structures = int(np.sum(gs.dam_permeability < 0.9))
        total_dam_strength = float(np.sum(1.0 - gs.dam_permeability))
        grid_size = gs.dam_permeability.shape[0]
        infrastructure_coverage = n_structures / (grid_size * grid_size)

        # Hydrology
        avg_water = float(np.mean(gs.water_depth))
        water_var = float(np.var(gs.water_depth))
        flood_cells = int(np.sum(gs.water_depth > 1.0))
        drought_cells = int(np.sum(gs.water_depth < 0.01))
        water_efficiency = 1.0 - (water_var / (avg_water + 1e-6))

        # Vegetation
        total_veg = float(np.sum(gs.vegetation))
        avg_veg = float(np.mean(gs.vegetation))
        wetland = int(np.sum((gs.water_depth > 0.1) & (gs.vegetation > 0.3)))

        # Information thermodynamics
        info_spent = 0.0
        info_gained = 0.0
        semantic_entropy = 0.0
        coherence = 1.0
        n_contradictions = 0

        if env.pheromone_field is not None:
            info_spent += env.pheromone_field.get_info_dissipation()

        if env.semantic_system is not None:
            info_spent += env.semantic_system.get_info_dissipation()
            semantic_entropy = env.semantic_system.shared_graph.compute_semantic_entropy()
            coherence = env.semantic_system.shared_graph.compute_coherence()
            n_contradictions = len(env.semantic_system.shared_graph.contradictions)

        # Estimate info gained from recovery
        if self.metrics.step_metrics:
            prev_total_info = sum(info_energies)
            curr_total_info = sum(info_energies)
            info_gained = max(0, curr_total_info - prev_total_info + info_spent)

        # Projects
        n_active = 0
        n_completed = 0
        n_abandoned = 0
        if env.project_manager is not None:
            n_active = len(env.project_manager.active_projects)
            n_completed = getattr(env.project_manager, 'completed_count', 0)
            n_abandoned = getattr(env.project_manager, 'abandoned_count', 0)

        # Cumulative reward
        cumulative = env.episode_reward

        step_metric = StepMetrics(
            step=step,
            timestamp=time.time() - self.start_time,
            n_alive=n_alive,
            n_total=n_total,
            survival_rate=n_alive / max(1, n_total),
            avg_energy=float(np.mean(energies)),
            min_energy=float(np.min(energies)),
            avg_info_energy=float(np.mean(info_energies)),
            min_info_energy=float(np.min(info_energies)),
            n_structures=n_structures,
            total_dam_strength=total_dam_strength,
            infrastructure_coverage=infrastructure_coverage,
            avg_water_depth=avg_water,
            water_variance=water_var,
            flood_cells=flood_cells,
            drought_cells=drought_cells,
            water_efficiency=water_efficiency,
            total_vegetation=total_veg,
            avg_vegetation=avg_veg,
            wetland_cells=wetland,
            info_spent=info_spent,
            info_gained=info_gained,
            net_info=info_gained - info_spent,
            semantic_entropy=semantic_entropy,
            coherence=coherence,
            n_contradictions=n_contradictions,
            n_active_projects=n_active,
            n_completed_projects=n_completed,
            n_abandoned_projects=n_abandoned,
            avg_project_duration=np.mean(self._project_durations) if self._project_durations else 0,
            episode_reward=info.get("rewards", {}).get("total", 0) if isinstance(info.get("rewards"), dict) else 0,
            cumulative_reward=cumulative,
        )

        self.metrics.step_metrics.append(step_metric)

        # Track infrastructure for stability detection
        self._infrastructure_history.append(n_structures)

        # Track recovery events
        survival_rate = n_alive / max(1, n_total)
        if survival_rate < 0.7 and not self._in_crisis:
            self._in_crisis = True
            self._crisis_start = step
        elif survival_rate > 0.9 and self._in_crisis:
            recovery_time = step - self._crisis_start
            self.metrics.recovery_events.append({
                "crisis_start": self._crisis_start,
                "recovery_step": step,
                "recovery_time": recovery_time,
                "min_survival": self._last_survival_rate,
            })
            self._in_crisis = False

        self._last_survival_rate = min(self._last_survival_rate, survival_rate)

        # Accumulate info cost
        self.metrics.total_info_cost += info_spent

    def finalize(self):
        """Compute derived metrics at end of experiment"""
        self.metrics.end_time = time.time()

        if not self.metrics.step_metrics:
            return

        # Time to stable infrastructure
        self.metrics.time_to_stable_infrastructure = self._compute_stability_time()

        # Final survival
        self.metrics.final_survival_rate = self.metrics.step_metrics[-1].survival_rate

        # Peak infrastructure
        self.metrics.peak_infrastructure = max(
            m.n_structures for m in self.metrics.step_metrics
        )

        # Infrastructure efficiency
        if self.metrics.total_info_cost > 0:
            self.metrics.infrastructure_efficiency = (
                self.metrics.peak_infrastructure / self.metrics.total_info_cost
            )

        # Reward statistics
        rewards = [m.cumulative_reward for m in self.metrics.step_metrics]
        self.metrics.mean_reward = float(np.mean(rewards))
        self.metrics.std_reward = float(np.std(rewards))

        # Coherence statistics
        coherences = [m.coherence for m in self.metrics.step_metrics]
        self.metrics.max_coherence = max(coherences)
        self.metrics.min_coherence = min(coherences)

        # Projects
        self.metrics.total_projects_completed = self.metrics.step_metrics[-1].n_completed_projects

    def _compute_stability_time(self) -> int:
        """Find the first step where infrastructure becomes stable"""
        if len(self._infrastructure_history) < self._stability_window:
            return -1

        for i in range(len(self._infrastructure_history) - self._stability_window):
            window = self._infrastructure_history[i:i + self._stability_window]
            if len(window) > 0:
                variance = np.var(window)
                mean = np.mean(window)
                if mean > 0 and variance / mean < self._stability_threshold:
                    return i

        return -1  # Never stabilized

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary"""
        return {
            "experiment_id": self.metrics.experiment_id,
            "config_name": self.metrics.config_name,
            "seed": self.metrics.seed,
            "duration": self.metrics.end_time - self.metrics.start_time,
            "n_steps": len(self.metrics.step_metrics),
            "time_to_stable": self.metrics.time_to_stable_infrastructure,
            "final_survival_rate": self.metrics.final_survival_rate,
            "total_info_cost": self.metrics.total_info_cost,
            "peak_infrastructure": self.metrics.peak_infrastructure,
            "infrastructure_efficiency": self.metrics.infrastructure_efficiency,
            "mean_reward": self.metrics.mean_reward,
            "std_reward": self.metrics.std_reward,
            "max_coherence": self.metrics.max_coherence,
            "min_coherence": self.metrics.min_coherence,
            "total_projects": self.metrics.total_projects_completed,
            "recovery_events": len(self.metrics.recovery_events),
        }

    def to_dataframe(self) -> 'pd.DataFrame':
        """Export step metrics as pandas DataFrame"""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame export")

        data = []
        for m in self.metrics.step_metrics:
            data.append({
                "step": m.step,
                "timestamp": m.timestamp,
                "survival_rate": m.survival_rate,
                "n_structures": m.n_structures,
                "water_efficiency": m.water_efficiency,
                "coherence": m.coherence,
                "info_spent": m.info_spent,
                "cumulative_reward": m.cumulative_reward,
                "semantic_entropy": m.semantic_entropy,
                "n_contradictions": m.n_contradictions,
                "wetland_cells": m.wetland_cells,
            })

        return pd.DataFrame(data)

    def save(self, path: Path):
        """Save metrics to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save summary
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save detailed metrics if pandas available
        if HAS_PANDAS:
            df = self.to_dataframe()
            df.to_csv(path.with_suffix('.csv'), index=False)


def aggregate_metrics(collectors: List[MetricsCollector],
                      experiment_name: str) -> AggregatedMetrics:
    """Aggregate metrics across multiple experiment runs"""
    if not collectors:
        return AggregatedMetrics(experiment_name=experiment_name, n_runs=0, config_names=[])

    config_names = list(set(c.config_name for c in collectors))

    # Collect per-run values
    survival_rates = [c.metrics.final_survival_rate for c in collectors]
    stability_times = [c.metrics.time_to_stable_infrastructure for c in collectors
                      if c.metrics.time_to_stable_infrastructure > 0]
    efficiencies = [c.metrics.infrastructure_efficiency for c in collectors]
    info_costs = [c.metrics.total_info_cost for c in collectors]
    coherences = [c.metrics.max_coherence for c in collectors]
    rewards = [c.metrics.mean_reward for c in collectors]

    # Per-config breakdown
    per_config = defaultdict(lambda: defaultdict(list))
    for c in collectors:
        per_config[c.config_name]["survival_rate"].append(c.metrics.final_survival_rate)
        per_config[c.config_name]["time_to_stable"].append(
            c.metrics.time_to_stable_infrastructure)
        per_config[c.config_name]["efficiency"].append(c.metrics.infrastructure_efficiency)
        per_config[c.config_name]["info_cost"].append(c.metrics.total_info_cost)
        per_config[c.config_name]["coherence"].append(c.metrics.max_coherence)
        per_config[c.config_name]["reward"].append(c.metrics.mean_reward)

    # Compute per-config means
    per_config_metrics = {}
    for config, metrics in per_config.items():
        per_config_metrics[config] = {
            key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for key, values in metrics.items()
        }

    return AggregatedMetrics(
        experiment_name=experiment_name,
        n_runs=len(collectors),
        config_names=config_names,
        survival_rate_mean=float(np.mean(survival_rates)),
        survival_rate_std=float(np.std(survival_rates)),
        time_to_stable_mean=float(np.mean(stability_times)) if stability_times else -1,
        time_to_stable_std=float(np.std(stability_times)) if stability_times else 0,
        infrastructure_efficiency_mean=float(np.mean(efficiencies)),
        infrastructure_efficiency_std=float(np.std(efficiencies)),
        info_cost_mean=float(np.mean(info_costs)),
        info_cost_std=float(np.std(info_costs)),
        coherence_mean=float(np.mean(coherences)),
        coherence_std=float(np.std(coherences)),
        reward_mean=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)),
        per_config_metrics=per_config_metrics,
    )
