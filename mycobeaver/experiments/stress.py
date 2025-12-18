"""
Stress Test Experiments
=======================

PHASE 5: Experiment 2 - System resilience under stress.

Introduce:
- Floods (sudden water increase)
- Droughts (water removal)
- Communication noise (message corruption)
- Agent loss (sudden death events)
- Infrastructure damage (dam destruction)

Measure:
- Recovery time
- Knowledge decay
- Structural reuse
- Adaptation speed

This tests the robustness and resilience of emergent coordination.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import copy

from ..config import SimulationConfig
from ..environment import MycoBeaverEnv
from .metrics import MetricsCollector, aggregate_metrics, AggregatedMetrics


class StressType(Enum):
    """Types of stress to apply"""
    FLOOD = "flood"
    DROUGHT = "drought"
    COMMUNICATION_NOISE = "comm_noise"
    AGENT_LOSS = "agent_loss"
    INFRASTRUCTURE_DAMAGE = "infra_damage"
    COMBINED = "combined"


@dataclass
class StressScenario:
    """Definition of a stress scenario"""
    name: str
    stress_type: StressType

    # When to apply stress (step numbers)
    onset_step: int = 100
    duration: int = 50  # How long stress lasts
    recovery_window: int = 200  # Steps to measure recovery

    # Intensity parameters
    intensity: float = 0.5  # 0-1 scale

    # Type-specific parameters
    flood_water_increase: float = 2.0  # Multiplier
    drought_water_decrease: float = 0.1  # Target fraction
    noise_corruption_rate: float = 0.3  # Fraction of messages corrupted
    agent_loss_fraction: float = 0.3  # Fraction of agents to kill
    damage_fraction: float = 0.5  # Fraction of infrastructure to damage


@dataclass
class StressConfig:
    """Configuration for stress experiments"""
    # Scenarios to test
    scenarios: List[StressScenario] = field(default_factory=lambda: [
        StressScenario("mild_flood", StressType.FLOOD, intensity=0.3),
        StressScenario("severe_flood", StressType.FLOOD, intensity=0.8),
        StressScenario("mild_drought", StressType.DROUGHT, intensity=0.3),
        StressScenario("severe_drought", StressType.DROUGHT, intensity=0.8),
        StressScenario("comm_disruption", StressType.COMMUNICATION_NOISE, intensity=0.5),
        StressScenario("agent_loss_20", StressType.AGENT_LOSS, intensity=0.2,
                      agent_loss_fraction=0.2),
        StressScenario("agent_loss_50", StressType.AGENT_LOSS, intensity=0.5,
                      agent_loss_fraction=0.5),
        StressScenario("infra_damage", StressType.INFRASTRUCTURE_DAMAGE, intensity=0.5),
        StressScenario("combined_stress", StressType.COMBINED, intensity=0.5),
    ])

    # Experiment parameters
    steps_per_episode: int = 500
    n_seeds: int = 5
    n_beavers: int = 20
    grid_size: int = 32

    # Baseline steps before stress
    warmup_steps: int = 100

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/stress"))


@dataclass
class RecoveryMetrics:
    """Metrics specifically for recovery analysis"""
    scenario_name: str
    stress_type: str

    # Pre-stress baseline
    pre_stress_survival: float
    pre_stress_structures: int
    pre_stress_coherence: float

    # Stress impact
    min_survival_during_stress: float
    structures_lost: int
    coherence_drop: float

    # Recovery
    recovery_time: int  # Steps to return to 90% of baseline
    final_survival: float
    final_structures: int
    final_coherence: float

    # Structural reuse
    structures_rebuilt: int
    reuse_rate: float  # Fraction of original structures rebuilt


class StressExperiment:
    """
    Stress testing framework for MycoBeaver.

    Applies various environmental and system stresses to measure
    resilience and recovery capabilities.
    """

    def __init__(self, config: Optional[StressConfig] = None):
        self.config = config or StressConfig()
        self.results: Dict[str, List[RecoveryMetrics]] = {}

    def apply_stress(self, env: MycoBeaverEnv, scenario: StressScenario,
                     step: int, rng: np.random.Generator):
        """Apply stress to the environment based on scenario"""
        if step < scenario.onset_step or step > scenario.onset_step + scenario.duration:
            return  # Not in stress window

        stress_type = scenario.stress_type
        intensity = scenario.intensity

        if stress_type == StressType.FLOOD or stress_type == StressType.COMBINED:
            self._apply_flood(env, intensity * scenario.flood_water_increase, rng)

        if stress_type == StressType.DROUGHT:
            self._apply_drought(env, 1.0 - intensity * (1 - scenario.drought_water_decrease))

        if stress_type == StressType.COMMUNICATION_NOISE or stress_type == StressType.COMBINED:
            self._apply_comm_noise(env, intensity * scenario.noise_corruption_rate, rng)

        if stress_type == StressType.AGENT_LOSS:
            # Only apply once at onset
            if step == scenario.onset_step:
                self._apply_agent_loss(env, scenario.agent_loss_fraction, rng)

        if stress_type == StressType.INFRASTRUCTURE_DAMAGE or stress_type == StressType.COMBINED:
            # Only apply once at onset
            if step == scenario.onset_step:
                self._apply_infra_damage(env, scenario.damage_fraction, rng)

    def _apply_flood(self, env: MycoBeaverEnv, multiplier: float, rng: np.random.Generator):
        """Increase water levels"""
        # Add water especially in low areas
        flood_water = rng.uniform(0.2, 0.5, env.grid_state.water_depth.shape) * multiplier
        env.grid_state.water_depth += flood_water
        env.grid_state.water_depth = np.clip(env.grid_state.water_depth, 0, 3.0)

    def _apply_drought(self, env: MycoBeaverEnv, target_fraction: float):
        """Reduce water levels"""
        env.grid_state.water_depth *= target_fraction

    def _apply_comm_noise(self, env: MycoBeaverEnv, corruption_rate: float,
                          rng: np.random.Generator):
        """Add noise to communication (if communication hub exists)"""
        if hasattr(env, 'communication_hub') and env.communication_hub is not None:
            # Corrupt pending messages
            hub = env.communication_hub
            if hasattr(hub, 'pending_messages'):
                for msg in hub.pending_messages:
                    if rng.random() < corruption_rate:
                        # Corrupt message by randomizing content
                        if 'content' in msg:
                            msg['corrupted'] = True

    def _apply_agent_loss(self, env: MycoBeaverEnv, fraction: float,
                          rng: np.random.Generator):
        """Kill a fraction of agents"""
        alive_agents = [a for a in env.agents if a.alive]
        n_to_kill = int(len(alive_agents) * fraction)

        # Randomly select agents to kill
        victims = rng.choice(alive_agents, size=n_to_kill, replace=False)
        for agent in victims:
            agent.alive = False
            agent.energy = 0
            y, x = agent.position
            env.grid_state.agent_positions[y, x] -= 1

    def _apply_infra_damage(self, env: MycoBeaverEnv, fraction: float,
                            rng: np.random.Generator):
        """Damage existing infrastructure"""
        # Find cells with dams
        dam_mask = env.grid_state.dam_permeability < 0.9
        dam_cells = np.argwhere(dam_mask)

        if len(dam_cells) == 0:
            return

        n_to_damage = int(len(dam_cells) * fraction)
        damage_idx = rng.choice(len(dam_cells), size=min(n_to_damage, len(dam_cells)),
                               replace=False)

        for idx in damage_idx:
            y, x = dam_cells[idx]
            # Reduce dam strength (increase permeability)
            env.grid_state.dam_permeability[y, x] = min(
                1.0,
                env.grid_state.dam_permeability[y, x] + 0.5
            )

    def run_scenario(self, scenario: StressScenario,
                      seed: int,
                      policy_fn: Optional[Callable] = None,
                      verbose: bool = False) -> Tuple[MetricsCollector, RecoveryMetrics]:
        """
        Run a single stress scenario.

        Returns both full metrics and recovery-specific metrics.
        """
        # Create config and environment
        config = SimulationConfig()
        config.n_beavers = self.config.n_beavers
        config.grid.grid_size = self.config.grid_size
        config.training.max_steps_per_episode = self.config.steps_per_episode

        env = MycoBeaverEnv(config)
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Metrics collector
        collector = MetricsCollector(
            experiment_id=f"stress_{scenario.name}_seed{seed}",
            config_name=scenario.name,
            seed=seed,
        )

        # Track for recovery metrics
        pre_stress_metrics = None
        min_survival = 1.0
        structures_at_onset = 0
        coherence_at_onset = 1.0
        recovery_step = -1

        for step in range(self.config.steps_per_episode):
            # Record pre-stress baseline
            if step == scenario.onset_step - 1:
                pre_stress_metrics = collector.metrics.step_metrics[-1] if collector.metrics.step_metrics else None
                structures_at_onset = np.sum(env.grid_state.dam_permeability < 0.9)
                if env.semantic_system:
                    coherence_at_onset = env.semantic_system.shared_graph.compute_coherence()

            # Apply stress
            self.apply_stress(env, scenario, step, rng)

            # Get actions
            if policy_fn is not None:
                actions = policy_fn(obs, env)
            else:
                actions = {f"agent_{i}": np.random.randint(0, config.policy.n_actions)
                          for i in range(config.n_beavers)}

            # Step
            obs, rewards, terminated, truncated, info = env.step(actions)
            collector.collect_step(env, info)

            # Track during stress
            current_survival = sum(1 for a in env.agents if a.alive) / len(env.agents)
            if scenario.onset_step <= step <= scenario.onset_step + scenario.duration:
                min_survival = min(min_survival, current_survival)

            # Check recovery (90% of baseline survival)
            if pre_stress_metrics and recovery_step < 0:
                if step > scenario.onset_step + scenario.duration:
                    if current_survival >= 0.9 * pre_stress_metrics.survival_rate:
                        recovery_step = step - (scenario.onset_step + scenario.duration)

            if terminated or truncated:
                break

            if verbose and step % 100 == 0:
                print(f"    Step {step}: survival={current_survival:.2%}")

        collector.finalize()

        # Compute recovery metrics
        final_metrics = collector.metrics.step_metrics[-1] if collector.metrics.step_metrics else None
        final_structures = int(np.sum(env.grid_state.dam_permeability < 0.9))
        final_coherence = 1.0
        if env.semantic_system:
            final_coherence = env.semantic_system.shared_graph.compute_coherence()

        recovery = RecoveryMetrics(
            scenario_name=scenario.name,
            stress_type=scenario.stress_type.value,
            pre_stress_survival=pre_stress_metrics.survival_rate if pre_stress_metrics else 1.0,
            pre_stress_structures=int(structures_at_onset),
            pre_stress_coherence=coherence_at_onset,
            min_survival_during_stress=min_survival,
            structures_lost=max(0, int(structures_at_onset) - final_structures),
            coherence_drop=max(0, coherence_at_onset - final_coherence),
            recovery_time=recovery_step if recovery_step > 0 else self.config.steps_per_episode,
            final_survival=final_metrics.survival_rate if final_metrics else 0,
            final_structures=final_structures,
            final_coherence=final_coherence,
            structures_rebuilt=max(0, final_structures - (int(structures_at_onset) - int(structures_at_onset * scenario.damage_fraction))),
            reuse_rate=final_structures / max(1, structures_at_onset),
        )

        env.close()
        return collector, recovery

    def run_all(self, policy_fn: Optional[Callable] = None,
                verbose: bool = True) -> Dict[str, List[RecoveryMetrics]]:
        """Run all stress scenarios"""
        if verbose:
            print("\n" + "=" * 60)
            print("STRESS TEST EXPERIMENTS")
            print("=" * 60)
            print(f"Scenarios: {[s.name for s in self.config.scenarios]}")
            print(f"Seeds per scenario: {self.config.n_seeds}")

        results = {}

        for scenario in self.config.scenarios:
            if verbose:
                print(f"\n{'='*40}")
                print(f"Scenario: {scenario.name} ({scenario.stress_type.value})")
                print(f"{'='*40}")

            recovery_metrics = []
            for seed in range(self.config.n_seeds):
                if verbose:
                    print(f"\n  Seed {seed + 1}/{self.config.n_seeds}")

                collector, recovery = self.run_scenario(
                    scenario, seed, policy_fn, verbose=verbose
                )
                recovery_metrics.append(recovery)

                if verbose:
                    print(f"    Min survival: {recovery.min_survival_during_stress:.2%}")
                    print(f"    Recovery time: {recovery.recovery_time} steps")
                    print(f"    Structures reuse: {recovery.reuse_rate:.2%}")

            results[scenario.name] = recovery_metrics

        # Print summary
        if verbose:
            self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, List[RecoveryMetrics]]):
        """Print summary of stress test results"""
        print("\n" + "=" * 90)
        print("STRESS TEST RESULTS SUMMARY")
        print("=" * 90)

        print(f"{'Scenario':<20} {'Min Survival':<15} {'Recovery Time':<15} "
              f"{'Structures Lost':<15} {'Reuse Rate':<15}")
        print("-" * 90)

        for name, metrics in results.items():
            min_survivals = [m.min_survival_during_stress for m in metrics]
            recovery_times = [m.recovery_time for m in metrics]
            structures_lost = [m.structures_lost for m in metrics]
            reuse_rates = [m.reuse_rate for m in metrics]

            print(f"{name:<20} "
                  f"{np.mean(min_survivals):.2%} ± {np.std(min_survivals):.2%}    "
                  f"{np.mean(recovery_times):.0f} ± {np.std(recovery_times):.0f}        "
                  f"{np.mean(structures_lost):.1f} ± {np.std(structures_lost):.1f}        "
                  f"{np.mean(reuse_rates):.2%} ± {np.std(reuse_rates):.2%}")

        print("=" * 90)


def run_stress_tests(scenarios: Optional[List[StressScenario]] = None,
                     n_seeds: int = 5,
                     steps_per_episode: int = 500,
                     policy_fn: Optional[Callable] = None,
                     verbose: bool = True) -> Dict[str, List[RecoveryMetrics]]:
    """
    Convenience function to run stress tests.

    Args:
        scenarios: List of stress scenarios (default: standard set)
        n_seeds: Seeds per scenario
        steps_per_episode: Episode length
        policy_fn: Optional trained policy
        verbose: Print progress

    Returns:
        Dict mapping scenario name to recovery metrics

    Example:
        >>> results = run_stress_tests(n_seeds=3)
        >>> print(results["severe_flood"][0].recovery_time)
    """
    config = StressConfig(
        n_seeds=n_seeds,
        steps_per_episode=steps_per_episode,
    )

    if scenarios:
        config.scenarios = scenarios

    experiment = StressExperiment(config)
    return experiment.run_all(policy_fn=policy_fn, verbose=verbose)
