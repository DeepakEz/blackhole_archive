"""
MycoBeaver Overmind Meta-Controller
====================================
Meta-level controller that adjusts colony parameters based on global state.

ARCHITECTURAL CONTRACT:
======================
Overmind is a MODULATOR, not a CONTROLLER.

Overmind IS allowed to control:
- PPO learning rate (global or per-agent group)
- Entropy coefficient (exploration pressure)
- Communication bandwidth (max messages / step)
- Pheromone evaporation & diffusion rates
- Project quorum thresholds
- Physarum adaptation speed

Overmind is NOT allowed to:
- Modify rewards directly
- Modify policy logits
- Inject gradients
- Change agent actions

All output flows through OvermindSignals TypedDict.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .config import OvermindConfig, SimulationConfig
from .contracts import (
    OvermindSignals,
    GlobalObservation,
    SignalRouter,
    DEFAULT_OVERMIND_SIGNALS,
    SIGNAL_BOUNDS,
    validate_signals,
    clamp_signals,
    ContractViolationError,
)


@dataclass
class WisdomState:
    """Tracks wisdom signal components over time"""
    water_variance: float = 0.0
    flood_fraction: float = 0.0
    drought_fraction: float = 0.0
    colony_health: float = 1.0
    project_efficiency: float = 0.0
    habitat_quality: float = 0.0

    # === PHASE 2: Information Thermodynamics ===
    info_dissipation_rate: float = 0.0  # Total info energy spent this step
    avg_agent_info_energy: float = 100.0  # Average info energy across agents
    info_blocked_actions: int = 0  # Actions blocked due to insufficient info

    # Derived
    total_wisdom: float = 0.0

    # History for smoothing
    history_len: int = 10
    water_variance_history: List[float] = field(default_factory=list)
    flood_history: List[float] = field(default_factory=list)
    info_dissipation_history: List[float] = field(default_factory=list)


@dataclass
class EmergencyState:
    """
    Tracks emergency conditions for aggressive intervention.

    Emergency types:
    - SURVIVAL_CRISIS: Colony survival drops below threshold
    - WATER_VOLATILITY: Water variance spikes suddenly
    - ENTROPY_COLLAPSE: Policy entropy drops too low
    - RESOURCE_FAMINE: Vegetation depleted critically
    - BUILD_STAGNATION: No structures built for too long
    - EXPLORATION_STAGNATION: Agents not exploring enough

    Each emergency triggers specific countermeasures.
    """
    # Emergency flags
    survival_crisis: bool = False
    water_volatility_spike: bool = False
    entropy_collapse: bool = False
    resource_famine: bool = False
    build_stagnation: bool = False  # No new structures for too long
    exploration_stagnation: bool = False  # Agents not exploring

    # Severity levels (0-1, higher = worse)
    survival_severity: float = 0.0
    volatility_severity: float = 0.0
    entropy_severity: float = 0.0
    famine_severity: float = 0.0
    build_stagnation_severity: float = 0.0
    exploration_severity: float = 0.0

    # Emergency duration (steps in emergency state)
    emergency_duration: int = 0
    max_emergency_duration: int = 100

    # Recovery tracking
    recovery_steps: int = 0
    recovery_threshold: int = 20  # Steps of stability needed to exit emergency

    # Intervention history
    interventions_triggered: int = 0
    interventions_successful: int = 0

    # Stagnation tracking
    steps_since_last_structure: int = 0
    last_structure_count: int = 0
    build_attempts: int = 0

    def is_in_emergency(self) -> bool:
        """Check if any emergency is active."""
        return (self.survival_crisis or self.water_volatility_spike or
                self.entropy_collapse or self.resource_famine or
                self.build_stagnation or self.exploration_stagnation)

    def get_overall_severity(self) -> float:
        """Get combined emergency severity."""
        return max(self.survival_severity, self.volatility_severity,
                   self.entropy_severity, self.famine_severity,
                   self.build_stagnation_severity, self.exploration_severity)


class Overmind:
    """
    Meta-controller for the beaver colony.

    STRICT CONTRACT: This class only produces OvermindSignals.
    It does NOT directly modify environment state, rewards, or policies.

    The SignalRouter is the ONLY interface through which signals flow.

    Key functions:
    1. Observe global state (read-only)
    2. Compute wisdom signal (internal metric)
    3. Produce OvermindSignals (rate/budget modulation)

    The wisdom signal indicates how well the colony is doing:
    - Positive: stable water, healthy agents, successful projects
    - Negative: floods, droughts, agent deaths, failed projects
    """

    def __init__(self, config: OvermindConfig, sim_config: SimulationConfig):
        self.config = config
        self.sim_config = sim_config

        # Signal router - the ONLY output interface
        self.signal_router = SignalRouter()

        # Current signal values (internal state)
        self._current_signals: OvermindSignals = DEFAULT_OVERMIND_SIGNALS.copy()

        # Wisdom state
        self.wisdom_state = WisdomState()

        # === EMERGENCY INTERVENTION SYSTEM ===
        self.emergency_state = EmergencyState()

        # Emergency thresholds
        self.survival_threshold = 0.5  # Below this = survival crisis
        self.volatility_threshold = 0.3  # Above this = water volatility spike
        self.entropy_threshold = 0.3  # Below this = entropy collapse
        self.famine_threshold = 0.2  # Below this = resource famine

        # Stagnation detection thresholds
        self.build_stagnation_steps = 200  # Steps without new structure = stagnation
        self.exploration_threshold = 0.1  # Below this coverage = exploration stagnation
        self.zero_build_attempts_threshold = 50  # Steps with 0 build attempts = crisis

        # Emergency response strength
        self.emergency_entropy_boost = 3.0  # Multiply entropy by this in emergency
        self.emergency_lr_boost = 0.5  # Reduce LR by this factor in emergency
        self.stagnation_entropy_boost = 2.0  # Boost for build stagnation

        # Observation history (for trend detection)
        self.observation_history: List[np.ndarray] = []
        self.wisdom_history: List[float] = []
        self.entropy_history: List[float] = []  # Track policy entropy

        # Step counter
        self.step_count = 0

        # Smoothing factor for signal changes
        self.signal_smoothing = 0.9

    def produce_signals(self, observation: np.ndarray) -> OvermindSignals:
        """
        MAIN INTERFACE: Produce modulation signals from global observation.

        This is the ONLY method that should be called by external systems.
        Returns a validated OvermindSignals structure.

        Args:
            observation: Global state features (from environment)

        Returns:
            OvermindSignals TypedDict with all modulation values
        """
        self.step_count += 1

        # Store observation for trend analysis
        self.observation_history.append(observation)
        if len(self.observation_history) > 100:
            self.observation_history.pop(0)

        # Compute wisdom signal (internal assessment)
        wisdom = self._compute_wisdom(observation)
        self.wisdom_state.total_wisdom = wisdom
        self.wisdom_history.append(wisdom)
        if len(self.wisdom_history) > 100:
            self.wisdom_history.pop(0)

        # Decide on signal adjustments (rule-based)
        target_signals = self._decide_signals(observation, wisdom)

        # Apply smoothing to avoid sudden changes
        smoothed_signals = self._smooth_signals(target_signals)

        # Validate and clamp
        smoothed_signals = clamp_signals(smoothed_signals)

        # Update internal state
        self._current_signals = smoothed_signals

        # Route signals through the validated interface
        self.signal_router.update_signals(smoothed_signals)

        return smoothed_signals

    def _compute_wisdom(self, observation: np.ndarray) -> float:
        """
        Compute wisdom signal from global observations.

        W = -λ_σ * σ²(water) - λ_F * flood_frac - λ_D * drought_frac
            - λ_B * failures + λ_H * habitat_quality

        This is an internal assessment metric, NOT a reward signal.

        PHASE 2: Now includes info dissipation metrics.
        """
        c = self.config

        # Extract features from observation
        # PHASE 2: Extended observation now includes info metrics
        if len(observation) >= 12:
            water_mean = observation[0]
            water_variance = observation[1]
            water_max = observation[2]
            vegetation = observation[3]
            flood_fraction = observation[4]
            drought_fraction = observation[5]
            agent_survival = observation[6]
            avg_energy = observation[7]
            project_count = observation[8]
            # PHASE 2: Info metrics
            info_dissipation = observation[9]
            avg_info_energy = observation[10]
            info_blocked = observation[11]
        elif len(observation) >= 9:
            water_mean = observation[0]
            water_variance = observation[1]
            water_max = observation[2]
            vegetation = observation[3]
            flood_fraction = observation[4]
            drought_fraction = observation[5]
            agent_survival = observation[6]
            avg_energy = observation[7]
            project_count = observation[8]
            # Fallback info metrics
            info_dissipation = 0.0
            avg_info_energy = 100.0
            info_blocked = 0
        else:
            # Fallback defaults
            water_variance = 0.1
            flood_fraction = 0.1
            drought_fraction = 0.1
            agent_survival = 0.8
            vegetation = 0.5
            info_dissipation = 0.0
            avg_info_energy = 100.0
            info_blocked = 0

        # Update wisdom state
        self.wisdom_state.water_variance = water_variance
        self.wisdom_state.flood_fraction = flood_fraction
        self.wisdom_state.drought_fraction = drought_fraction
        self.wisdom_state.colony_health = agent_survival

        # PHASE 2: Update info metrics
        self.wisdom_state.info_dissipation_rate = info_dissipation
        self.wisdom_state.avg_agent_info_energy = avg_info_energy
        self.wisdom_state.info_blocked_actions = int(info_blocked)

        # Track history
        self.wisdom_state.water_variance_history.append(water_variance)
        self.wisdom_state.flood_history.append(flood_fraction)
        self.wisdom_state.info_dissipation_history.append(info_dissipation)
        if len(self.wisdom_state.water_variance_history) > self.wisdom_state.history_len:
            self.wisdom_state.water_variance_history.pop(0)
            self.wisdom_state.flood_history.pop(0)
            self.wisdom_state.info_dissipation_history.pop(0)

        # Compute habitat quality
        habitat_quality = vegetation * (1.0 - flood_fraction) * (1.0 - drought_fraction)
        self.wisdom_state.habitat_quality = habitat_quality

        # Wisdom calculation (higher = better)
        failure_rate = 1.0 - agent_survival
        wisdom = (
            -c.water_variance_weight * water_variance -
            c.flood_weight * flood_fraction -
            c.drought_weight * drought_fraction -
            c.failure_weight * failure_rate +
            c.habitat_weight * habitat_quality
        )

        return wisdom

    def _detect_emergencies(self, observation: np.ndarray) -> None:
        """
        Detect emergency conditions from current state.

        Emergency types:
        1. SURVIVAL_CRISIS: Colony health drops below threshold
        2. WATER_VOLATILITY_SPIKE: Rapid change in water variance
        3. ENTROPY_COLLAPSE: Policy entropy too low (if tracked)
        4. RESOURCE_FAMINE: Vegetation/resources critically depleted

        Each emergency has a severity level and duration tracking.
        """
        prev_in_emergency = self.emergency_state.is_in_emergency()

        # === SURVIVAL CRISIS ===
        # Triggered when colony health drops below threshold
        survival = self.wisdom_state.colony_health
        if survival < self.survival_threshold:
            self.emergency_state.survival_crisis = True
            # Severity scales with how far below threshold
            self.emergency_state.survival_severity = (
                (self.survival_threshold - survival) / self.survival_threshold
            )
        else:
            self.emergency_state.survival_crisis = False
            self.emergency_state.survival_severity = 0.0

        # === WATER VOLATILITY SPIKE ===
        # Triggered by rapid increase in water variance
        if len(self.wisdom_state.water_variance_history) >= 3:
            recent_variance = np.mean(self.wisdom_state.water_variance_history[-3:])
            older_variance = np.mean(self.wisdom_state.water_variance_history[:-3]) \
                if len(self.wisdom_state.water_variance_history) > 3 else recent_variance

            variance_delta = recent_variance - older_variance
            if variance_delta > self.volatility_threshold or recent_variance > 0.5:
                self.emergency_state.water_volatility_spike = True
                self.emergency_state.volatility_severity = min(1.0, variance_delta / 0.5)
            else:
                self.emergency_state.water_volatility_spike = False
                self.emergency_state.volatility_severity = 0.0
        else:
            self.emergency_state.water_volatility_spike = False
            self.emergency_state.volatility_severity = 0.0

        # === ENTROPY COLLAPSE ===
        # Triggered when policy entropy is too low (exploration exhausted)
        # We track this via the entropy_history if available
        if len(self.entropy_history) >= 3:
            recent_entropy = np.mean(self.entropy_history[-3:])
            if recent_entropy < self.entropy_threshold:
                self.emergency_state.entropy_collapse = True
                self.emergency_state.entropy_severity = (
                    (self.entropy_threshold - recent_entropy) / self.entropy_threshold
                )
            else:
                self.emergency_state.entropy_collapse = False
                self.emergency_state.entropy_severity = 0.0
        else:
            self.emergency_state.entropy_collapse = False
            self.emergency_state.entropy_severity = 0.0

        # === RESOURCE FAMINE ===
        # Triggered when vegetation/habitat quality is critically low
        # Extract vegetation from observation
        if len(observation) >= 4:
            vegetation = observation[3]  # vegetation index in observation
        else:
            vegetation = 0.5  # fallback

        habitat = self.wisdom_state.habitat_quality

        if vegetation < self.famine_threshold or habitat < self.famine_threshold:
            self.emergency_state.resource_famine = True
            self.emergency_state.famine_severity = max(
                (self.famine_threshold - vegetation) / self.famine_threshold,
                (self.famine_threshold - habitat) / self.famine_threshold
            )
        else:
            self.emergency_state.resource_famine = False
            self.emergency_state.famine_severity = 0.0

        # === BUILD STAGNATION ===
        # Triggered when no new structures for too long
        # This requires external info about structure count
        self.emergency_state.steps_since_last_structure += 1
        if self.emergency_state.steps_since_last_structure > self.build_stagnation_steps:
            self.emergency_state.build_stagnation = True
            # Severity scales with duration beyond threshold
            overtime = self.emergency_state.steps_since_last_structure - self.build_stagnation_steps
            self.emergency_state.build_stagnation_severity = min(1.0, overtime / 200.0)
        else:
            self.emergency_state.build_stagnation = False
            self.emergency_state.build_stagnation_severity = 0.0

        # === EXPLORATION STAGNATION ===
        # Triggered when agents aren't exploring enough
        # Check if exploration coverage from observation is too low
        if len(observation) >= 9:
            # Use project count as proxy for activity (index 8 in observation)
            project_count = observation[8]
            if project_count < 0.1 and self.step_count > 100:
                self.emergency_state.exploration_stagnation = True
                self.emergency_state.exploration_severity = 0.5
            else:
                self.emergency_state.exploration_stagnation = False
                self.emergency_state.exploration_severity = 0.0
        else:
            self.emergency_state.exploration_stagnation = False
            self.emergency_state.exploration_severity = 0.0

        # Track emergency duration
        if self.emergency_state.is_in_emergency():
            self.emergency_state.emergency_duration += 1
            self.emergency_state.recovery_steps = 0

            # Track intervention trigger
            if not prev_in_emergency:
                self.emergency_state.interventions_triggered += 1
        else:
            # Track recovery
            if prev_in_emergency:
                self.emergency_state.recovery_steps += 1
                if self.emergency_state.recovery_steps >= self.emergency_state.recovery_threshold:
                    # Successful recovery
                    self.emergency_state.interventions_successful += 1
                    self.emergency_state.emergency_duration = 0
                    self.emergency_state.recovery_steps = 0

    def _apply_emergency_countermeasures(self, signals: OvermindSignals) -> OvermindSignals:
        """
        Apply aggressive countermeasures during emergencies.

        Countermeasures by emergency type:
        - SURVIVAL_CRISIS: Slow learning, boost exploration, lower quorum
        - WATER_VOLATILITY: Boost pheromone decay, faster physarum
        - ENTROPY_COLLAPSE: Massive entropy boost
        - RESOURCE_FAMINE: Focus on exploration, reduce build pressure

        Countermeasures stack and scale with severity.
        """
        severity = self.emergency_state.get_overall_severity()

        # === SURVIVAL CRISIS COUNTERMEASURES ===
        if self.emergency_state.survival_crisis:
            # Slow down learning to avoid catastrophic forgetting
            signals["lr_scale"] *= self.emergency_lr_boost  # Reduce LR
            # Boost exploration to find better strategies
            signals["entropy_scale"] *= (1.0 + severity * 0.5)
            # Lower quorum to speed up decisions
            signals["quorum_multiplier"] = max(0.4, signals["quorum_multiplier"] - 0.2 * severity)

        # === WATER VOLATILITY COUNTERMEASURES ===
        if self.emergency_state.water_volatility_spike:
            # Boost pheromone decay for faster adaptation
            signals["pheromone_decay"] = min(0.5, signals["pheromone_decay"] * (1.5 + severity))
            # Boost diffusion to spread information faster
            signals["pheromone_diffusion"] = min(0.3, signals["pheromone_diffusion"] * 2.0)
            # Speed up physarum adaptation
            signals["physarum_tau"] = max(0.3, signals["physarum_tau"] * (1.0 - 0.3 * severity))

        # === ENTROPY COLLAPSE COUNTERMEASURES ===
        if self.emergency_state.entropy_collapse:
            # Massive entropy boost to restore exploration
            signals["entropy_scale"] *= self.emergency_entropy_boost
            # Also increase semantic temperature for more diverse reasoning
            signals["semantic_temperature"] = min(5.0, signals["semantic_temperature"] * 2.0)

        # === RESOURCE FAMINE COUNTERMEASURES ===
        if self.emergency_state.resource_famine:
            # Boost exploration to find resources
            signals["entropy_scale"] *= (1.0 + severity * 0.3)
            # Increase dance gain to spread resource information
            signals["dance_gain"] = min(5.0, signals["dance_gain"] * (1.5 + severity))
            # Lower recruitment decay so resource signals persist
            signals["recruitment_decay"] = max(0.01, signals["recruitment_decay"] * 0.5)

        # === BUILD STAGNATION COUNTERMEASURES ===
        if self.emergency_state.build_stagnation:
            # Massive entropy boost to break out of exploitation loop
            signals["entropy_scale"] *= self.stagnation_entropy_boost
            # Lower quorum to make building easier
            signals["quorum_multiplier"] = max(0.3, signals["quorum_multiplier"] - 0.3)
            # Boost dance gain to encourage building recruitment
            signals["dance_gain"] = min(5.0, signals["dance_gain"] * 2.0)
            # Reduce LR to not reinforce bad behavior
            signals["lr_scale"] *= 0.5

        # === EXPLORATION STAGNATION COUNTERMEASURES ===
        if self.emergency_state.exploration_stagnation:
            # Boost entropy to encourage exploration
            signals["entropy_scale"] *= 1.5
            # Increase pheromone decay to forget old paths
            signals["pheromone_decay"] = min(0.4, signals["pheromone_decay"] * 1.5)
            # Higher semantic temperature for more diverse reasoning
            signals["semantic_temperature"] = min(4.0, signals["semantic_temperature"] * 1.5)

        # === EMERGENCY DURATION ESCALATION ===
        # If emergency persists, escalate countermeasures
        if self.emergency_state.emergency_duration > 50:
            # Escalation factor grows with duration
            escalation = min(2.0, 1.0 + (self.emergency_state.emergency_duration - 50) / 100)
            signals["entropy_scale"] *= escalation
            signals["pheromone_decay"] *= escalation

        # Cap escalation to prevent instability
        if self.emergency_state.emergency_duration > self.emergency_state.max_emergency_duration:
            # Max escalation reached - system may need external intervention
            pass

        return signals

    def report_entropy(self, entropy: float) -> None:
        """
        Report current policy entropy for collapse detection.

        Called by training loop to provide entropy information.
        """
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > 100:
            self.entropy_history.pop(0)

    def report_structure_count(self, structure_count: int, build_attempts: int = 0) -> None:
        """
        Report current structure count for stagnation detection.

        Called by training loop to track building progress.

        Args:
            structure_count: Total number of structures in environment
            build_attempts: Number of build actions attempted this step
        """
        # Track build attempts
        self.emergency_state.build_attempts += build_attempts

        # Check if new structure was built
        if structure_count > self.emergency_state.last_structure_count:
            # Reset stagnation counter
            self.emergency_state.steps_since_last_structure = 0
            self.emergency_state.last_structure_count = structure_count
            # Clear build stagnation if active
            if self.emergency_state.build_stagnation:
                self.emergency_state.build_stagnation = False
                self.emergency_state.build_stagnation_severity = 0.0
        else:
            # Structure count didn't increase - this is tracked in _detect_emergencies
            pass

    def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency status for logging."""
        return {
            "in_emergency": self.emergency_state.is_in_emergency(),
            "survival_crisis": self.emergency_state.survival_crisis,
            "water_volatility_spike": self.emergency_state.water_volatility_spike,
            "entropy_collapse": self.emergency_state.entropy_collapse,
            "resource_famine": self.emergency_state.resource_famine,
            "build_stagnation": self.emergency_state.build_stagnation,
            "exploration_stagnation": self.emergency_state.exploration_stagnation,
            "overall_severity": self.emergency_state.get_overall_severity(),
            "build_stagnation_severity": self.emergency_state.build_stagnation_severity,
            "exploration_severity": self.emergency_state.exploration_severity,
            "steps_since_last_structure": self.emergency_state.steps_since_last_structure,
            "emergency_duration": self.emergency_state.emergency_duration,
            "interventions_triggered": self.emergency_state.interventions_triggered,
            "interventions_successful": self.emergency_state.interventions_successful,
        }

    def _decide_signals(self, observation: np.ndarray, wisdom: float) -> OvermindSignals:
        """
        Decide signal values based on wisdom and observations.

        Uses rule-based control to determine modulation factors.
        This does NOT touch rewards or policies - only rates and budgets.

        EMERGENCY SYSTEM: Detects critical conditions and applies aggressive
        countermeasures to prevent colony collapse.
        """
        # Start from current values
        signals: OvermindSignals = self._current_signals.copy()

        # Detect trends
        wisdom_trend = 0.0
        if len(self.wisdom_history) >= 2:
            wisdom_trend = self.wisdom_history[-1] - self.wisdom_history[-2]

        # Detect issues
        flood_issue = self.wisdom_state.flood_fraction > 0.1
        drought_issue = self.wisdom_state.drought_fraction > 0.2
        health_issue = self.wisdom_state.colony_health < 0.7
        variance_issue = self.wisdom_state.water_variance > 0.2
        crisis = flood_issue or drought_issue or health_issue

        # === EMERGENCY DETECTION ===
        self._detect_emergencies(observation)
        in_emergency = self.emergency_state.is_in_emergency()

        # Apply emergency countermeasures if needed
        if in_emergency:
            signals = self._apply_emergency_countermeasures(signals)

        # === LEARNING RATE SCALE ===
        # Slow down learning if things are going well, speed up in crisis
        if crisis:
            signals["lr_scale"] = min(2.0, signals["lr_scale"] + 0.1)
        elif wisdom > 0.5:
            signals["lr_scale"] = max(0.5, signals["lr_scale"] - 0.05)

        # === ENTROPY SCALE ===
        # More exploration when stuck, less when doing well
        if variance_issue or wisdom < -0.5:
            signals["entropy_scale"] = min(3.0, signals["entropy_scale"] + 0.2)
        elif wisdom > 0.3:
            signals["entropy_scale"] = max(0.5, signals["entropy_scale"] - 0.1)

        # === COMMUNICATION BUDGET ===
        # PHASE 2: Throttle based on info dissipation rate
        # High dissipation = agents are spending info fast = reduce budget
        # Low avg info energy = agents are depleted = reduce budget
        # Many blocked actions = system constrained = throttle carefully
        info_strained = (
            self.wisdom_state.avg_agent_info_energy < 50.0 or
            self.wisdom_state.info_blocked_actions > 5
        )
        info_healthy = self.wisdom_state.avg_agent_info_energy > 80.0

        # High dissipation rate means agents are communicating heavily
        # Use history to detect sustained high dissipation
        high_dissipation = False
        if len(self.wisdom_state.info_dissipation_history) >= 3:
            avg_dissipation = np.mean(self.wisdom_state.info_dissipation_history[-3:])
            high_dissipation = avg_dissipation > 20.0  # Threshold for heavy communication

        if info_strained or high_dissipation:
            # Throttle communication to conserve info energy
            signals["comm_budget"] = max(3, signals["comm_budget"] - 2)
        elif crisis and info_healthy:
            # Allow more communication in crisis, but only if info is healthy
            signals["comm_budget"] = min(50, signals["comm_budget"] + 5)
        elif info_healthy:
            # Normal operation with healthy info levels
            signals["comm_budget"] = max(5, signals["comm_budget"] - 1)

        # === PHEROMONE DYNAMICS ===
        # Higher decay = more exploration
        if variance_issue or wisdom < -0.5:
            signals["pheromone_decay"] = min(0.3, signals["pheromone_decay"] + 0.02)
        elif wisdom > 0.5:
            signals["pheromone_decay"] = max(0.05, signals["pheromone_decay"] - 0.01)

        # Diffusion tracks decay
        signals["pheromone_diffusion"] = signals["pheromone_decay"] * 0.3

        # === PHYSARUM DYNAMICS ===
        # Faster adaptation in crisis
        if crisis:
            signals["physarum_tau"] = max(0.5, signals["physarum_tau"] - 0.1)
        else:
            signals["physarum_tau"] = min(2.0, signals["physarum_tau"] + 0.05)

        # Higher flow exponent = stronger reinforcement of good paths
        if wisdom > 0.3:
            signals["physarum_flow_exp"] = min(1.5, signals["physarum_flow_exp"] + 0.05)
        else:
            signals["physarum_flow_exp"] = max(0.8, signals["physarum_flow_exp"] - 0.02)

        # === PROJECT RECRUITMENT ===
        # Lower quorum in crisis (faster decisions)
        if crisis:
            signals["quorum_multiplier"] = max(0.6, signals["quorum_multiplier"] - 0.05)
            signals["recruitment_decay"] = max(0.05, signals["recruitment_decay"] - 0.01)
            signals["dance_gain"] = min(3.0, signals["dance_gain"] + 0.2)
        else:
            signals["quorum_multiplier"] = min(1.2, signals["quorum_multiplier"] + 0.02)
            signals["recruitment_decay"] = min(0.2, signals["recruitment_decay"] + 0.005)
            signals["dance_gain"] = max(0.5, signals["dance_gain"] - 0.05)

        # === SEMANTIC TEMPERATURE ===
        # Higher temp = more exploration in knowledge graph
        if variance_issue:
            signals["semantic_temperature"] = min(3.0, signals["semantic_temperature"] + 0.1)
        else:
            signals["semantic_temperature"] = max(0.5, signals["semantic_temperature"] - 0.05)

        return signals

    def _smooth_signals(self, target: OvermindSignals) -> OvermindSignals:
        """
        Apply exponential smoothing to signal changes.

        Prevents sudden jumps that could destabilize training.
        """
        smoothed = {}
        for key in target:
            current = self._current_signals.get(key, target[key])
            if key == "comm_budget":
                # Integer smoothing
                smoothed[key] = int(
                    self.signal_smoothing * current +
                    (1 - self.signal_smoothing) * target[key]
                )
            else:
                smoothed[key] = (
                    self.signal_smoothing * current +
                    (1 - self.signal_smoothing) * target[key]
                )
        return smoothed

    def get_wisdom_signal(self) -> float:
        """
        Get current wisdom signal for agent observations.

        This is a READ-ONLY value broadcast to agents.
        Returns normalized wisdom in [-1, 1].
        """
        wisdom = self.wisdom_state.total_wisdom
        return float(np.clip(wisdom / 2.0, -1.0, 1.0))

    def get_signal_router(self) -> SignalRouter:
        """
        Get the signal router for consumers.

        Consumers should use router.get_* methods for read-only access.
        """
        return self.signal_router

    def get_current_signals(self) -> OvermindSignals:
        """Get copy of current signals (for logging)."""
        return self._current_signals.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get overmind statistics for logging"""
        stats = {
            "step_count": self.step_count,
            "current_wisdom": self.wisdom_state.total_wisdom,
            "avg_wisdom": float(np.mean(self.wisdom_history)) if self.wisdom_history else 0.0,
            "wisdom_trend": (
                self.wisdom_history[-1] - self.wisdom_history[0]
                if len(self.wisdom_history) >= 2 else 0.0
            ),
            "signals": self._current_signals.copy(),
            "water_variance": self.wisdom_state.water_variance,
            "flood_fraction": self.wisdom_state.flood_fraction,
            "colony_health": self.wisdom_state.colony_health,
            # PHASE 2: Info metrics
            "info_dissipation_rate": self.wisdom_state.info_dissipation_rate,
            "avg_agent_info_energy": self.wisdom_state.avg_agent_info_energy,
            "info_blocked_actions": self.wisdom_state.info_blocked_actions,
        }
        # Add emergency status
        stats.update(self.get_emergency_status())
        return stats

    def reset(self):
        """Reset overmind to initial state"""
        self._current_signals = DEFAULT_OVERMIND_SIGNALS.copy()
        self.signal_router = SignalRouter()
        self.wisdom_state = WisdomState()
        self.emergency_state = EmergencyState()
        self.observation_history.clear()
        self.wisdom_history.clear()
        self.entropy_history.clear()
        self.step_count = 0

    def update(self, observation: np.ndarray, env) -> None:
        """
        Update step: produce signals and apply to environment subsystems.

        This method bridges the contract-compliant signal production with
        environment subsystem configuration.

        Args:
            observation: Global observation array
            env: Environment instance to apply signals to
        """
        # Produce signals through the contract-compliant interface
        signals = self.produce_signals(observation)

        # Apply signals to pheromone field
        if hasattr(env, 'pheromone_field') and env.pheromone_field is not None:
            env.pheromone_field.config.evaporation_rate = signals["pheromone_decay"]

        # Apply signals to project manager
        if hasattr(env, 'project_manager') and env.project_manager is not None:
            env.project_manager.config.recruitment_decay = signals["recruitment_decay"]
            env.project_manager.config.dance_gain = signals["dance_gain"]

        # Apply signals to physarum network (adaptation rates)
        if hasattr(env, 'physarum_network') and env.physarum_network is not None:
            # tau and flow_exp would be used during physarum.update()
            pass

        # Apply signals to communication system
        if hasattr(env, 'communication_system') and env.communication_system is not None:
            env.communication_system.config.max_bandwidth_per_step = signals["comm_budget"]


class NeuralOvermind(Overmind):
    """
    Neural network-based overmind using learned policy.

    STILL RESPECTS THE CONTRACT:
    - Outputs only OvermindSignals
    - Cannot modify rewards or policies
    - Uses neural network to decide signal values
    """

    def __init__(self, config: OvermindConfig, sim_config: SimulationConfig):
        super().__init__(config, sim_config)

        # Neural network (lazy init)
        self.policy_network = None
        self.value_network = None
        self.optimizer = None

        # Training buffer
        self.experience_buffer: List[Tuple] = []
        self.max_buffer_size = 1000

        # Mode flag
        self.use_neural = False

    def initialize_networks(self):
        """Initialize neural networks for learning signal production"""
        try:
            import torch
            import torch.nn as nn

            class SignalNetwork(nn.Module):
                """Network that outputs OvermindSignals values"""
                def __init__(self, obs_dim: int, n_signals: int, hidden_dims: List[int]):
                    super().__init__()

                    layers = []
                    prev_dim = obs_dim
                    for h in hidden_dims:
                        layers.extend([
                            nn.Linear(prev_dim, h),
                            nn.ReLU(),
                            nn.LayerNorm(h),
                        ])
                        prev_dim = h

                    # Output in [-1, 1] for all signals
                    layers.append(nn.Linear(prev_dim, n_signals))
                    layers.append(nn.Tanh())

                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            class ValueNetwork(nn.Module):
                """Critic for signal quality estimation"""
                def __init__(self, obs_dim: int, hidden_dims: List[int]):
                    super().__init__()

                    layers = []
                    prev_dim = obs_dim
                    for h in hidden_dims:
                        layers.extend([
                            nn.Linear(prev_dim, h),
                            nn.ReLU(),
                        ])
                        prev_dim = h
                    layers.append(nn.Linear(prev_dim, 1))

                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            obs_dim = self.config.n_observation_features
            n_signals = len(DEFAULT_OVERMIND_SIGNALS)
            hidden_dims = self.config.hidden_dims

            self.policy_network = SignalNetwork(obs_dim, n_signals, hidden_dims)
            self.value_network = ValueNetwork(obs_dim, hidden_dims)

            params = list(self.policy_network.parameters()) + list(self.value_network.parameters())
            self.optimizer = torch.optim.Adam(params, lr=1e-4)

            self.use_neural = True

        except ImportError:
            self.use_neural = False

    def _decide_signals(self, observation: np.ndarray, wisdom: float) -> OvermindSignals:
        """Override to use neural network when available"""
        if self.use_neural and self.policy_network is not None:
            return self._neural_decide(observation)
        return super()._decide_signals(observation, wisdom)

    def _neural_decide(self, observation: np.ndarray) -> OvermindSignals:
        """Use neural network to produce signals"""
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            raw_output = self.policy_network(obs_tensor).squeeze(0).numpy()

        # Map [-1, 1] outputs to signal bounds
        signal_keys = list(DEFAULT_OVERMIND_SIGNALS.keys())
        signals = {}

        for i, key in enumerate(signal_keys):
            if i < len(raw_output):
                low, high = SIGNAL_BOUNDS[key]
                # Map from [-1, 1] to [low, high]
                normalized = (raw_output[i] + 1) / 2  # Now in [0, 1]
                value = low + normalized * (high - low)

                if key == "comm_budget":
                    signals[key] = int(value)
                else:
                    signals[key] = float(value)
            else:
                signals[key] = DEFAULT_OVERMIND_SIGNALS[key]

        return signals

    def store_experience(self, observation: np.ndarray, signals: OvermindSignals,
                         wisdom_delta: float, next_observation: np.ndarray,
                         done: bool):
        """
        Store experience for training.

        Note: We use wisdom_delta as reward proxy, NOT actual environment rewards.
        This maintains the contract that Overmind cannot see/modify rewards.
        """
        self.experience_buffer.append((
            observation, signals, wisdom_delta, next_observation, done
        ))

        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step using wisdom as reward signal"""
        if not self.use_neural or len(self.experience_buffer) < batch_size:
            return None

        import torch
        import torch.nn.functional as F

        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        obs_batch = torch.FloatTensor([e[0] for e in batch])
        # wisdom_delta serves as reward (NOT environment reward!)
        reward_batch = torch.FloatTensor([e[2] for e in batch])
        next_obs_batch = torch.FloatTensor([e[3] for e in batch])
        done_batch = torch.FloatTensor([float(e[4]) for e in batch])

        # Value targets
        with torch.no_grad():
            next_values = self.value_network(next_obs_batch).squeeze()
            targets = reward_batch + 0.99 * (1 - done_batch) * next_values

        # Value loss
        values = self.value_network(obs_batch).squeeze()
        value_loss = F.mse_loss(values, targets)

        # Policy loss
        advantages = (targets - values).detach()
        policy_out = self.policy_network(obs_batch)
        policy_loss = -(policy_out.mean(dim=1) * advantages).mean()

        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        return loss.item()


# =============================================================================
# LEGACY COMPATIBILITY - Adapter for old interface
# =============================================================================

def create_overmind_adapter(overmind: Overmind):
    """
    Creates an adapter that applies signals to environment subsystems.

    This is the ONLY place where signals are converted to subsystem changes.
    The adapter reads from SignalRouter and writes to subsystems.
    """

    class OvermindAdapter:
        """Adapter that applies Overmind signals to environment"""

        def __init__(self, overmind_instance: Overmind):
            self.overmind = overmind_instance

        def update(self, observation: np.ndarray, env) -> None:
            """
            Update step: produce signals and apply to environment.

            This is the bridge between the contract-compliant Overmind
            and the environment subsystems.
            """
            # Overmind produces signals
            signals = self.overmind.produce_signals(observation)

            # Apply to pheromone field
            if hasattr(env, 'pheromone_field') and env.pheromone_field is not None:
                env.pheromone_field.config.evaporation_rate = signals["pheromone_decay"]

            # Apply to project manager
            if hasattr(env, 'project_manager') and env.project_manager is not None:
                env.project_manager.config.recruitment_decay = signals["recruitment_decay"]
                env.project_manager.config.dance_gain = signals["dance_gain"]
                # Note: quorum_multiplier would need to be applied during checks

            # Apply to physarum network
            if hasattr(env, 'physarum_network') and env.physarum_network is not None:
                # tau and flow_exp would be used during physarum.update()
                pass

            # Apply to communication system
            if hasattr(env, 'communication_system') and env.communication_system is not None:
                env.communication_system.config.max_bandwidth_per_step = signals["comm_budget"]

        def get_wisdom_signal(self) -> float:
            return self.overmind.get_wisdom_signal()

        def get_statistics(self) -> Dict[str, Any]:
            return self.overmind.get_statistics()

        def reset(self):
            self.overmind.reset()

    return OvermindAdapter(overmind)
