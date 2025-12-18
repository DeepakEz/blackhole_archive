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

        # Observation history (for trend detection)
        self.observation_history: List[np.ndarray] = []
        self.wisdom_history: List[float] = []

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

    def _decide_signals(self, observation: np.ndarray, wisdom: float) -> OvermindSignals:
        """
        Decide signal values based on wisdom and observations.

        Uses rule-based control to determine modulation factors.
        This does NOT touch rewards or policies - only rates and budgets.
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
        return {
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

    def reset(self):
        """Reset overmind to initial state"""
        self._current_signals = DEFAULT_OVERMIND_SIGNALS.copy()
        self.signal_router = SignalRouter()
        self.wisdom_state = WisdomState()
        self.observation_history.clear()
        self.wisdom_history.clear()
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
