"""
MycoBeaver Overmind Meta-Controller
====================================
Meta-level controller that adjusts colony parameters based on global state.

Based on MycoBeaver Simulator Design Plan Section 2.5:
- Observes global colony state (water stability, agent health, project status)
- Computes "wisdom signal" from environmental indicators
- Adjusts pheromone evaporation, recruitment decay, dance gain, thresholds
- Uses simple policy network or rule-based control

Also incorporates Distributed Cognitive Architecture concepts:
- Information thermodynamics (semantic entropy)
- Coherence monitoring
- Adaptive meta-parameter control
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .config import OvermindConfig, SimulationConfig


@dataclass
class WisdomState:
    """Tracks wisdom signal components over time"""
    water_variance: float = 0.0
    flood_fraction: float = 0.0
    drought_fraction: float = 0.0
    colony_health: float = 1.0
    project_efficiency: float = 0.0
    habitat_quality: float = 0.0

    # Derived
    total_wisdom: float = 0.0

    # History for smoothing
    history_len: int = 10
    water_variance_history: List[float] = field(default_factory=list)
    flood_history: List[float] = field(default_factory=list)


class Overmind:
    """
    Meta-controller for the beaver colony.

    The Overmind observes the global state and adjusts colony parameters
    to improve overall performance. It acts on a slower timescale than
    individual agents.

    Key functions:
    1. Compute wisdom signal from environmental state
    2. Adjust pheromone evaporation rate ρ
    3. Adjust recruitment decay κ
    4. Adjust dance gain γ_dance
    5. Scale agent thresholds

    The wisdom signal indicates how well the colony is doing:
    - Positive: stable water, healthy agents, successful projects
    - Negative: floods, droughts, agent deaths, failed projects
    """

    def __init__(self, config: OvermindConfig, sim_config: SimulationConfig):
        self.config = config
        self.sim_config = sim_config

        # Current parameter values
        self.current_params = {
            "pheromone_evaporation": sim_config.pheromone.evaporation_rate,
            "recruitment_decay": sim_config.project.recruitment_decay,
            "dance_gain": sim_config.project.dance_gain,
            "threshold_scale": 1.0,  # Multiplier for agent thresholds
        }

        # Target parameter values (smoothly approach)
        self.target_params = self.current_params.copy()

        # Wisdom state
        self.wisdom_state = WisdomState()

        # Observation history
        self.observation_history: List[np.ndarray] = []
        self.wisdom_history: List[float] = []

        # Action history
        self.action_history: List[Dict] = []

        # Step counter
        self.step_count = 0

        # Learning rate for parameter adjustment
        self.learning_rate = 0.1
        self.param_smoothing = 0.9  # Exponential smoothing for parameters

    def update(self, observation: np.ndarray, env) -> Dict[str, float]:
        """
        Update overmind based on observation.

        Args:
            observation: Global state features
            env: Environment reference (for accessing subsystems)

        Returns:
            Dictionary of parameter adjustments
        """
        self.step_count += 1

        # Store observation
        self.observation_history.append(observation)
        if len(self.observation_history) > 100:
            self.observation_history.pop(0)

        # Compute wisdom signal
        wisdom = self._compute_wisdom(observation, env)
        self.wisdom_state.total_wisdom = wisdom
        self.wisdom_history.append(wisdom)
        if len(self.wisdom_history) > 100:
            self.wisdom_history.pop(0)

        # Decide on parameter adjustments
        adjustments = self._decide_adjustments(observation, wisdom, env)

        # Apply adjustments (smoothly)
        self._apply_adjustments(adjustments, env)

        return adjustments

    def _compute_wisdom(self, observation: np.ndarray, env) -> float:
        """
        Compute wisdom signal from global observations.

        W = -λ_σ * σ²(water) - λ_F * flood_frac - λ_D * drought_frac
            - λ_B * failures + λ_H * habitat_quality

        Args:
            observation: Global state features
            env: Environment reference

        Returns:
            Wisdom value (higher = better)
        """
        c = self.config

        # Extract features from observation
        # Assuming observation format from environment.py
        if len(observation) >= 9:
            water_mean = observation[0]
            water_variance = observation[1]
            water_max = observation[2]
            vegetation = observation[3]
            flood_fraction = observation[4]
            drought_fraction = observation[5]
            agent_survival = observation[6]
            avg_energy = observation[7]
            project_count = observation[8]
        else:
            # Fallback
            water_variance = 0.1
            flood_fraction = 0.1
            drought_fraction = 0.1
            agent_survival = 0.8
            vegetation = 0.5
            project_count = 0

        # Update wisdom state
        self.wisdom_state.water_variance = water_variance
        self.wisdom_state.flood_fraction = flood_fraction
        self.wisdom_state.drought_fraction = drought_fraction
        self.wisdom_state.colony_health = agent_survival

        # Track history for trend detection
        self.wisdom_state.water_variance_history.append(water_variance)
        self.wisdom_state.flood_history.append(flood_fraction)
        if len(self.wisdom_state.water_variance_history) > self.wisdom_state.history_len:
            self.wisdom_state.water_variance_history.pop(0)
            self.wisdom_state.flood_history.pop(0)

        # Compute failure indicator
        failure_rate = 1.0 - agent_survival

        # Compute habitat quality
        habitat_quality = vegetation * (1.0 - flood_fraction) * (1.0 - drought_fraction)
        self.wisdom_state.habitat_quality = habitat_quality

        # Wisdom calculation
        wisdom = (
            -c.water_variance_weight * water_variance -
            c.flood_weight * flood_fraction -
            c.drought_weight * drought_fraction -
            c.failure_weight * failure_rate +
            c.habitat_weight * habitat_quality
        )

        return wisdom

    def _decide_adjustments(self, observation: np.ndarray,
                            wisdom: float, env) -> Dict[str, float]:
        """
        Decide parameter adjustments based on wisdom signal.

        Uses rule-based control with smooth adjustments.

        Args:
            observation: Current observation
            wisdom: Current wisdom value
            env: Environment reference

        Returns:
            Dictionary of target parameter values
        """
        adjustments = {}

        # Get trends
        if len(self.wisdom_history) >= 2:
            wisdom_trend = self.wisdom_history[-1] - self.wisdom_history[-2]
        else:
            wisdom_trend = 0.0

        # Analyze specific issues
        flood_issue = self.wisdom_state.flood_fraction > 0.1
        drought_issue = self.wisdom_state.drought_fraction > 0.2
        health_issue = self.wisdom_state.colony_health < 0.7
        variance_issue = self.wisdom_state.water_variance > 0.2

        # Adjust pheromone evaporation
        # High variance or exploration needed -> higher evaporation
        # Stable, good paths -> lower evaporation
        if variance_issue or wisdom < -0.5:
            # Increase exploration
            target_evap = min(
                self.config.pheromone_evap_bounds[1],
                self.current_params["pheromone_evaporation"] + 0.02
            )
        elif wisdom > 0.5:
            # Exploit good paths
            target_evap = max(
                self.config.pheromone_evap_bounds[0],
                self.current_params["pheromone_evaporation"] - 0.02
            )
        else:
            target_evap = self.current_params["pheromone_evaporation"]

        adjustments["pheromone_evaporation"] = target_evap

        # Adjust recruitment decay
        # Crisis situations -> faster recruitment (lower decay)
        # Stable -> normal recruitment
        if flood_issue or drought_issue:
            target_decay = max(
                self.config.recruitment_decay_bounds[0],
                self.current_params["recruitment_decay"] - 0.02
            )
        elif wisdom > 0.3:
            target_decay = min(
                self.config.recruitment_decay_bounds[1],
                self.current_params["recruitment_decay"] + 0.01
            )
        else:
            target_decay = self.current_params["recruitment_decay"]

        adjustments["recruitment_decay"] = target_decay

        # Adjust dance gain
        # Need more coordination -> higher dance gain
        # Over-recruitment -> lower dance gain
        if health_issue or flood_issue:
            target_dance = min(
                self.config.dance_gain_bounds[1],
                self.current_params["dance_gain"] + 0.1
            )
        else:
            target_dance = self.current_params["dance_gain"]

        adjustments["dance_gain"] = target_dance

        # Adjust threshold scale
        # Lower thresholds = agents respond to weaker recruitment signals
        # Useful in crisis situations
        if health_issue:
            target_threshold = max(
                self.config.threshold_adjustment_bounds[0],
                self.current_params["threshold_scale"] - 0.05
            )
        elif wisdom > 0.5:
            target_threshold = min(
                self.config.threshold_adjustment_bounds[1],
                self.current_params["threshold_scale"] + 0.02
            )
        else:
            target_threshold = self.current_params["threshold_scale"]

        adjustments["threshold_scale"] = target_threshold

        return adjustments

    def _apply_adjustments(self, adjustments: Dict[str, float], env):
        """
        Apply parameter adjustments smoothly.

        Uses exponential smoothing to avoid sudden changes.

        Args:
            adjustments: Target parameter values
            env: Environment to update
        """
        for param, target in adjustments.items():
            # Smooth update
            current = self.current_params[param]
            new_value = (
                self.param_smoothing * current +
                (1 - self.param_smoothing) * target
            )
            self.current_params[param] = new_value

        # Apply to environment subsystems
        if hasattr(env, 'pheromone_field') and env.pheromone_field is not None:
            env.pheromone_field.config.evaporation_rate = \
                self.current_params["pheromone_evaporation"]

        if hasattr(env, 'project_manager') and env.project_manager is not None:
            env.project_manager.config.recruitment_decay = \
                self.current_params["recruitment_decay"]
            env.project_manager.config.dance_gain = \
                self.current_params["dance_gain"]

        # Store action
        self.action_history.append(self.current_params.copy())
        if len(self.action_history) > 100:
            self.action_history.pop(0)

    def get_wisdom_signal(self) -> float:
        """
        Get current wisdom signal for agent observations.

        Returns normalized wisdom in [-1, 1].
        """
        # Normalize to [-1, 1]
        wisdom = self.wisdom_state.total_wisdom
        return np.clip(wisdom / 2.0, -1.0, 1.0)

    def get_parameter_guidance(self, agent_role: str) -> Dict[str, float]:
        """
        Get parameter guidance for specific agent roles.

        Different roles may get different threshold adjustments.

        Args:
            agent_role: Agent's role (scout, worker, guardian)

        Returns:
            Role-specific parameter adjustments
        """
        base_threshold = self.current_params["threshold_scale"]

        if agent_role == "scout":
            # Scouts should be more responsive in exploration mode
            return {"threshold_scale": base_threshold * 0.9}
        elif agent_role == "worker":
            # Workers follow recruitment more closely
            return {"threshold_scale": base_threshold * 1.0}
        elif agent_role == "guardian":
            # Guardians are more conservative
            return {"threshold_scale": base_threshold * 1.1}

        return {"threshold_scale": base_threshold}

    def get_statistics(self) -> Dict[str, Any]:
        """Get overmind statistics"""
        return {
            "step_count": self.step_count,
            "current_wisdom": self.wisdom_state.total_wisdom,
            "avg_wisdom": np.mean(self.wisdom_history) if self.wisdom_history else 0.0,
            "wisdom_trend": (
                self.wisdom_history[-1] - self.wisdom_history[0]
                if len(self.wisdom_history) >= 2 else 0.0
            ),
            "params": self.current_params.copy(),
            "water_variance": self.wisdom_state.water_variance,
            "flood_fraction": self.wisdom_state.flood_fraction,
            "colony_health": self.wisdom_state.colony_health,
        }

    def get_action_recommendation(self, crisis_type: Optional[str] = None) -> Dict:
        """
        Get recommended actions for a specific crisis type.

        Used by external systems or for debugging.

        Args:
            crisis_type: Type of crisis (flood, drought, health, etc.)

        Returns:
            Recommended parameter values
        """
        if crisis_type == "flood":
            return {
                "message": "Prioritize dam building upstream",
                "pheromone_evaporation": 0.05,  # Strengthen existing paths
                "recruitment_decay": 0.05,  # Fast recruitment
                "dance_gain": 3.0,  # Strong advertising
            }
        elif crisis_type == "drought":
            return {
                "message": "Explore for water sources",
                "pheromone_evaporation": 0.3,  # More exploration
                "recruitment_decay": 0.15,
                "dance_gain": 1.5,
            }
        elif crisis_type == "health":
            return {
                "message": "Prioritize foraging and rest",
                "pheromone_evaporation": 0.15,
                "recruitment_decay": 0.1,
                "dance_gain": 2.0,
            }
        else:
            return {
                "message": "Normal operation",
                "pheromone_evaporation": 0.1,
                "recruitment_decay": 0.1,
                "dance_gain": 1.0,
            }

    def reset(self):
        """Reset overmind to initial state"""
        self.current_params = {
            "pheromone_evaporation": self.sim_config.pheromone.evaporation_rate,
            "recruitment_decay": self.sim_config.project.recruitment_decay,
            "dance_gain": self.sim_config.project.dance_gain,
            "threshold_scale": 1.0,
        }
        self.target_params = self.current_params.copy()
        self.wisdom_state = WisdomState()
        self.observation_history.clear()
        self.wisdom_history.clear()
        self.action_history.clear()
        self.step_count = 0


class NeuralOvermind(Overmind):
    """
    Neural network-based overmind using learned policy.

    Extends the rule-based Overmind with a trainable neural network
    that learns to adjust parameters based on experience.
    """

    def __init__(self, config: OvermindConfig, sim_config: SimulationConfig):
        super().__init__(config, sim_config)

        # Neural network will be initialized lazily (requires torch)
        self.policy_network = None
        self.value_network = None
        self.optimizer = None

        # Training buffer
        self.experience_buffer: List[Tuple] = []
        self.max_buffer_size = 1000

        # Whether to use neural policy (can fall back to rules)
        self.use_neural = False

    def initialize_networks(self):
        """Initialize neural networks for learning"""
        try:
            import torch
            import torch.nn as nn

            class OvermindPolicy(nn.Module):
                def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
                    super().__init__()

                    layers = []
                    prev_dim = obs_dim
                    for h in hidden_dims:
                        layers.extend([
                            nn.Linear(prev_dim, h),
                            nn.ReLU(),
                        ])
                        prev_dim = h

                    layers.append(nn.Linear(prev_dim, action_dim))
                    layers.append(nn.Tanh())  # Output in [-1, 1]

                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            class OvermindValue(nn.Module):
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
            action_dim = self.config.n_actions
            hidden_dims = self.config.hidden_dims

            self.policy_network = OvermindPolicy(obs_dim, action_dim, hidden_dims)
            self.value_network = OvermindValue(obs_dim, hidden_dims)

            params = list(self.policy_network.parameters()) + list(self.value_network.parameters())
            self.optimizer = torch.optim.Adam(params, lr=1e-4)

            self.use_neural = True

        except ImportError:
            # PyTorch not available, fall back to rule-based
            self.use_neural = False

    def _decide_adjustments(self, observation: np.ndarray,
                            wisdom: float, env) -> Dict[str, float]:
        """Override to use neural policy when available"""
        if self.use_neural and self.policy_network is not None:
            return self._neural_decide(observation)
        else:
            return super()._decide_adjustments(observation, wisdom, env)

    def _neural_decide(self, observation: np.ndarray) -> Dict[str, float]:
        """Use neural network to decide parameter adjustments"""
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action = self.policy_network(obs_tensor).squeeze(0).numpy()

        # Map network output [-1, 1] to parameter bounds
        adjustments = {}

        # Pheromone evaporation
        evap_bounds = self.config.pheromone_evap_bounds
        adjustments["pheromone_evaporation"] = (
            evap_bounds[0] + (action[0] + 1) / 2 * (evap_bounds[1] - evap_bounds[0])
        )

        # Recruitment decay
        decay_bounds = self.config.recruitment_decay_bounds
        adjustments["recruitment_decay"] = (
            decay_bounds[0] + (action[1] + 1) / 2 * (decay_bounds[1] - decay_bounds[0])
        )

        # Dance gain
        dance_bounds = self.config.dance_gain_bounds
        adjustments["dance_gain"] = (
            dance_bounds[0] + (action[2] + 1) / 2 * (dance_bounds[1] - dance_bounds[0])
        )

        # Threshold scale
        thresh_bounds = self.config.threshold_adjustment_bounds
        adjustments["threshold_scale"] = (
            thresh_bounds[0] + (action[3] + 1) / 2 * (thresh_bounds[1] - thresh_bounds[0])
        )

        return adjustments

    def store_experience(self, observation: np.ndarray, action: Dict,
                         reward: float, next_observation: np.ndarray,
                         done: bool):
        """Store experience for training"""
        self.experience_buffer.append((
            observation, action, reward, next_observation, done
        ))

        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step"""
        if not self.use_neural or len(self.experience_buffer) < batch_size:
            return None

        import torch
        import torch.nn.functional as F

        # Sample batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        obs_batch = torch.FloatTensor([e[0] for e in batch])
        reward_batch = torch.FloatTensor([e[2] for e in batch])
        next_obs_batch = torch.FloatTensor([e[3] for e in batch])
        done_batch = torch.FloatTensor([float(e[4]) for e in batch])

        # Compute value targets
        with torch.no_grad():
            next_values = self.value_network(next_obs_batch).squeeze()
            targets = reward_batch + 0.99 * (1 - done_batch) * next_values

        # Value loss
        values = self.value_network(obs_batch).squeeze()
        value_loss = F.mse_loss(values, targets)

        # Policy loss (using advantage)
        advantages = (targets - values).detach()
        actions = self.policy_network(obs_batch)
        policy_loss = -(actions.mean(dim=1) * advantages).mean()

        # Total loss
        loss = value_loss + policy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
