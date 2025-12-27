"""
MycoBeaver Neural Network Policies
===================================
PyTorch-based policy and value networks for multi-agent RL.

Based on MycoBeaver Simulator Design Plan Section 3:
- CNN for local grid observation processing
- Shared policy network across agents
- PPO-compatible actor-critic architecture
- Attention mechanisms for global information
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy classes for type hints when torch is not available
    class nn:
        class Module:
            pass

from .config import PolicyNetworkConfig, SimulationConfig


def check_torch():
    """Check if PyTorch is available"""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for neural network policies. "
            "Install with: pip install torch"
        )


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        check_torch()

        self.conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LocalEncoder(nn.Module):
    """
    CNN encoder for local grid observations.

    Takes (batch, channels, height, width) tensor and outputs
    flattened feature vector.
    """

    def __init__(self, config: PolicyNetworkConfig):
        super().__init__()
        check_torch()

        self.config = config
        view_size = 2 * config.local_view_radius + 1

        # Build convolutional layers
        layers = []
        in_channels = config.n_local_channels

        for out_channels in config.conv_channels:
            layers.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate output size
        # Each conv with stride=1, padding=1 preserves spatial size
        # So output is (batch, final_channels, view_size, view_size)
        conv_output_size = config.conv_channels[-1] * view_size * view_size

        # Reduce to feature vector
        self.fc = nn.Linear(conv_output_size, config.fc_hidden_dims[0])

    def forward(self, local_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_grid: (batch, channels, height, width)

        Returns:
            (batch, hidden_dim) feature vector
        """
        x = self.conv_layers(local_grid)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        return x


class GlobalEncoder(nn.Module):
    """
    Encoder for global features (recruitment signals, colony state).

    Uses attention to weight different global features.
    """

    def __init__(self, config: PolicyNetworkConfig):
        super().__init__()
        check_torch()

        self.config = config

        # MLP for global features
        self.fc1 = nn.Linear(config.n_global_features, config.fc_hidden_dims[0] // 2)
        self.fc2 = nn.Linear(config.fc_hidden_dims[0] // 2, config.fc_hidden_dims[0] // 4)

        # Attention weights
        self.attention = nn.Linear(config.n_global_features, config.n_global_features)

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_features: (batch, n_global_features)

        Returns:
            (batch, hidden_dim) feature vector
        """
        # Compute attention weights
        attn_weights = F.softmax(self.attention(global_features), dim=-1)

        # Apply attention
        weighted = global_features * attn_weights

        # MLP
        x = F.relu(self.fc1(weighted))
        x = F.relu(self.fc2(x))

        return x


class InternalEncoder(nn.Module):
    """
    Encoder for agent internal state (energy, satiety, role, etc.)
    """

    def __init__(self, config: PolicyNetworkConfig):
        super().__init__()
        check_torch()

        self.fc = nn.Linear(config.n_internal_features, config.fc_hidden_dims[0] // 4)

    def forward(self, internal_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            internal_state: (batch, n_internal_features)

        Returns:
            (batch, hidden_dim) feature vector
        """
        return F.relu(self.fc(internal_state))


class PolicyNetwork(nn.Module):
    """
    Actor network for policy.

    Combines local, global, and internal features to produce
    action probabilities.
    """

    def __init__(self, config: PolicyNetworkConfig):
        super().__init__()
        check_torch()

        self.config = config

        # Encoders
        self.local_encoder = LocalEncoder(config)
        self.global_encoder = GlobalEncoder(config)
        self.internal_encoder = InternalEncoder(config)

        # Combined dimension
        combined_dim = (
            config.fc_hidden_dims[0] +  # Local
            config.fc_hidden_dims[0] // 4 +  # Global
            config.fc_hidden_dims[0] // 4   # Internal
        )

        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(combined_dim, config.fc_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.fc_hidden_dims[1], config.fc_hidden_dims[1] // 2),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Linear(config.fc_hidden_dims[1] // 2, config.n_actions)

    def forward(self, local_grid: torch.Tensor,
                global_features: torch.Tensor,
                internal_state: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits.

        Args:
            local_grid: (batch, channels, height, width)
            global_features: (batch, n_global_features)
            internal_state: (batch, n_internal_features)

        Returns:
            (batch, n_actions) action logits
        """
        # Encode each input
        local_feat = self.local_encoder(local_grid)
        global_feat = self.global_encoder(global_features)
        internal_feat = self.internal_encoder(internal_state)

        # Concatenate
        combined = torch.cat([local_feat, global_feat, internal_feat], dim=-1)

        # Hidden layers
        hidden = self.hidden_layers(combined)

        # Action logits
        logits = self.action_head(hidden)

        return logits

    def get_action(self, local_grid: torch.Tensor,
                   global_features: torch.Tensor,
                   internal_state: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            local_grid: (batch, channels, height, width)
            global_features: (batch, n_global_features)
            internal_state: (batch, n_internal_features)
            deterministic: If True, return argmax action

        Returns:
            (actions, log_probs)
        """
        logits = self.forward(local_grid, global_features, internal_state)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            actions = torch.argmax(probs, dim=-1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
        else:
            dist = Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions, log_probs


class ValueNetwork(nn.Module):
    """
    Critic network for value estimation.

    Similar architecture to policy but outputs single value.
    """

    def __init__(self, config: PolicyNetworkConfig):
        super().__init__()
        check_torch()

        self.config = config

        # Encoders
        self.local_encoder = LocalEncoder(config)
        self.global_encoder = GlobalEncoder(config)
        self.internal_encoder = InternalEncoder(config)

        # Combined dimension
        combined_dim = (
            config.fc_hidden_dims[0] +
            config.fc_hidden_dims[0] // 4 +
            config.fc_hidden_dims[0] // 4
        )

        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(combined_dim, config.fc_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.fc_hidden_dims[1], config.fc_hidden_dims[1] // 2),
            nn.ReLU(),
        )

        # Value head
        self.value_head = nn.Linear(config.fc_hidden_dims[1] // 2, 1)

    def forward(self, local_grid: torch.Tensor,
                global_features: torch.Tensor,
                internal_state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Returns:
            (batch, 1) value estimates
        """
        local_feat = self.local_encoder(local_grid)
        global_feat = self.global_encoder(global_features)
        internal_feat = self.internal_encoder(internal_state)

        combined = torch.cat([local_feat, global_feat, internal_feat], dim=-1)
        hidden = self.hidden_layers(combined)
        value = self.value_head(hidden)

        return value


class ActorCritic(nn.Module):
    """
    Combined actor-critic network.

    Shares feature extraction between policy and value networks
    for more efficient training.
    """

    def __init__(self, config: PolicyNetworkConfig):
        super().__init__()
        check_torch()

        self.config = config

        # Shared encoders
        self.local_encoder = LocalEncoder(config)
        self.global_encoder = GlobalEncoder(config)
        self.internal_encoder = InternalEncoder(config)

        # Combined dimension
        combined_dim = (
            config.fc_hidden_dims[0] +
            config.fc_hidden_dims[0] // 4 +
            config.fc_hidden_dims[0] // 4
        )

        # Shared hidden layer
        self.shared_hidden = nn.Sequential(
            nn.Linear(combined_dim, config.fc_hidden_dims[1]),
            nn.ReLU(),
        )

        # Separate heads
        head_input_dim = config.fc_hidden_dims[1]

        self.policy_hidden = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(head_input_dim // 2, config.n_actions)

        self.value_hidden = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(head_input_dim // 2, 1)

    def _encode(self, local_grid: torch.Tensor,
                global_features: torch.Tensor,
                internal_state: torch.Tensor) -> torch.Tensor:
        """Encode observations through shared layers"""
        local_feat = self.local_encoder(local_grid)
        global_feat = self.global_encoder(global_features)
        internal_feat = self.internal_encoder(internal_state)

        combined = torch.cat([local_feat, global_feat, internal_feat], dim=-1)
        hidden = self.shared_hidden(combined)

        return hidden

    def forward(self, local_grid: torch.Tensor,
                global_features: torch.Tensor,
                internal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy logits and value.

        Returns:
            (action_logits, value)
        """
        shared = self._encode(local_grid, global_features, internal_state)

        policy_feat = self.policy_hidden(shared)
        action_logits = self.policy_head(policy_feat)

        value_feat = self.value_hidden(shared)
        value = self.value_head(value_feat)

        return action_logits, value

    def get_action_and_value(
        self,
        local_grid: torch.Tensor,
        global_features: torch.Tensor,
        internal_state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            local_grid: Local grid observations
            global_features: Global features
            internal_state: Internal state
            action: If provided, compute log prob for this action
            deterministic: Whether to sample or take argmax

        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value = self.forward(local_grid, global_features, internal_state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def compute_expected_free_energy(
        self,
        local_grid: torch.Tensor,
        global_features: torch.Tensor,
        internal_state: torch.Tensor,
        efe_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute Expected Free Energy (EFE) bias for action selection.

        Framework §10: G(u) = Epistemic + Pragmatic value

        EFE decomposes into:
        - Epistemic: Actions that reduce uncertainty (explore unknown)
        - Pragmatic: Actions that satisfy preferences (exploit known good)

        This method computes EFE for each action and returns a bias to
        add to action logits before softmax.

        Args:
            local_grid: (batch, channels, height, width)
            global_features: (batch, n_global_features)
            internal_state: (batch, n_internal_features)
            efe_weight: How much to weight EFE vs policy logits

        Returns:
            (batch, n_actions) EFE bias to subtract from logits
            (lower EFE = more preferred action)
        """
        batch_size = local_grid.shape[0]
        n_actions = self.config.n_actions

        # Get current value and logits
        logits, value = self.forward(local_grid, global_features, internal_state)

        # === EPISTEMIC VALUE ===
        # Approximate with observation entropy: actions that lead to
        # diverse observations reduce uncertainty
        # Use coordination field (channel 6) gradient as proxy
        coord_field = local_grid[:, 6, :, :]  # (batch, h, w)
        field_var = coord_field.var(dim=(1, 2))  # Variance per sample

        # High variance in coordination field = uncertain about best direction
        # Actions toward high-Ψ areas have epistemic value
        epistemic_per_sample = field_var.unsqueeze(1)  # (batch, 1)

        # === PRAGMATIC VALUE ===
        # Use value function: higher predicted value = lower pragmatic EFE
        # Normalize value to [0, 1] range for stability
        value_normalized = torch.sigmoid(value)  # (batch, 1)
        pragmatic_per_sample = -value_normalized  # Negative: high value = low EFE

        # === ACTION-SPECIFIC EFE ===
        # Different actions have different expected outcomes
        # Movement actions (0-4): epistemic (explore)
        # Build actions (8-11): pragmatic (construct)
        # Forage actions (5-7): survival

        action_type_bias = torch.zeros(batch_size, n_actions, device=local_grid.device)

        # Energy-based action preferences from internal state
        # internal_state[0] = energy/initial_energy
        energy_ratio = internal_state[:, 0:1]  # (batch, 1)

        # Low energy -> prefer foraging (survival)
        # High energy -> prefer building (pragmatic)
        # Medium energy -> prefer exploring (epistemic)
        forage_bias = (1.0 - energy_ratio) * 0.5  # Stronger when low energy
        build_bias = energy_ratio * 0.3  # Stronger when high energy
        explore_bias = 0.2 * torch.ones_like(energy_ratio)  # Constant exploration

        # Apply to action categories
        # Actions: 0=stay, 1-4=move, 5=forage, 6=gather, 7=ping_resource,
        #          8=build_dam, 9=repair_dam, 10=advertise, 11=recruit, etc.
        if n_actions > 5:
            action_type_bias[:, 5:8] = -forage_bias.expand(-1, 3)  # Forage: lower EFE when hungry
        if n_actions > 8:
            action_type_bias[:, 8:12] = -build_bias.expand(-1, min(4, n_actions-8))  # Build: lower EFE when energized

        # Movement gets epistemic bonus
        action_type_bias[:, 1:5] = -explore_bias.expand(-1, 4) * epistemic_per_sample

        # Combine epistemic and pragmatic
        # EFE = epistemic_term + pragmatic_term + action_bias
        efe = (
            epistemic_per_sample.expand(-1, n_actions) * 0.3 +
            pragmatic_per_sample.expand(-1, n_actions) * 0.7 +
            action_type_bias
        )

        return efe * efe_weight

    def get_action_with_efe(
        self,
        local_grid: torch.Tensor,
        global_features: torch.Tensor,
        internal_state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        use_efe: bool = True,
        efe_weight: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action with Expected Free Energy biasing.

        Combines PPO policy with active inference EFE for action selection.

        Args:
            local_grid: Local grid observations
            global_features: Global features
            internal_state: Internal state
            action: If provided, compute log prob for this action
            deterministic: Whether to sample or take argmax
            use_efe: Whether to apply EFE bias
            efe_weight: Weight for EFE bias

        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value = self.forward(local_grid, global_features, internal_state)

        # Apply EFE bias to logits
        if use_efe:
            efe_bias = self.compute_expected_free_energy(
                local_grid, global_features, internal_state, efe_weight
            )
            # Subtract EFE (lower EFE = higher preference)
            logits = logits - efe_bias

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class MultiAgentPolicy:
    """
    Multi-agent policy manager.

    Handles batched inference for multiple agents with shared policy.
    """

    def __init__(self, config: SimulationConfig, device: str = "cpu"):
        check_torch()

        self.config = config
        self.device = torch.device(device)

        # Shared actor-critic network
        self.network = ActorCritic(config.policy).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.policy.learning_rate
        )

    def get_actions(self, observations: Dict[str, Dict[str, np.ndarray]],
                    deterministic: bool = False,
                    use_efe: bool = True,
                    efe_weight: float = 0.1) -> Dict[str, int]:
        """
        Get actions for all agents using EFE-biased action selection.

        Framework §10: Combines PPO policy with Expected Free Energy for
        active inference-style planning.

        Args:
            observations: Dict mapping agent_id -> observation dict
            deterministic: Whether to use deterministic actions
            use_efe: Whether to apply Expected Free Energy bias
            efe_weight: Weight for EFE bias (0 = pure PPO, higher = more EFE)

        Returns:
            Dict mapping agent_id -> action
        """
        # Collect observations
        agent_ids = list(observations.keys())
        n_agents = len(agent_ids)

        local_grids = []
        global_feats = []
        internal_states = []

        for agent_id in agent_ids:
            obs = observations[agent_id]
            local_grids.append(obs["local_grid"])
            global_feats.append(obs["global_features"])
            internal_states.append(obs["internal_state"])

        # Stack into batches
        local_batch = torch.FloatTensor(np.stack(local_grids)).to(self.device)
        global_batch = torch.FloatTensor(np.stack(global_feats)).to(self.device)
        internal_batch = torch.FloatTensor(np.stack(internal_states)).to(self.device)

        # Get actions with EFE biasing
        with torch.no_grad():
            actions, _, _, _ = self.network.get_action_with_efe(
                local_batch, global_batch, internal_batch,
                deterministic=deterministic,
                use_efe=use_efe,
                efe_weight=efe_weight
            )

        # Convert to dict
        actions_np = actions.cpu().numpy()
        return {agent_ids[i]: int(actions_np[i]) for i in range(n_agents)}

    def evaluate_actions(
        self,
        local_grid: torch.Tensor,
        global_features: torch.Tensor,
        internal_state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Returns:
            (log_probs, entropy, values)
        """
        _, log_probs, entropy, values = self.network.get_action_and_value(
            local_grid, global_features, internal_state, action=actions
        )
        return log_probs, entropy, values

    def save(self, path: str):
        """Save policy to file"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load policy from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


@dataclass
class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO training.
    """
    observations_local: List[np.ndarray] = None
    observations_global: List[np.ndarray] = None
    observations_internal: List[np.ndarray] = None
    actions: List[int] = None
    log_probs: List[float] = None
    rewards: List[float] = None
    values: List[float] = None
    dones: List[bool] = None

    def __post_init__(self):
        self.observations_local = []
        self.observations_global = []
        self.observations_internal = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs_local: np.ndarray, obs_global: np.ndarray,
            obs_internal: np.ndarray, action: int, log_prob: float,
            reward: float, value: float, done: bool):
        """Add a transition to buffer"""
        self.observations_local.append(obs_local)
        self.observations_global.append(obs_global)
        self.observations_internal.append(obs_internal)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self) -> Tuple[torch.Tensor, ...]:
        """Get all data as tensors"""
        check_torch()

        return (
            torch.FloatTensor(np.stack(self.observations_local)),
            torch.FloatTensor(np.stack(self.observations_global)),
            torch.FloatTensor(np.stack(self.observations_internal)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.BoolTensor(self.dones),
        )

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and GAE advantages.

        Args:
            last_value: Value estimate for final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            (returns, advantages)
        """
        check_torch()

        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])

        # GAE computation
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + np.array(self.values)

        return torch.FloatTensor(returns), torch.FloatTensor(advantages)

    def clear(self):
        """Clear the buffer"""
        self.observations_local.clear()
        self.observations_global.clear()
        self.observations_internal.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)
