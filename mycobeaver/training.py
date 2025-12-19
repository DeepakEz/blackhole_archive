"""
MycoBeaver Training Pipeline
=============================
PPO-based training pipeline for multi-agent reinforcement learning.

Based on MycoBeaver Simulator Design Plan Section 3:
- Proximal Policy Optimization (PPO)
- Generalized Advantage Estimation (GAE)
- Multi-agent parallel training
- Curriculum learning support
- Ablation study support
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import time
import json
import os
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import SimulationConfig, TrainingConfig, create_default_config
from .environment import MycoBeaverEnv
from .policy import MultiAgentPolicy, RolloutBuffer, check_torch
from .contracts import OvermindSignals, SignalRouter, DEFAULT_OVERMIND_SIGNALS


@dataclass
class TrainingMetrics:
    """Metrics tracked during training"""
    episode: int = 0
    total_steps: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0

    # Policy metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0

    # === PHASE 6: Enhanced PPO metrics ===
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_variance: float = 0.0  # Critical for stability monitoring
    value_pred_mean: float = 0.0
    value_target_mean: float = 0.0
    explained_variance: float = 0.0  # How well value function explains returns

    # KL early stopping metrics
    kl_early_stopped: bool = False
    epochs_completed: int = 0

    # Gradient metrics
    grad_norm: float = 0.0
    grad_norm_before_clip: float = 0.0

    # Entropy comparison (PHASE 6: policy vs semantic)
    policy_entropy: float = 0.0
    semantic_entropy: float = 0.0
    entropy_ratio: float = 0.0  # policy / semantic - shows exploration vs knowledge

    # Learning rate (after Overmind modulation)
    current_lr: float = 0.0
    current_entropy_coef: float = 0.0

    # Environment metrics
    n_alive_agents: int = 0
    avg_water_level: float = 0.0
    total_vegetation: float = 0.0
    n_structures: int = 0
    n_completed_projects: int = 0

    # Subsystem metrics
    avg_pheromone: float = 0.0
    avg_conductivity: float = 0.0
    semantic_coherence: float = 0.0
    wisdom_signal: float = 0.0

    # Info thermodynamics
    avg_info_energy: float = 0.0
    total_info_spent: float = 0.0


@dataclass
class TrainingHistory:
    """History of training metrics"""
    metrics: List[TrainingMetrics] = field(default_factory=list)

    def add(self, metrics: TrainingMetrics):
        self.metrics.append(metrics)

    def get_recent(self, n: int = 100) -> List[TrainingMetrics]:
        return self.metrics[-n:]

    def get_smoothed(self, key: str, window: int = 100) -> List[float]:
        values = [getattr(m, key) for m in self.metrics]
        if len(values) < window:
            return values

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(values[start:i+1]))
        return smoothed


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for MycoBeaver.

    Implements:
    - Clipped objective
    - Generalized Advantage Estimation
    - Multi-epoch updates
    - Gradient clipping
    - Learning rate scheduling

    PHASE 6 Enhancements:
    - KL-target early stopping
    - Value loss clipping
    - Advantage variance logging
    - Policy vs semantic entropy tracking
    - Full reproducibility with seed control
    """

    def __init__(self, config: SimulationConfig, device: str = "auto"):
        check_torch()

        self.config = config
        self.training_config = config.training

        # === PHASE 6: Reproducibility - Seed control ===
        self._setup_seeds()

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Deterministic mode (PHASE 6)
        if config.training.deterministic and TORCH_AVAILABLE:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
            print("Deterministic mode enabled (may be slower)")

        print(f"Training on device: {self.device}")

        # Create environment
        env_seed = config.training.env_seed or config.training.seed
        self.env = MycoBeaverEnv(config)

        # Create policy
        self.policy = MultiAgentPolicy(config, str(self.device))

        # Training state
        self.total_steps = 0
        self.episode_count = 0

        # History
        self.history = TrainingHistory()

        # Rollout buffers (one per agent)
        self.buffers: Dict[str, RolloutBuffer] = {}

        # Signal router for Overmind modulation
        # This allows Overmind to adjust lr_scale and entropy_scale
        self._signal_router: Optional[SignalRouter] = None
        self._base_lr = config.policy.learning_rate
        self._base_entropy_coef = config.policy.entropy_coef

        # === PHASE 6: Value normalization state ===
        self._value_running_mean = 0.0
        self._value_running_std = 1.0
        self._value_count = 0

        # Learning rate scheduler
        self.lr_scheduler = None
        if hasattr(torch.optim.lr_scheduler, 'CosineAnnealingLR'):
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.policy.optimizer,
                T_max=config.training.n_episodes,
                eta_min=config.policy.learning_rate * 0.1
            )

        # === PHASE 6: Config dumping ===
        if config.training.dump_config_on_start:
            self._dump_config()

    def _setup_seeds(self):
        """PHASE 6: Set up all random seeds for reproducibility"""
        tc = self.training_config
        main_seed = tc.seed

        # NumPy seed
        np_seed = tc.numpy_seed if tc.numpy_seed is not None else main_seed
        np.random.seed(np_seed)

        # PyTorch seed
        if TORCH_AVAILABLE:
            torch_seed = tc.torch_seed if tc.torch_seed is not None else main_seed
            torch.manual_seed(torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(torch_seed)

        print(f"Seeds initialized: main={main_seed}, numpy={np_seed}")

    def _dump_config(self):
        """PHASE 6: Dump full config to file for reproducibility"""
        from datetime import datetime
        import dataclasses

        config_dir = Path(self.training_config.config_dump_dir)
        config_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = config_dir / f"config_{timestamp}.json"

        # Convert config to dict (handle dataclasses)
        def config_to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: config_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [config_to_dict(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: config_to_dict(v) for k, v in obj.items()}
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            else:
                return obj

        config_dict = config_to_dict(self.config)

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        print(f"Config dumped to: {config_path}")

    def set_signal_router(self, router: SignalRouter) -> None:
        """
        Connect the trainer to the Overmind's signal router.

        This allows Overmind to modulate:
        - lr_scale: Multiplier for learning rate
        - entropy_scale: Multiplier for entropy coefficient

        Args:
            router: SignalRouter from Overmind
        """
        self._signal_router = router

    def _get_modulated_lr(self) -> float:
        """Get learning rate modulated by Overmind signal."""
        if self._signal_router is not None:
            return self._base_lr * self._signal_router.get_lr_scale()
        return self._base_lr

    def _get_modulated_entropy_coef(self) -> float:
        """Get entropy coefficient modulated by Overmind signal."""
        if self._signal_router is not None:
            return self._base_entropy_coef * self._signal_router.get_entropy_scale()
        return self._base_entropy_coef

    def _apply_lr_modulation(self) -> None:
        """Apply Overmind's learning rate modulation to optimizer."""
        modulated_lr = self._get_modulated_lr()
        for param_group in self.policy.optimizer.param_groups:
            param_group['lr'] = modulated_lr

    def collect_rollout(self, n_steps: int) -> Dict[str, float]:
        """
        Collect rollout data from environment.

        Args:
            n_steps: Number of steps to collect

        Returns:
            Rollout statistics
        """
        # Initialize buffers for each agent
        for i in range(self.config.n_beavers):
            agent_key = f"agent_{i}"
            if agent_key not in self.buffers:
                self.buffers[agent_key] = RolloutBuffer()

        # Reset if needed
        if self.env.current_step == 0:
            observations, info = self.env.reset()
        else:
            observations = self.env._get_observations()

        total_reward = 0.0
        episode_lengths = []
        current_episode_length = 0

        for step in range(n_steps):
            # Get actions and values
            with torch.no_grad():
                actions = self.policy.get_actions(observations)

                # Get values for each agent
                for agent_key, obs in observations.items():
                    local = torch.FloatTensor(obs["local_grid"]).unsqueeze(0).to(self.device)
                    glob = torch.FloatTensor(obs["global_features"]).unsqueeze(0).to(self.device)
                    internal = torch.FloatTensor(obs["internal_state"]).unsqueeze(0).to(self.device)

                    logits, value = self.policy.network(local, glob, internal)
                    probs = F.softmax(logits, dim=-1)

                    action = actions[agent_key]
                    log_prob = torch.log(probs[0, action] + 1e-8).item()

                    # Store in buffer
                    self.buffers[agent_key].add(
                        obs_local=obs["local_grid"],
                        obs_global=obs["global_features"],
                        obs_internal=obs["internal_state"],
                        action=action,
                        log_prob=log_prob,
                        reward=0.0,  # Will be filled after step
                        value=value.item(),
                        done=False
                    )

            # Step environment
            next_observations, rewards, terminated, truncated, info = self.env.step(actions)

            # Store rewards and dones
            for agent_key in actions.keys():
                if len(self.buffers[agent_key]) > 0:
                    self.buffers[agent_key].rewards[-1] = rewards.get(agent_key, 0.0)
                    self.buffers[agent_key].dones[-1] = terminated or truncated

            total_reward += sum(rewards.values())
            current_episode_length += 1
            self.total_steps += 1

            # Check episode end
            done = terminated or truncated
            if done:
                episode_lengths.append(current_episode_length)
                current_episode_length = 0
                self.episode_count += 1

                # Reset environment
                observations, info = self.env.reset()
            else:
                observations = next_observations

        return {
            "total_reward": total_reward,
            "avg_episode_length": np.mean(episode_lengths) if episode_lengths else current_episode_length,
            "n_episodes": len(episode_lengths),
        }

    def compute_advantages(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute returns and advantages for all agents.

        Returns:
            Dict mapping agent_key to (returns, advantages)
        """
        advantages_dict = {}

        for agent_key, buffer in self.buffers.items():
            if len(buffer) == 0:
                continue

            # Get last value estimate
            last_obs = {
                "local_grid": buffer.observations_local[-1],
                "global_features": buffer.observations_global[-1],
                "internal_state": buffer.observations_internal[-1],
            }

            with torch.no_grad():
                local = torch.FloatTensor(last_obs["local_grid"]).unsqueeze(0).to(self.device)
                glob = torch.FloatTensor(last_obs["global_features"]).unsqueeze(0).to(self.device)
                internal = torch.FloatTensor(last_obs["internal_state"]).unsqueeze(0).to(self.device)

                _, last_value = self.policy.network(local, glob, internal)
                last_value = last_value.item()

            # Compute advantages
            returns, advantages = buffer.compute_returns_and_advantages(
                last_value,
                gamma=self.config.policy.gamma,
                gae_lambda=self.config.policy.gae_lambda
            )

            advantages_dict[agent_key] = (returns, advantages)

        return advantages_dict

    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected rollouts.

        PHASE 6 Enhancements:
        - KL-target early stopping: Prevents catastrophic policy updates
        - Value loss clipping: PPO-style value function clipping
        - Gradient norm tracking: Before and after clipping
        - Advantage variance logging: Critical stability metric

        Returns:
            Training statistics with enhanced PHASE 6 metrics
        """
        # Compute advantages
        advantages_dict = self.compute_advantages()

        # Combine all agent data
        all_obs_local = []
        all_obs_global = []
        all_obs_internal = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        all_old_values = []  # PHASE 6: Track old values for clipping

        for agent_key, buffer in self.buffers.items():
            if agent_key not in advantages_dict:
                continue

            returns, advantages = advantages_dict[agent_key]

            local, glob, internal, actions, log_probs, _, values, _ = buffer.get()

            all_obs_local.append(local)
            all_obs_global.append(glob)
            all_obs_internal.append(internal)
            all_actions.append(actions)
            all_old_log_probs.append(log_probs)
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_old_values.append(values)

        if not all_obs_local:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Concatenate
        obs_local = torch.cat(all_obs_local).to(self.device)
        obs_global = torch.cat(all_obs_global).to(self.device)
        obs_internal = torch.cat(all_obs_internal).to(self.device)
        actions = torch.cat(all_actions).to(self.device)
        old_log_probs = torch.cat(all_old_log_probs).to(self.device)
        returns = torch.cat(all_returns).to(self.device)
        advantages_raw = torch.cat(all_advantages).to(self.device)
        old_values = torch.cat(all_old_values).to(self.device)

        # === PHASE 6: Track advantage statistics BEFORE normalization ===
        advantage_mean = advantages_raw.mean().item()
        advantage_std = advantages_raw.std().item()
        advantage_variance = advantage_std ** 2  # Critical stability metric

        # Normalize advantages
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        # === PHASE 6: Value target normalization ===
        if self.training_config.normalize_value_targets:
            # Update running statistics
            batch_mean = returns.mean().item()
            batch_std = returns.std().item()
            self._value_count += 1
            alpha = 1.0 / self._value_count
            self._value_running_mean = (1 - alpha) * self._value_running_mean + alpha * batch_mean
            self._value_running_std = (1 - alpha) * self._value_running_std + alpha * batch_std

            # Normalize returns for value function
            normalized_returns = (returns - self._value_running_mean) / (self._value_running_std + 1e-8)
        else:
            normalized_returns = returns

        # Apply Overmind learning rate modulation BEFORE update
        self._apply_lr_modulation()

        # Get modulated entropy coefficient from Overmind
        entropy_coef = self._get_modulated_entropy_coef()

        # PPO update
        batch_size = self.config.training.batch_size
        n_samples = len(returns)
        indices = np.arange(n_samples)

        # === PHASE 6: Enhanced metric tracking ===
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_grad_norm_before = 0.0
        total_grad_norm_after = 0.0
        total_value_pred = 0.0
        total_value_target = 0.0
        n_updates = 0

        kl_early_stopped = False
        epochs_completed = 0

        for epoch in range(self.config.training.n_epochs_per_update):
            np.random.shuffle(indices)
            epoch_kl = 0.0
            epoch_updates = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get batch
                batch_obs_local = obs_local[batch_indices]
                batch_obs_global = obs_global[batch_indices]
                batch_obs_internal = obs_internal[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = normalized_returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]

                # === PHASE 6: Per-batch advantage normalization (optional) ===
                if self.training_config.normalize_advantages_per_batch:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # Evaluate actions
                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_obs_local, batch_obs_global, batch_obs_internal,
                    batch_actions
                )

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clip_eps = self.config.policy.clip_epsilon

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # === PHASE 6: Value loss with clipping (PPO-style) ===
                value_clip_range = self.training_config.value_clip_range
                values_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -value_clip_range, value_clip_range
                )
                value_loss_unclipped = F.mse_loss(values, batch_returns, reduction='none')
                value_loss_clipped = F.mse_loss(values_clipped, batch_returns, reduction='none')
                # Take max to be conservative (pessimistic bound)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss (using Overmind-modulated entropy coefficient)
                loss = (
                    policy_loss +
                    self.config.policy.value_coef * value_loss +
                    entropy_coef * entropy_loss  # Modulated by Overmind signals
                )

                # Optimize
                self.policy.optimizer.zero_grad()
                loss.backward()

                # === PHASE 6: Track gradient norm BEFORE clipping ===
                grad_norm_before = 0.0
                for p in self.policy.network.parameters():
                    if p.grad is not None:
                        grad_norm_before += p.grad.data.norm(2).item() ** 2
                grad_norm_before = grad_norm_before ** 0.5

                # Gradient clipping
                grad_norm_after = nn.utils.clip_grad_norm_(
                    self.policy.network.parameters(),
                    self.config.policy.max_grad_norm
                )
                if isinstance(grad_norm_after, torch.Tensor):
                    grad_norm_after = grad_norm_after.item()

                self.policy.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                    clip_fraction = (torch.abs(ratio - 1) > clip_eps).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                total_grad_norm_before += grad_norm_before
                total_grad_norm_after += grad_norm_after
                total_value_pred += values.mean().item()
                total_value_target += batch_returns.mean().item()
                n_updates += 1
                epoch_kl += approx_kl
                epoch_updates += 1

            epochs_completed = epoch + 1

            # === PHASE 6: KL-target early stopping ===
            if self.training_config.kl_early_stop and epoch_updates > 0:
                avg_epoch_kl = epoch_kl / epoch_updates
                if avg_epoch_kl > self.training_config.kl_target:
                    kl_early_stopped = True
                    break  # Stop training - policy changed too much

        # Clear buffers
        for buffer in self.buffers.values():
            buffer.clear()

        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # === PHASE 6: Compute explained variance ===
        # How well does value function predict returns?
        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0

        # Get current learning rate
        current_lr = self.policy.optimizer.param_groups[0]['lr']

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "approx_kl": total_approx_kl / max(1, n_updates),
            "clip_fraction": total_clip_fraction / max(1, n_updates),
            # === PHASE 6: Enhanced metrics ===
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
            "advantage_variance": advantage_variance,
            "value_pred_mean": total_value_pred / max(1, n_updates),
            "value_target_mean": total_value_target / max(1, n_updates),
            "explained_variance": explained_var,
            "kl_early_stopped": kl_early_stopped,
            "epochs_completed": epochs_completed,
            "grad_norm_before_clip": total_grad_norm_before / max(1, n_updates),
            "grad_norm": total_grad_norm_after / max(1, n_updates),
            "current_lr": current_lr,
            "current_entropy_coef": entropy_coef,
        }

    def train(self, n_episodes: Optional[int] = None,
              callback: Optional[Callable[[TrainingMetrics], None]] = None) -> TrainingHistory:
        """
        Main training loop.

        Args:
            n_episodes: Number of episodes (default from config)
            callback: Optional callback for each episode

        Returns:
            Training history
        """
        if n_episodes is None:
            n_episodes = self.config.training.n_episodes

        rollout_steps = self.config.training.max_steps_per_episode

        print(f"Starting training for {n_episodes} episodes")
        print(f"Config: {self.config.n_beavers} agents, {self.config.grid.grid_size}x{self.config.grid.grid_size} grid")

        # Connect to Overmind's signal router if available
        # This allows Overmind to modulate lr_scale and entropy_scale
        if self.config.training.use_overmind and self.env.overmind is not None:
            router = self.env.overmind.get_signal_router()
            self.set_signal_router(router)
            print("Connected to Overmind signal router for lr/entropy modulation")

        start_time = time.time()

        for episode in range(n_episodes):
            # Collect rollout
            rollout_stats = self.collect_rollout(rollout_steps)

            # Update policy
            update_stats = self.update_policy()

            # Get environment metrics
            env_info = self.env._get_info()

            # Create metrics
            metrics = TrainingMetrics(
                episode=episode,
                total_steps=self.total_steps,
                episode_reward=rollout_stats["total_reward"],
                episode_length=int(rollout_stats["avg_episode_length"]),
                policy_loss=update_stats["policy_loss"],
                value_loss=update_stats["value_loss"],
                entropy=update_stats["entropy"],
                approx_kl=update_stats["approx_kl"],
                clip_fraction=update_stats["clip_fraction"],
                n_alive_agents=env_info["n_alive_agents"],
                avg_water_level=env_info["avg_water_level"],
                total_vegetation=env_info["total_vegetation"],
                n_structures=env_info["n_structures"],
                # === PHASE 6: Enhanced PPO metrics ===
                advantage_mean=update_stats.get("advantage_mean", 0.0),
                advantage_std=update_stats.get("advantage_std", 0.0),
                advantage_variance=update_stats.get("advantage_variance", 0.0),
                value_pred_mean=update_stats.get("value_pred_mean", 0.0),
                value_target_mean=update_stats.get("value_target_mean", 0.0),
                explained_variance=update_stats.get("explained_variance", 0.0),
                kl_early_stopped=update_stats.get("kl_early_stopped", False),
                epochs_completed=update_stats.get("epochs_completed", 0),
                grad_norm=update_stats.get("grad_norm", 0.0),
                grad_norm_before_clip=update_stats.get("grad_norm_before_clip", 0.0),
                policy_entropy=update_stats.get("entropy", 0.0),
                current_lr=update_stats.get("current_lr", 0.0),
                current_entropy_coef=update_stats.get("current_entropy_coef", 0.0),
            )

            # Get subsystem metrics
            if self.env.pheromone_field is not None:
                pheromone_stats = self.env.pheromone_field.get_statistics()
                metrics.avg_pheromone = pheromone_stats["mean_pheromone"]

            if self.env.physarum_network is not None:
                physarum_stats = self.env.physarum_network.get_statistics()
                metrics.avg_conductivity = physarum_stats["mean_conductivity"]

            if self.env.overmind is not None:
                metrics.wisdom_signal = self.env.overmind.get_wisdom_signal()

            self.history.add(metrics)

            # Callback
            if callback is not None:
                callback(metrics)

            # Logging
            if episode % self.config.training.log_every == 0:
                elapsed = time.time() - start_time
                # Basic metrics
                log_msg = (f"Episode {episode}/{n_episodes} | "
                           f"Reward: {metrics.episode_reward:.1f} | "
                           f"Policy Loss: {metrics.policy_loss:.4f} | "
                           f"Value Loss: {metrics.value_loss:.4f} | "
                           f"Agents Alive: {metrics.n_alive_agents} | "
                           f"Time: {elapsed:.1f}s")
                print(log_msg)
                # PHASE 6: Enhanced stability logging (every 10 log intervals)
                if episode % (self.config.training.log_every * 10) == 0:
                    kl_status = "STOPPED" if metrics.kl_early_stopped else f"{metrics.epochs_completed}"
                    print(f"  └─ Stability: KL={metrics.approx_kl:.4f} | "
                          f"Adv.Var={metrics.advantage_variance:.4f} | "
                          f"ExpVar={metrics.explained_variance:.3f} | "
                          f"GradNorm={metrics.grad_norm:.3f} | "
                          f"Epochs={kl_status}")

            # Checkpointing
            if episode % self.config.training.checkpoint_every == 0 and episode > 0:
                self.save_checkpoint(episode)

        # === Save final checkpoint (ensures last episode is always saved) ===
        final_episode = n_episodes - 1  # 0-indexed, so episode 999 for 1000 episodes
        # Only save if we haven't just saved (avoid duplicate)
        if final_episode % self.config.training.checkpoint_every != 0:
            self.save_checkpoint(final_episode)
        # Also save a "final" checkpoint for easy access
        self.save_checkpoint_final()

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Final reward: {self.history.metrics[-1].episode_reward:.1f}")

        return self.history

    def save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save policy
        policy_path = checkpoint_dir / f"policy_ep{episode}.pt"
        self.policy.save(str(policy_path))

        # Save metrics
        metrics_path = checkpoint_dir / f"metrics_ep{episode}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "episode": episode,
                "total_steps": self.total_steps,
                "recent_metrics": [
                    {
                        "episode": m.episode,
                        "reward": m.episode_reward,
                        "policy_loss": m.policy_loss,
                        "value_loss": m.value_loss,
                    }
                    for m in self.history.get_recent(100)
                ]
            }, f, indent=2)

        print(f"Saved checkpoint at episode {episode}")

    def save_checkpoint_final(self):
        """Save final training checkpoint with complete history"""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save policy as "final"
        policy_path = checkpoint_dir / "policy_final.pt"
        self.policy.save(str(policy_path))

        # Save complete training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                "total_episodes": len(self.history.metrics),
                "total_steps": self.total_steps,
                "final_reward": self.history.metrics[-1].episode_reward if self.history.metrics else 0,
                "final_survival": self.history.metrics[-1].n_alive_agents if self.history.metrics else 0,
                "metrics": [
                    {
                        "episode": m.episode,
                        "reward": m.episode_reward,
                        "policy_loss": m.policy_loss,
                        "value_loss": m.value_loss,
                        "entropy": m.entropy,
                        "approx_kl": m.approx_kl,
                        "n_alive_agents": m.n_alive_agents,
                        "n_structures": m.n_structures,
                        "avg_water_level": m.avg_water_level,
                        "total_vegetation": m.total_vegetation,
                        "explained_variance": m.explained_variance,
                        "advantage_variance": m.advantage_variance,
                    }
                    for m in self.history.metrics
                ]
            }, f, indent=2)

        print(f"Saved final checkpoint: {policy_path}")
        print(f"Saved training history: {history_path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        self.policy.load(path)
        print(f"Loaded checkpoint from {path}")


class AblationStudy:
    """
    Run ablation studies with different component configurations.

    Tests the contribution of each subsystem:
    - Pheromones
    - Physarum network
    - Projects (waggle dance)
    - Overmind
    """

    def __init__(self, base_config: SimulationConfig):
        self.base_config = base_config
        self.results: Dict[str, TrainingHistory] = {}

    def create_ablation_config(self, ablation: str) -> SimulationConfig:
        """Create config with specific ablation"""
        config = create_default_config()

        # Copy base settings
        config.grid = self.base_config.grid
        config.n_beavers = self.base_config.n_beavers
        config.training = self.base_config.training

        # Apply ablation
        if ablation == "no_pheromones":
            config.training.use_pheromones = False
        elif ablation == "no_physarum":
            config.training.use_physarum = False
        elif ablation == "no_projects":
            config.training.use_projects = False
        elif ablation == "no_overmind":
            config.training.use_overmind = False
        elif ablation == "baseline":
            config.training.use_pheromones = False
            config.training.use_physarum = False
            config.training.use_projects = False
            config.training.use_overmind = False
        elif ablation == "full":
            pass  # Keep all components

        return config

    def run(self, ablations: List[str], n_episodes: int = 1000) -> Dict[str, TrainingHistory]:
        """
        Run ablation study.

        Args:
            ablations: List of ablation names
            n_episodes: Episodes per ablation

        Returns:
            Dict mapping ablation name to training history
        """
        for ablation in ablations:
            print(f"\n{'='*50}")
            print(f"Running ablation: {ablation}")
            print(f"{'='*50}")

            config = self.create_ablation_config(ablation)
            config.training.n_episodes = n_episodes

            trainer = PPOTrainer(config)
            history = trainer.train()

            self.results[ablation] = history

        return self.results

    def summarize(self) -> Dict[str, Dict[str, float]]:
        """Summarize ablation results"""
        summary = {}

        for ablation, history in self.results.items():
            metrics = history.get_recent(100)

            summary[ablation] = {
                "final_reward": np.mean([m.episode_reward for m in metrics]),
                "final_survival": np.mean([m.n_alive_agents for m in metrics]),
                "avg_policy_loss": np.mean([m.policy_loss for m in metrics]),
            }

        return summary


def train_mycobeaver(
    config: Optional[SimulationConfig] = None,
    n_episodes: int = 1000,
    device: str = "auto",
    checkpoint_dir: str = "./checkpoints"
) -> TrainingHistory:
    """
    Convenience function to train MycoBeaver.

    Args:
        config: Configuration (default: create_default_config())
        n_episodes: Number of episodes
        device: Device to train on
        checkpoint_dir: Directory for checkpoints

    Returns:
        Training history
    """
    if config is None:
        config = create_default_config()

    config.training.n_episodes = n_episodes
    config.training.checkpoint_dir = checkpoint_dir

    trainer = PPOTrainer(config, device)
    history = trainer.train()

    return history
