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
    """

    def __init__(self, config: SimulationConfig, device: str = "auto"):
        check_torch()

        self.config = config
        self.training_config = config.training

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Training on device: {self.device}")

        # Create environment
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

        # Learning rate scheduler
        self.lr_scheduler = None
        if hasattr(torch.optim.lr_scheduler, 'CosineAnnealingLR'):
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.policy.optimizer,
                T_max=config.training.n_episodes,
                eta_min=config.policy.learning_rate * 0.1
            )

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

        Returns:
            Training statistics
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

        for agent_key, buffer in self.buffers.items():
            if agent_key not in advantages_dict:
                continue

            returns, advantages = advantages_dict[agent_key]

            local, glob, internal, actions, log_probs, _, _, _ = buffer.get()

            all_obs_local.append(local)
            all_obs_global.append(glob)
            all_obs_internal.append(internal)
            all_actions.append(actions)
            all_old_log_probs.append(log_probs)
            all_returns.append(returns)
            all_advantages.append(advantages)

        if not all_obs_local:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Concatenate
        obs_local = torch.cat(all_obs_local).to(self.device)
        obs_global = torch.cat(all_obs_global).to(self.device)
        obs_internal = torch.cat(all_obs_internal).to(self.device)
        actions = torch.cat(all_actions).to(self.device)
        old_log_probs = torch.cat(all_old_log_probs).to(self.device)
        returns = torch.cat(all_returns).to(self.device)
        advantages = torch.cat(all_advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        batch_size = self.config.training.batch_size
        n_samples = len(returns)
        indices = np.arange(n_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n_updates = 0

        for epoch in range(self.config.training.n_epochs_per_update):
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get batch
                batch_obs_local = obs_local[batch_indices]
                batch_obs_global = obs_global[batch_indices]
                batch_obs_internal = obs_internal[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

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

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.policy.value_coef * value_loss +
                    self.config.policy.entropy_coef * entropy_loss
                )

                # Optimize
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.policy.network.parameters(),
                    self.config.policy.max_grad_norm
                )

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
                n_updates += 1

        # Clear buffers
        for buffer in self.buffers.values():
            buffer.clear()

        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "approx_kl": total_approx_kl / max(1, n_updates),
            "clip_fraction": total_clip_fraction / max(1, n_updates),
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
                print(f"Episode {episode}/{n_episodes} | "
                      f"Reward: {metrics.episode_reward:.1f} | "
                      f"Policy Loss: {metrics.policy_loss:.4f} | "
                      f"Value Loss: {metrics.value_loss:.4f} | "
                      f"Agents Alive: {metrics.n_alive_agents} | "
                      f"Time: {elapsed:.1f}s")

            # Checkpointing
            if episode % self.config.training.checkpoint_every == 0 and episode > 0:
                self.save_checkpoint(episode)

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
