"""
MycoBeaver Main Simulation Runner
==================================
Entry point for running MycoBeaver simulations and experiments.

Provides:
- CLI interface for training and evaluation
- Visualization utilities
- Experiment configurations
- Benchmark scenarios
"""

import argparse
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time

from .config import (
    SimulationConfig, create_default_config,
    GridConfig, AgentConfig, PheromoneConfig, ProjectConfig,
    PhysarumConfig, OvermindConfig, TrainingConfig
)
from .environment import MycoBeaverEnv


def create_benchmark_config(scenario: str = "standard") -> SimulationConfig:
    """
    Create configuration for benchmark scenarios.

    Args:
        scenario: One of "standard", "small", "large", "flood", "drought"

    Returns:
        SimulationConfig for the scenario
    """
    config = create_default_config()

    if scenario == "small":
        # Small quick test
        config.grid.grid_size = 32
        config.n_beavers = 5
        config.training.n_episodes = 100
        config.training.max_steps_per_episode = 200

    elif scenario == "standard":
        # Standard benchmark
        config.grid.grid_size = 64
        config.n_beavers = 10
        config.training.n_episodes = 1000
        config.training.max_steps_per_episode = 500

    elif scenario == "large":
        # Large scale
        config.grid.grid_size = 128
        config.n_beavers = 20
        config.training.n_episodes = 2000
        config.training.max_steps_per_episode = 1000

    elif scenario == "flood":
        # Flood challenge scenario
        config.grid.grid_size = 64
        config.grid.rainfall_rate = 0.1  # High rainfall
        config.grid.rainfall_variance = 0.05
        config.n_beavers = 15
        config.initial_water_level = 0.3  # Start with water

    elif scenario == "drought":
        # Drought challenge scenario
        config.grid.grid_size = 64
        config.grid.rainfall_rate = 0.001  # Very low rainfall
        config.grid.evaporation_rate = 0.05  # High evaporation
        config.n_beavers = 10
        config.initial_water_level = 0.05

    return config


def run_simulation(
    config: Optional[SimulationConfig] = None,
    n_steps: int = 1000,
    render: bool = False,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a simulation with random or trained policy.

    Args:
        config: Simulation configuration
        n_steps: Number of steps to run
        render: Whether to render frames
        seed: Random seed

    Returns:
        Simulation results dictionary
    """
    if config is None:
        config = create_default_config()

    env = MycoBeaverEnv(config, render_mode="rgb_array" if render else None)
    observations, info = env.reset(seed=seed)

    results = {
        "steps": [],
        "rewards": [],
        "agent_counts": [],
        "water_levels": [],
        "vegetation": [],
        "structures": [],
        "frames": [] if render else None,
    }

    total_reward = 0.0

    for step in range(n_steps):
        # Random actions for baseline
        actions = {}
        for i in range(config.n_beavers):
            actions[f"agent_{i}"] = np.random.randint(0, config.policy.n_actions)

        observations, rewards, terminated, truncated, info = env.step(actions)

        # Track metrics
        step_reward = sum(rewards.values())
        total_reward += step_reward

        results["steps"].append(step)
        results["rewards"].append(step_reward)
        results["agent_counts"].append(info["n_alive_agents"])
        results["water_levels"].append(info["avg_water_level"])
        results["vegetation"].append(info["total_vegetation"])
        results["structures"].append(info["n_structures"])

        if render:
            frame = env.render()
            if frame is not None:
                results["frames"].append(frame)

        if terminated or truncated:
            observations, info = env.reset()

    results["total_reward"] = total_reward
    results["final_info"] = info

    env.close()

    return results


def evaluate_policy(
    policy_path: str,
    config: Optional[SimulationConfig] = None,
    n_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate a trained policy.

    Args:
        policy_path: Path to saved policy
        config: Simulation configuration
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions

    Returns:
        Evaluation metrics
    """
    try:
        from .policy import MultiAgentPolicy
        from .training import PPOTrainer
    except ImportError as e:
        print(f"PyTorch required for policy evaluation: {e}")
        return {}

    if config is None:
        config = create_default_config()

    policy = MultiAgentPolicy(config)
    policy.load(policy_path)

    env = MycoBeaverEnv(config)

    episode_rewards = []
    episode_lengths = []
    survival_rates = []

    for episode in range(n_episodes):
        observations, info = env.reset()
        episode_reward = 0.0
        episode_length = 0

        while True:
            actions = policy.get_actions(observations, deterministic=deterministic)
            observations, rewards, terminated, truncated, info = env.step(actions)

            episode_reward += sum(rewards.values())
            episode_length += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        survival_rates.append(info["n_alive_agents"] / config.n_beavers)

    env.close()

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_survival": np.mean(survival_rates),
    }


def visualize_simulation(results: Dict[str, Any], output_path: Optional[str] = None):
    """
    Create visualization of simulation results.

    Args:
        results: Results from run_simulation
        output_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib required for visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rewards
    axes[0, 0].plot(results["steps"], results["rewards"])
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Episode Rewards")

    # Agent survival
    axes[0, 1].plot(results["steps"], results["agent_counts"])
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Alive Agents")
    axes[0, 1].set_title("Agent Survival")

    # Water levels
    axes[1, 0].plot(results["steps"], results["water_levels"])
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Avg Water Level")
    axes[1, 0].set_title("Water Dynamics")

    # Structures
    axes[1, 1].plot(results["steps"], results["structures"])
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Structures Built")
    axes[1, 1].set_title("Construction Progress")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def save_animation(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Save simulation frames as animation.

    Args:
        frames: List of RGB frames
        output_path: Output file path
        fps: Frames per second
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print("Matplotlib required for animation")
        return

    if not frames:
        print("No frames to save")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    im = ax.imshow(frames[0])

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)

    try:
        writer = FFMpegWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"Saved animation to {output_path}")
    except Exception as e:
        print(f"Could not save animation: {e}")
        # Try saving as GIF instead
        try:
            anim.save(output_path.replace('.mp4', '.gif'), writer='pillow', fps=fps)
            print(f"Saved as GIF instead")
        except Exception:
            print("Could not save animation in any format")


def print_config_summary(config: SimulationConfig):
    """Print a summary of the configuration"""
    print("\n" + "="*60)
    print("MycoBeaver Configuration Summary")
    print("="*60)
    print(f"Grid size: {config.grid.grid_size}x{config.grid.grid_size}")
    print(f"Number of beavers: {config.n_beavers}")
    print(f"Time step: {config.grid.dt}")
    print()
    print("Subsystems enabled:")
    print(f"  - Pheromones: {config.training.use_pheromones}")
    print(f"  - Projects: {config.training.use_projects}")
    print(f"  - Physarum: {config.training.use_physarum}")
    print(f"  - Overmind: {config.training.use_overmind}")
    print()
    print("Training settings:")
    print(f"  - Episodes: {config.training.n_episodes}")
    print(f"  - Max steps/episode: {config.training.max_steps_per_episode}")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Learning rate: {config.policy.learning_rate}")
    print("="*60 + "\n")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MycoBeaver Multi-Agent Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test simulation
  python -m mycobeaver.main --mode simulate --scenario small

  # Train with standard settings
  python -m mycobeaver.main --mode train --scenario standard

  # Train with custom settings
  python -m mycobeaver.main --mode train --grid-size 64 --n-beavers 15 --episodes 500

  # Evaluate trained policy
  python -m mycobeaver.main --mode evaluate --policy checkpoints/policy.pt

  # Run ablation study
  python -m mycobeaver.main --mode ablation --episodes 500
        """
    )

    parser.add_argument(
        "--mode",
        choices=["simulate", "train", "evaluate", "ablation"],
        default="simulate",
        help="Running mode"
    )

    parser.add_argument(
        "--scenario",
        choices=["small", "standard", "large", "flood", "drought"],
        default="standard",
        help="Benchmark scenario"
    )

    parser.add_argument("--grid-size", type=int, help="Grid size")
    parser.add_argument("--n-beavers", type=int, help="Number of beaver agents")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--seed", type=int, help="Random seed")

    parser.add_argument("--policy", type=str, help="Path to trained policy")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output", type=str, help="Output path for results")

    parser.add_argument("--render", action="store_true", help="Render simulation")
    parser.add_argument("--no-pheromones", action="store_true")
    parser.add_argument("--no-physarum", action="store_true")
    parser.add_argument("--no-projects", action="store_true")
    parser.add_argument("--no-overmind", action="store_true")

    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    # Create config
    config = create_benchmark_config(args.scenario)

    # Apply overrides
    if args.grid_size:
        config.grid.grid_size = args.grid_size
    if args.n_beavers:
        config.n_beavers = args.n_beavers
    if args.episodes:
        config.training.n_episodes = args.episodes

    # Ablation flags
    if args.no_pheromones:
        config.training.use_pheromones = False
    if args.no_physarum:
        config.training.use_physarum = False
    if args.no_projects:
        config.training.use_projects = False
    if args.no_overmind:
        config.training.use_overmind = False

    config.training.checkpoint_dir = args.checkpoint_dir

    print_config_summary(config)

    # Run mode
    if args.mode == "simulate":
        print("Running simulation...")
        results = run_simulation(
            config=config,
            n_steps=args.steps,
            render=args.render,
            seed=args.seed
        )
        print(f"\nSimulation complete!")
        print(f"Total reward: {results['total_reward']:.1f}")
        print(f"Final agents alive: {results['agent_counts'][-1]}")
        print(f"Final structures: {results['structures'][-1]}")

        if args.output:
            visualize_simulation(results, args.output)
            if args.render and results["frames"]:
                save_animation(results["frames"], args.output.replace('.png', '.mp4'))

    elif args.mode == "train":
        try:
            from .training import train_mycobeaver
        except ImportError as e:
            print(f"PyTorch required for training: {e}")
            return

        print("Starting training...")
        history = train_mycobeaver(
            config=config,
            n_episodes=config.training.n_episodes,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir
        )

        # Save final results
        if args.output:
            results = {
                "final_reward": history.metrics[-1].episode_reward,
                "final_survival": history.metrics[-1].n_alive_agents,
                "total_episodes": len(history.metrics),
            }
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

    elif args.mode == "evaluate":
        if not args.policy:
            print("Error: --policy required for evaluate mode")
            return

        print(f"Evaluating policy: {args.policy}")
        metrics = evaluate_policy(
            policy_path=args.policy,
            config=config,
            n_episodes=10
        )
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")

    elif args.mode == "ablation":
        try:
            from .training import AblationStudy
        except ImportError as e:
            print(f"PyTorch required for ablation study: {e}")
            return

        print("Running ablation study...")
        study = AblationStudy(config)

        ablations = ["full", "no_pheromones", "no_physarum", "no_projects", "no_overmind", "baseline"]
        results = study.run(ablations, n_episodes=config.training.n_episodes)

        print("\nAblation Study Results:")
        summary = study.summarize()
        for ablation, metrics in summary.items():
            print(f"\n{ablation}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.3f}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
