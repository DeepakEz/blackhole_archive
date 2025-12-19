#!/usr/bin/env python
"""
MycoBeaver Research Visualization
==================================
Generate publication-quality plots, graphs, and maps from training results.

Usage:
    # Plot training curves from a training run
    python visualize.py --mode training --input checkpoints/training_history.json --output figures/

    # Plot ablation study comparison
    python visualize.py --mode ablation --input results/ablation/ --output figures/

    # Generate all visualizations
    python visualize.py --mode all --input checkpoints/ --output figures/

    # Generate environment map visualization
    python visualize.py --mode map --policy checkpoints/policy_final.pt --output figures/
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

# Publication-quality settings
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def setup_plot_style():
    """Apply publication-quality plot settings"""
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(PLOT_STYLE)


def smooth_curve(data: List[float], window: int = 50) -> np.ndarray:
    """Apply moving average smoothing to a curve"""
    if len(data) < window:
        return np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    # Pad the beginning
    padding = np.array(data[:window-1])
    return np.concatenate([padding, smoothed])


def plot_training_curves(history_path: str, output_dir: str, show: bool = False):
    """
    Generate training curve plots from training history.

    Creates:
    - training_reward.pdf: Episode reward over time
    - training_losses.pdf: Policy and value losses
    - training_agents.pdf: Agent survival and structures
    - training_stability.pdf: PPO stability metrics
    - training_combined.pdf: All metrics in one figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    setup_plot_style()

    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    metrics = history.get("metrics", [])
    if not metrics:
        print(f"No metrics found in {history_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = [m["episode"] for m in metrics]

    # 1. Reward curve
    fig, ax = plt.subplots(figsize=(10, 6))
    rewards = [m["reward"] for m in metrics]
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    ax.plot(episodes, smooth_curve(rewards), color='blue', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Reward Curve')
    ax.legend()
    fig.savefig(output_dir / 'training_reward.pdf')
    fig.savefig(output_dir / 'training_reward.png')
    print(f"Saved: {output_dir / 'training_reward.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 2. Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    policy_loss = [m["policy_loss"] for m in metrics]
    ax1.plot(episodes, policy_loss, alpha=0.3, color='red')
    ax1.plot(episodes, smooth_curve(policy_loss), color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Policy Loss')
    ax1.set_title('Policy Loss')

    value_loss = [m["value_loss"] for m in metrics]
    ax2.plot(episodes, value_loss, alpha=0.3, color='green')
    ax2.plot(episodes, smooth_curve(value_loss), color='green', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('Value Function Loss')

    plt.tight_layout()
    fig.savefig(output_dir / 'training_losses.pdf')
    fig.savefig(output_dir / 'training_losses.png')
    print(f"Saved: {output_dir / 'training_losses.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 3. Agent metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    alive_agents = [m["n_alive_agents"] for m in metrics]
    ax1.plot(episodes, alive_agents, color='purple', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Agents Alive')
    ax1.set_title('Agent Survival')

    structures = [m["n_structures"] for m in metrics]
    ax2.plot(episodes, structures, alpha=0.3, color='brown')
    ax2.plot(episodes, smooth_curve(structures), color='brown', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Structures Built')
    ax2.set_title('Infrastructure Construction')

    plt.tight_layout()
    fig.savefig(output_dir / 'training_agents.pdf')
    fig.savefig(output_dir / 'training_agents.png')
    print(f"Saved: {output_dir / 'training_agents.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 4. PPO Stability metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    entropy = [m.get("entropy", 0) for m in metrics]
    axes[0, 0].plot(episodes, entropy, alpha=0.3, color='orange')
    axes[0, 0].plot(episodes, smooth_curve(entropy), color='orange', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('Policy Entropy')

    kl = [m.get("approx_kl", 0) for m in metrics]
    axes[0, 1].plot(episodes, kl, alpha=0.3, color='red')
    axes[0, 1].plot(episodes, smooth_curve(kl), color='red', linewidth=2)
    axes[0, 1].axhline(y=0.015, color='black', linestyle='--', label='KL Target')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Approx KL')
    axes[0, 1].set_title('KL Divergence')
    axes[0, 1].legend()

    exp_var = [m.get("explained_variance", 0) for m in metrics]
    axes[1, 0].plot(episodes, exp_var, alpha=0.3, color='green')
    axes[1, 0].plot(episodes, smooth_curve(exp_var), color='green', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Explained Variance')
    axes[1, 0].set_title('Value Function Quality')
    axes[1, 0].set_ylim(-0.1, 1.1)

    adv_var = [m.get("advantage_variance", 0) for m in metrics]
    axes[1, 1].plot(episodes, adv_var, alpha=0.3, color='blue')
    axes[1, 1].plot(episodes, smooth_curve(adv_var), color='blue', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Advantage Variance')
    axes[1, 1].set_title('Advantage Estimation Stability')

    plt.tight_layout()
    fig.savefig(output_dir / 'training_stability.pdf')
    fig.savefig(output_dir / 'training_stability.png')
    print(f"Saved: {output_dir / 'training_stability.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 5. Combined overview figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Reward (large)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(episodes, rewards, alpha=0.2, color='blue')
    ax1.plot(episodes, smooth_curve(rewards), color='blue', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')

    # Survival
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(episodes, alive_agents, color='purple')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Agents')
    ax2.set_title('Survival')

    # Losses
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, smooth_curve(policy_loss), color='red', label='Policy')
    ax3.plot(episodes, smooth_curve(value_loss), color='green', label='Value')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Losses')
    ax3.legend()

    # Entropy
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes, smooth_curve(entropy), color='orange')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy')

    # KL
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(episodes, smooth_curve(kl), color='red')
    ax5.axhline(y=0.015, color='black', linestyle='--')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('KL')
    ax5.set_title('KL Divergence')

    # Structures
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(episodes, smooth_curve(structures), color='brown')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Count')
    ax6.set_title('Structures')

    # Explained variance
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(episodes, smooth_curve(exp_var), color='green')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Exp. Var.')
    ax7.set_title('Value Quality')
    ax7.set_ylim(-0.1, 1.1)

    # Water level
    water = [m.get("avg_water_level", 0) for m in metrics]
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(episodes, smooth_curve(water), color='cyan')
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Level')
    ax8.set_title('Avg Water')

    fig.suptitle('MycoBeaver Training Summary', fontsize=18, y=0.98)
    fig.savefig(output_dir / 'training_combined.pdf')
    fig.savefig(output_dir / 'training_combined.png')
    print(f"Saved: {output_dir / 'training_combined.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    print(f"\nTraining summary:")
    print(f"  Total episodes: {len(metrics)}")
    print(f"  Final reward: {rewards[-1]:.1f}")
    print(f"  Final survival: {alive_agents[-1]}")
    print(f"  Final structures: {structures[-1]}")


def plot_ablation_comparison(ablation_dir: str, output_dir: str, show: bool = False):
    """
    Generate ablation study comparison plots.

    Creates:
    - ablation_rewards.pdf: Reward comparison across conditions
    - ablation_bar.pdf: Final performance bar chart
    - ablation_survival.pdf: Agent survival comparison
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    setup_plot_style()

    ablation_dir = Path(ablation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all ablation results
    ablation_data = {}

    # Look for training_history.json in subdirectories or metrics files
    for subdir in ablation_dir.iterdir():
        if subdir.is_dir():
            history_file = subdir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    ablation_data[subdir.name] = json.load(f)

    # Also check for metrics files directly
    for metrics_file in ablation_dir.glob("metrics_*.json"):
        name = metrics_file.stem.replace("metrics_", "")
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            if "recent_metrics" in data:
                ablation_data[name] = {"metrics": data["recent_metrics"]}

    if not ablation_data:
        print(f"No ablation results found in {ablation_dir}")
        print("Expected: subdirectories with training_history.json or metrics_*.json files")
        return

    # Color palette for conditions
    colors = {
        'full': '#2ecc71',       # Green
        'no_pheromones': '#e74c3c',  # Red
        'no_physarum': '#3498db',    # Blue
        'no_projects': '#9b59b6',    # Purple
        'no_overmind': '#f39c12',    # Orange
        'baseline': '#95a5a6',       # Gray
    }

    # 1. Reward curves comparison
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in sorted(ablation_data.items()):
        metrics = data.get("metrics", [])
        if not metrics:
            continue
        episodes = [m.get("episode", i) for i, m in enumerate(metrics)]
        rewards = [m.get("reward", m.get("episode_reward", 0)) for m in metrics]

        color = colors.get(name, '#333333')
        ax.plot(episodes, smooth_curve(rewards),
                label=name.replace('_', ' ').title(),
                color=color, linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Ablation Study: Reward Comparison')
    ax.legend(loc='best')

    fig.savefig(output_dir / 'ablation_rewards.pdf')
    fig.savefig(output_dir / 'ablation_rewards.png')
    print(f"Saved: {output_dir / 'ablation_rewards.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 2. Final performance bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = []
    final_rewards = []
    final_survival = []
    final_structures = []
    bar_colors = []

    for name, data in sorted(ablation_data.items()):
        metrics = data.get("metrics", [])
        if not metrics:
            continue
        # Use last 50 episodes average for final performance
        last_metrics = metrics[-50:] if len(metrics) >= 50 else metrics

        names.append(name.replace('_', '\n'))
        final_rewards.append(np.mean([m.get("reward", m.get("episode_reward", 0)) for m in last_metrics]))
        final_survival.append(np.mean([m.get("n_alive_agents", 0) for m in last_metrics]))
        final_structures.append(np.mean([m.get("n_structures", 0) for m in last_metrics]))
        bar_colors.append(colors.get(name, '#333333'))

    x = np.arange(len(names))

    axes[0].bar(x, final_rewards, color=bar_colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Final Reward')

    axes[1].bar(x, final_survival, color=bar_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylabel('Agents Alive')
    axes[1].set_title('Final Survival')

    axes[2].bar(x, final_structures, color=bar_colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, fontsize=9)
    axes[2].set_ylabel('Structures')
    axes[2].set_title('Final Infrastructure')

    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_bar.pdf')
    fig.savefig(output_dir / 'ablation_bar.png')
    print(f"Saved: {output_dir / 'ablation_bar.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 3. Survival curves
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in sorted(ablation_data.items()):
        metrics = data.get("metrics", [])
        if not metrics:
            continue
        episodes = [m.get("episode", i) for i, m in enumerate(metrics)]
        survival = [m.get("n_alive_agents", 0) for m in metrics]

        color = colors.get(name, '#333333')
        ax.plot(episodes, survival,
                label=name.replace('_', ' ').title(),
                color=color, linewidth=2, alpha=0.8)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Agents Alive')
    ax.set_title('Ablation Study: Agent Survival')
    ax.legend(loc='best')

    fig.savefig(output_dir / 'ablation_survival.pdf')
    fig.savefig(output_dir / 'ablation_survival.png')
    print(f"Saved: {output_dir / 'ablation_survival.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # Print summary table
    print("\nAblation Study Summary:")
    print("-" * 70)
    print(f"{'Condition':<20} {'Reward':>12} {'Survival':>12} {'Structures':>12}")
    print("-" * 70)
    for i, name in enumerate(names):
        print(f"{name.replace(chr(10), ' '):<20} {final_rewards[i]:>12.1f} {final_survival[i]:>12.1f} {final_structures[i]:>12.1f}")
    print("-" * 70)


def plot_environment_map(policy_path: Optional[str], output_dir: str,
                         n_steps: int = 200, show: bool = False):
    """
    Generate environment state visualizations (maps).

    Creates:
    - map_terrain.pdf: Terrain elevation map
    - map_water.pdf: Water depth map
    - map_vegetation.pdf: Vegetation distribution
    - map_agents.pdf: Agent positions and paths
    - map_combined.pdf: All layers combined
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    setup_plot_style()

    # Import environment components
    import sys
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

    from mycobeaver.config import create_default_config
    from mycobeaver.environment import MycoBeaverEnv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    config = create_default_config()
    config.grid.grid_size = 64
    config.n_beavers = 10

    env = MycoBeaverEnv(config, render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    # Optionally load policy
    policy = None
    if policy_path and Path(policy_path).exists():
        try:
            from mycobeaver.policy import MultiAgentPolicy
            policy = MultiAgentPolicy(config)
            policy.load(policy_path)
            print(f"Loaded policy from {policy_path}")
        except Exception as e:
            print(f"Could not load policy: {e}")

    # Run simulation to populate environment
    agent_paths = {f"agent_{i}": [] for i in range(config.n_beavers)}

    for step in range(n_steps):
        if policy is not None:
            actions = policy.get_actions(obs, deterministic=True)
        else:
            actions = {f"agent_{i}": np.random.randint(0, config.policy.n_actions)
                      for i in range(config.n_beavers)}

        # Track agent positions (position is a (y, x) tuple)
        for i, agent in enumerate(env.agents):
            if agent.alive:
                y, x = agent.position
                agent_paths[f"agent_{i}"].append((x, y))

        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    # Get final state
    grid = env.grid_state

    # Custom colormaps
    water_cmap = LinearSegmentedColormap.from_list('water',
        ['#f0f8ff', '#87ceeb', '#4169e1', '#00008b'])
    veg_cmap = LinearSegmentedColormap.from_list('vegetation',
        ['#f5deb3', '#90ee90', '#228b22', '#006400'])

    # 1. Terrain elevation
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(grid.terrain_height.T, origin='lower', cmap='terrain')
    plt.colorbar(im, ax=ax, label='Elevation')
    ax.set_title('Terrain Elevation Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig(output_dir / 'map_terrain.pdf')
    fig.savefig(output_dir / 'map_terrain.png')
    print(f"Saved: {output_dir / 'map_terrain.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 2. Water depth
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(grid.water_depth.T, origin='lower', cmap=water_cmap, vmin=0)
    plt.colorbar(im, ax=ax, label='Water Depth')
    ax.set_title('Water Distribution Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig(output_dir / 'map_water.pdf')
    fig.savefig(output_dir / 'map_water.png')
    print(f"Saved: {output_dir / 'map_water.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 3. Vegetation
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(grid.vegetation.T, origin='lower', cmap=veg_cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Vegetation Density')
    ax.set_title('Vegetation Distribution Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig(output_dir / 'map_vegetation.pdf')
    fig.savefig(output_dir / 'map_vegetation.png')
    print(f"Saved: {output_dir / 'map_vegetation.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 4. Agent positions and paths
    fig, ax = plt.subplots(figsize=(10, 10))
    # Background: terrain
    ax.imshow(grid.terrain_height.T, origin='lower', cmap='terrain', alpha=0.5)

    # Plot agent paths
    colors = plt.cm.tab10(np.linspace(0, 1, config.n_beavers))
    for i, (agent_id, path) in enumerate(agent_paths.items()):
        if len(path) > 1:
            xs, ys = zip(*path)
            ax.plot(xs, ys, color=colors[i], alpha=0.5, linewidth=1)
            # Final position
            ax.scatter(xs[-1], ys[-1], color=colors[i], s=100,
                      edgecolor='black', linewidth=2, zorder=10)

    # Plot structures (from dam_permeability grid)
    structure_mask = grid.dam_permeability < 0.9
    structure_coords = np.where(structure_mask)
    if len(structure_coords[0]) > 0:
        ax.scatter(structure_coords[0], structure_coords[1], marker='s', s=30,
                  c='brown', edgecolor='black', alpha=0.7, label='Structures')

    ax.set_title('Agent Paths and Structures')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    fig.savefig(output_dir / 'map_agents.pdf')
    fig.savefig(output_dir / 'map_agents.png')
    print(f"Saved: {output_dir / 'map_agents.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    # 5. Combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Terrain
    im1 = axes[0, 0].imshow(grid.terrain_height.T, origin='lower', cmap='terrain')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    axes[0, 0].set_title('Terrain')

    # Water
    im2 = axes[0, 1].imshow(grid.water_depth.T, origin='lower', cmap=water_cmap, vmin=0)
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    axes[0, 1].set_title('Water')

    # Vegetation
    im3 = axes[1, 0].imshow(grid.vegetation.T, origin='lower', cmap=veg_cmap, vmin=0, vmax=1)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title('Vegetation')

    # Agents + structures overlay
    axes[1, 1].imshow(grid.terrain_height.T, origin='lower', cmap='terrain', alpha=0.4)
    axes[1, 1].imshow(grid.water_depth.T, origin='lower', cmap=water_cmap, alpha=0.3)
    for i, (agent_id, path) in enumerate(agent_paths.items()):
        if len(path) > 1:
            xs, ys = zip(*path)
            axes[1, 1].plot(xs, ys, color=colors[i], alpha=0.5, linewidth=1)
            axes[1, 1].scatter(xs[-1], ys[-1], color=colors[i], s=80, edgecolor='black')
    if len(structure_coords[0]) > 0:
        axes[1, 1].scatter(structure_coords[0], structure_coords[1], marker='s', s=25, c='brown', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Agents & Infrastructure')

    fig.suptitle('MycoBeaver Environment State', fontsize=16)
    plt.tight_layout()
    fig.savefig(output_dir / 'map_combined.pdf')
    fig.savefig(output_dir / 'map_combined.png')
    print(f"Saved: {output_dir / 'map_combined.pdf'}")
    if show:
        plt.show()
    plt.close(fig)

    env.close()

    n_structures = np.sum(structure_mask)
    print(f"\nEnvironment summary:")
    print(f"  Grid size: {config.grid.grid_size}x{config.grid.grid_size}")
    print(f"  Agents: {config.n_beavers}")
    print(f"  Structures built: {n_structures}")
    print(f"  Steps simulated: {n_steps}")


def main():
    parser = argparse.ArgumentParser(
        description="MycoBeaver Research Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot training curves
  python visualize.py --mode training --input checkpoints/training_history.json

  # Plot ablation comparison
  python visualize.py --mode ablation --input results/ablation/

  # Generate environment maps
  python visualize.py --mode map --policy checkpoints/policy_final.pt

  # Generate all visualizations
  python visualize.py --mode all --input checkpoints/
        """
    )

    parser.add_argument("--mode", choices=["training", "ablation", "map", "all"],
                        default="training", help="Visualization mode")
    parser.add_argument("--input", "-i", type=str,
                        default="checkpoints/training_history.json",
                        help="Input file or directory")
    parser.add_argument("--output", "-o", type=str, default="figures/",
                        help="Output directory for figures")
    parser.add_argument("--policy", type=str, help="Path to trained policy for map mode")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--steps", type=int, default=200,
                        help="Simulation steps for map generation")

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return

    if args.mode == "training":
        plot_training_curves(args.input, args.output, show=args.show)

    elif args.mode == "ablation":
        plot_ablation_comparison(args.input, args.output, show=args.show)

    elif args.mode == "map":
        plot_environment_map(args.policy, args.output,
                            n_steps=args.steps, show=args.show)

    elif args.mode == "all":
        input_path = Path(args.input)

        # Training curves
        history_file = input_path / "training_history.json"
        if history_file.exists():
            print("\n=== Training Curves ===")
            plot_training_curves(str(history_file), args.output, show=args.show)

        # Ablation (look for subdirs or parent dir)
        ablation_dir = input_path / "ablation" if (input_path / "ablation").exists() else input_path
        if ablation_dir.exists():
            print("\n=== Ablation Comparison ===")
            plot_ablation_comparison(str(ablation_dir), args.output, show=args.show)

        # Maps
        policy_file = args.policy or str(input_path / "policy_final.pt")
        print("\n=== Environment Maps ===")
        plot_environment_map(policy_file, args.output,
                            n_steps=args.steps, show=args.show)

    print(f"\nAll figures saved to: {args.output}")


if __name__ == "__main__":
    main()
