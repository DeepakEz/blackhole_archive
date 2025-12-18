"""
Grid Visualization - Environment & Infrastructure View
=======================================================

PHASE 4: Visualization of the physical simulation layer.

Shows:
- Water depth and flow vectors
- Dam permeability and structures
- Physarum network thickness
- Agent positions and movements
- Vegetation density

Key Insight: You should see infrastructure self-organize.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..environment import MycoBeaverEnv, GridState


@dataclass
class GridVisualizerConfig:
    """Configuration for grid visualization"""
    figsize: Tuple[int, int] = (16, 12)
    dpi: int = 100

    # Color maps
    water_cmap: str = "Blues"
    vegetation_cmap: str = "Greens"
    dam_cmap: str = "YlOrBr_r"  # Reversed - darker = more blocked
    physarum_cmap: str = "plasma"

    # Quiver (flow vectors)
    quiver_scale: float = 20.0
    quiver_color: str = "navy"
    quiver_alpha: float = 0.7

    # Agent markers
    agent_size: int = 100
    agent_colors: Dict[str, str] = field(default_factory=lambda: {
        "scout": "#FFD700",     # Gold
        "worker": "#4169E1",    # Royal Blue
        "guardian": "#DC143C",  # Crimson
    })

    # Animation
    animation_interval: int = 200  # ms between frames
    save_animation: bool = False
    animation_format: str = "gif"


class GridVisualizer:
    """
    Comprehensive grid visualization for the MycoBeaver environment.

    Provides multiple views into the physical simulation layer,
    answering the question: "Is infrastructure self-organizing?"
    """

    def __init__(self, config: Optional[GridVisualizerConfig] = None):
        self.config = config or GridVisualizerConfig()

        # History for animation
        self.history: List[Dict[str, Any]] = []
        self.max_history: int = 1000

    def capture_state(self, env: 'MycoBeaverEnv'):
        """Capture current state for history/animation"""
        state = {
            "step": env.current_step,
            "water_depth": env.grid_state.water_depth.copy(),
            "vegetation": env.grid_state.vegetation.copy(),
            "dam_permeability": env.grid_state.dam_permeability.copy(),
            "agent_positions": [(a.id, a.position, a.role.value)
                               for a in env.agents if a.alive],
            "lodge_map": env.grid_state.lodge_map.copy(),
        }

        # Physarum conductivity if available
        if env.physarum_network is not None:
            state["physarum"] = env.physarum_network.get_conductivity_matrix()

        self.history.append(state)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def plot_overview(self, env: 'MycoBeaverEnv',
                      title: Optional[str] = None) -> plt.Figure:
        """
        Create a 2x2 overview of the environment.

        Panels:
        1. Water depth with flow vectors
        2. Vegetation density
        3. Dam permeability (infrastructure)
        4. Physarum network + agents
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize,
                                  dpi=self.config.dpi)

        gs = env.grid_state
        step = env.current_step

        if title is None:
            title = f"MycoBeaver Environment - Step {step}"
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Panel 1: Water depth with flow vectors
        ax1 = axes[0, 0]
        self._plot_water_panel(ax1, env)

        # Panel 2: Vegetation
        ax2 = axes[0, 1]
        self._plot_vegetation_panel(ax2, gs)

        # Panel 3: Dam permeability
        ax3 = axes[1, 0]
        self._plot_dam_panel(ax3, gs)

        # Panel 4: Physarum + Agents
        ax4 = axes[1, 1]
        self._plot_physarum_agents_panel(ax4, env)

        plt.tight_layout()
        return fig

    def _plot_water_panel(self, ax: plt.Axes, env: 'MycoBeaverEnv'):
        """Plot water depth with flow vectors"""
        gs = env.grid_state

        # Water depth heatmap
        im = ax.imshow(gs.water_depth, cmap=self.config.water_cmap,
                       origin='upper', vmin=0, vmax=1.0)
        plt.colorbar(im, ax=ax, label='Water Depth (m)')

        # Compute flow vectors
        flow_y, flow_x = self._compute_flow_vectors(env)

        # Subsample for clarity
        step = max(1, gs.water_depth.shape[0] // 20)
        Y, X = np.mgrid[0:gs.water_depth.shape[0]:step,
                        0:gs.water_depth.shape[1]:step]

        ax.quiver(X, Y,
                  flow_x[::step, ::step],
                  -flow_y[::step, ::step],  # Flip Y for image coords
                  color=self.config.quiver_color,
                  alpha=self.config.quiver_alpha,
                  scale=self.config.quiver_scale)

        ax.set_title('Water Depth & Flow Vectors')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def _compute_flow_vectors(self, env: 'MycoBeaverEnv') -> Tuple[np.ndarray, np.ndarray]:
        """Compute water flow vectors from gradients"""
        gs = env.grid_state

        # Water surface height
        H = gs.elevation + gs.water_depth

        # Gradient (flow goes from high to low)
        grad_y, grad_x = np.gradient(H)

        # Flow is opposite to gradient (water flows downhill)
        flow_x = -grad_x
        flow_y = -grad_y

        # Scale by conductance (dam effect)
        flow_x *= gs.dam_permeability
        flow_y *= gs.dam_permeability

        return flow_y, flow_x

    def _plot_vegetation_panel(self, ax: plt.Axes, gs: 'GridState'):
        """Plot vegetation density"""
        im = ax.imshow(gs.vegetation, cmap=self.config.vegetation_cmap,
                       origin='upper', vmin=0)
        plt.colorbar(im, ax=ax, label='Vegetation Density')

        # Mark lodges
        lodge_y, lodge_x = np.where(gs.lodge_map)
        ax.scatter(lodge_x, lodge_y, c='white', s=200, marker='s',
                   edgecolors='black', linewidths=2, label='Lodge')

        ax.set_title('Vegetation Density')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')

    def _plot_dam_panel(self, ax: plt.Axes, gs: 'GridState'):
        """Plot dam permeability (infrastructure)"""
        # Invert so darker = more blocked
        dam_strength = 1.0 - gs.dam_permeability

        im = ax.imshow(dam_strength, cmap=self.config.dam_cmap,
                       origin='upper', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Dam Strength (1 - permeability)')

        # Highlight strong dams
        strong_dams = dam_strength > 0.5
        dam_y, dam_x = np.where(strong_dams)
        ax.scatter(dam_x, dam_y, c='brown', s=30, marker='s',
                   alpha=0.5, label='Strong Dam')

        ax.set_title('Infrastructure (Dam Permeability)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if len(dam_y) > 0:
            ax.legend(loc='upper right')

    def _plot_physarum_agents_panel(self, ax: plt.Axes, env: 'MycoBeaverEnv'):
        """Plot physarum network with agent positions"""
        gs = env.grid_state

        # Physarum conductivity
        if env.physarum_network is not None:
            conductivity = env.physarum_network.get_conductivity_matrix()
            im = ax.imshow(conductivity, cmap=self.config.physarum_cmap,
                          origin='upper', vmin=0, alpha=0.7)
            plt.colorbar(im, ax=ax, label='Physarum Conductivity')
        else:
            ax.imshow(np.zeros_like(gs.water_depth), cmap='gray',
                     origin='upper')

        # Plot agents by role
        for role, color in self.config.agent_colors.items():
            agents_of_role = [(a.position[1], a.position[0])
                             for a in env.agents
                             if a.alive and a.role.value == role]
            if agents_of_role:
                xs, ys = zip(*agents_of_role)
                ax.scatter(xs, ys, c=color, s=self.config.agent_size,
                          marker='o', edgecolors='black', linewidths=1,
                          label=role.capitalize())

        # Mark lodges
        lodge_y, lodge_x = np.where(gs.lodge_map)
        ax.scatter(lodge_x, lodge_y, c='white', s=200, marker='s',
                   edgecolors='black', linewidths=2, label='Lodge')

        ax.set_title('Physarum Network & Agents')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=8)

    def plot_physarum_detail(self, env: 'MycoBeaverEnv') -> plt.Figure:
        """
        Detailed physarum network visualization.

        Shows edge-level conductivity as a graph overlay.
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.config.dpi)
        gs = env.grid_state

        # Background: water + vegetation blend
        blend = 0.5 * (gs.water_depth / np.max(gs.water_depth + 1e-6)) + \
                0.5 * (gs.vegetation / np.max(gs.vegetation + 1e-6))
        ax.imshow(blend, cmap='terrain', origin='upper', alpha=0.3)

        if env.physarum_network is not None:
            # Get edge data
            edges = env.physarum_network.get_edge_data()

            if edges:
                # Create line segments
                segments = []
                colors = []

                for edge in edges:
                    y1, x1 = edge['source']
                    y2, x2 = edge['target']
                    segments.append([(x1, y1), (x2, y2)])
                    colors.append(edge['conductivity'])

                # Normalize colors
                colors = np.array(colors)
                if np.max(colors) > 0:
                    colors = colors / np.max(colors)

                # Create LineCollection
                lc = LineCollection(segments,
                                   cmap=self.config.physarum_cmap,
                                   linewidths=1 + 3 * colors)
                lc.set_array(colors)
                ax.add_collection(lc)
                plt.colorbar(lc, ax=ax, label='Normalized Conductivity')

        # Plot agents
        for agent in env.agents:
            if agent.alive:
                color = self.config.agent_colors.get(agent.role.value, 'gray')
                ax.scatter(agent.position[1], agent.position[0],
                          c=color, s=100, marker='o',
                          edgecolors='black', linewidths=1)

        ax.set_title(f'Physarum Network Detail - Step {env.current_step}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        return fig


def plot_water_flow(env: 'MycoBeaverEnv',
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot water depth and flow.

    Quick diagnostic: Are dams affecting flow?
    """
    viz = GridVisualizer()
    fig, ax = plt.subplots(figsize=(10, 8))
    viz._plot_water_panel(ax, env)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_infrastructure(env: 'MycoBeaverEnv',
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot dam infrastructure.

    Quick diagnostic: Is infrastructure emerging?
    """
    viz = GridVisualizer()
    fig, ax = plt.subplots(figsize=(10, 8))
    viz._plot_dam_panel(ax, env.grid_state)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_physarum_network(env: 'MycoBeaverEnv',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot physarum network.

    Quick diagnostic: Are efficient paths forming?
    """
    viz = GridVisualizer()
    fig = viz.plot_physarum_detail(env)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def animate_simulation(env: 'MycoBeaverEnv',
                       n_steps: int = 100,
                       actions_fn=None,
                       save_path: Optional[str] = None) -> FuncAnimation:
    """
    Create an animation of the simulation.

    Args:
        env: The environment
        n_steps: Number of steps to animate
        actions_fn: Function that returns actions dict (if None, random actions)
        save_path: Path to save animation (if provided)

    Returns:
        FuncAnimation object
    """
    viz = GridVisualizer()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def init():
        for ax in axes.flat:
            ax.clear()
        return []

    def update(frame):
        # Generate actions
        if actions_fn is not None:
            actions = actions_fn(env)
        else:
            # Random actions
            actions = {f"agent_{i}": np.random.randint(0, 14)
                      for i in range(len(env.agents))}

        # Step environment
        env.step(actions)

        # Clear and redraw
        for ax in axes.flat:
            ax.clear()

        gs = env.grid_state

        # Water
        axes[0, 0].imshow(gs.water_depth, cmap='Blues', origin='upper')
        axes[0, 0].set_title(f'Water - Step {env.current_step}')

        # Vegetation
        axes[0, 1].imshow(gs.vegetation, cmap='Greens', origin='upper')
        axes[0, 1].set_title('Vegetation')

        # Dams
        axes[1, 0].imshow(1 - gs.dam_permeability, cmap='YlOrBr', origin='upper')
        axes[1, 0].set_title('Dam Strength')

        # Agents
        if env.physarum_network is not None:
            axes[1, 1].imshow(env.physarum_network.get_conductivity_matrix(),
                             cmap='plasma', origin='upper', alpha=0.7)
        for agent in env.agents:
            if agent.alive:
                color = {'scout': 'gold', 'worker': 'blue', 'guardian': 'red'
                        }.get(agent.role.value, 'gray')
                axes[1, 1].scatter(agent.position[1], agent.position[0],
                                  c=color, s=50)
        axes[1, 1].set_title('Physarum + Agents')

        return []

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_steps, interval=200, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=5)

    return anim
