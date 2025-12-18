"""
Information Thermodynamics Dashboard
====================================

PHASE 4: This is where your system becomes scientifically interesting.

Visualizes the information-theoretic properties of the colony:
- Info energy spent vs gained
- Semantic entropy vs policy entropy
- Infrastructure ROI (energy saved vs info spent)
- Free energy landscape
- Phase transitions in colony behavior

Based on the Free Energy Principle and Information Thermodynamics framework
from the DCA (Distributed Cognitive Architecture) design.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import deque

if TYPE_CHECKING:
    from ..environment import MycoBeaverEnv


@dataclass
class ThermodynamicsConfig:
    """Configuration for thermodynamics visualization"""
    figsize: Tuple[int, int] = (16, 12)
    dpi: int = 100

    # Colors
    energy_spent_color: str = "#DC143C"  # Crimson
    energy_gained_color: str = "#228B22"  # Forest green
    entropy_color: str = "#4169E1"       # Royal blue
    coherence_color: str = "#32CD32"     # Lime green
    roi_positive_color: str = "#228B22"  # Green
    roi_negative_color: str = "#DC143C"  # Red

    # History size
    max_history: int = 1000

    # Smoothing window for rolling statistics
    smoothing_window: int = 20


class ThermodynamicsVisualizer:
    """
    Visualization tools for information thermodynamics.

    Tracks and displays:
    - Info energy budget (spent vs recovered)
    - Semantic entropy dynamics
    - Policy entropy (exploration vs exploitation)
    - Infrastructure ROI
    - Free energy landscape
    """

    def __init__(self, config: Optional[ThermodynamicsConfig] = None):
        self.config = config or ThermodynamicsConfig()

        # Energy budget tracking
        self.info_spent_history: deque = deque(maxlen=self.config.max_history)
        self.info_gained_history: deque = deque(maxlen=self.config.max_history)
        self.net_info_history: deque = deque(maxlen=self.config.max_history)

        # Entropy tracking
        self.semantic_entropy_history: deque = deque(maxlen=self.config.max_history)
        self.policy_entropy_history: deque = deque(maxlen=self.config.max_history)
        self.coherence_history: deque = deque(maxlen=self.config.max_history)

        # ROI tracking
        self.infrastructure_built: deque = deque(maxlen=self.config.max_history)
        self.energy_saved_by_infra: deque = deque(maxlen=self.config.max_history)
        self.info_spent_on_infra: deque = deque(maxlen=self.config.max_history)

        # Agent-level tracking
        self.agent_info_energies: List[List[float]] = []

        # Temperature tracking (simulated annealing dynamics)
        self.temperature_history: deque = deque(maxlen=self.config.max_history)

    def record_step(self, env: 'MycoBeaverEnv', policy_entropy: float = 0.0):
        """
        Record thermodynamic metrics for current step.

        Args:
            env: The environment
            policy_entropy: Optional policy entropy from PPO (if available)
        """
        # Info energy budget
        info_spent = 0.0
        info_gained = 0.0

        # Collect from pheromone field
        if env.pheromone_field is not None:
            info_spent += env.pheromone_field.get_info_dissipation()

        # Collect from semantic system
        if env.semantic_system is not None:
            info_spent += env.semantic_system.get_info_dissipation()

        # Collect from agents
        total_agent_info = sum(a.info_energy for a in env.agents if a.alive)
        if self.agent_info_energies:
            prev_total = sum(self.agent_info_energies[-1]) if self.agent_info_energies[-1] else 0
            info_gained = max(0, total_agent_info - prev_total + info_spent)

        self.info_spent_history.append(info_spent)
        self.info_gained_history.append(info_gained)
        self.net_info_history.append(info_gained - info_spent)

        # Agent info energies
        self.agent_info_energies.append([a.info_energy for a in env.agents if a.alive])
        if len(self.agent_info_energies) > self.config.max_history:
            self.agent_info_energies.pop(0)

        # Semantic entropy
        if env.semantic_system is not None:
            entropy = env.semantic_system.shared_graph.compute_semantic_entropy()
            coherence = env.semantic_system.shared_graph.compute_coherence()
            self.semantic_entropy_history.append(entropy)
            self.coherence_history.append(coherence)

            # Temperature
            temp = env.semantic_system.shared_graph.temperature
            self.temperature_history.append(temp)
        else:
            self.semantic_entropy_history.append(0.0)
            self.coherence_history.append(1.0)
            self.temperature_history.append(1.0)

        # Policy entropy (from training if available)
        self.policy_entropy_history.append(policy_entropy)

        # Infrastructure ROI
        n_structures = np.sum(env.grid_state.dam_permeability < 0.9)
        self.infrastructure_built.append(n_structures)

        # Estimate energy saved by infrastructure (more dams = more controlled water)
        water_variance = np.var(env.grid_state.water_depth)
        energy_saved = max(0, 1.0 - water_variance) * n_structures * 0.1
        self.energy_saved_by_infra.append(energy_saved)

        # Info spent on infrastructure (approximation)
        self.info_spent_on_infra.append(info_spent * 0.3)  # Assume 30% goes to building

    def plot_dashboard(self, env: 'MycoBeaverEnv') -> plt.Figure:
        """
        Create comprehensive thermodynamics dashboard.

        6-panel view showing all thermodynamic metrics.
        """
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        step = env.current_step
        fig.suptitle(f'Information Thermodynamics Dashboard - Step {step}',
                    fontsize=14, fontweight='bold')

        # Panel 1: Info Energy Budget (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_info_budget(ax1)

        # Panel 2: Semantic vs Policy Entropy (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_entropy_dynamics(ax2)

        # Panel 3: Infrastructure ROI (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_infrastructure_roi(ax3)

        # Panel 4: Agent Info Energy Distribution (middle-right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_agent_info_distribution(ax4)

        # Panel 5: Free Energy Landscape (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_free_energy(ax5)

        # Panel 6: Temperature Dynamics (bottom-right)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_temperature_dynamics(ax6)

        return fig

    def _plot_info_budget(self, ax: plt.Axes):
        """Plot info energy spent vs gained over time"""
        steps = range(len(self.info_spent_history))

        spent = list(self.info_spent_history)
        gained = list(self.info_gained_history)
        net = list(self.net_info_history)

        ax.fill_between(steps, 0, spent, color=self.config.energy_spent_color,
                       alpha=0.5, label='Info Spent')
        ax.fill_between(steps, 0, gained, color=self.config.energy_gained_color,
                       alpha=0.5, label='Info Gained')
        ax.plot(steps, net, 'k-', linewidth=2, label='Net Info')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Cumulative balance
        if net:
            cum_balance = np.cumsum(net)
            ax2 = ax.twinx()
            ax2.plot(steps, cum_balance, 'purple', linestyle=':', linewidth=2,
                    label='Cumulative Balance')
            ax2.set_ylabel('Cumulative Info Balance', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

        ax.set_xlabel('Step')
        ax.set_ylabel('Info Energy per Step')
        ax.set_title('Information Energy Budget\n(Spent vs Gained)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_entropy_dynamics(self, ax: plt.Axes):
        """Plot semantic entropy vs policy entropy"""
        steps = range(len(self.semantic_entropy_history))

        semantic = list(self.semantic_entropy_history)
        policy = list(self.policy_entropy_history)
        coherence = list(self.coherence_history)

        ax.plot(steps, semantic, color=self.config.entropy_color,
                linewidth=2, label='Semantic Entropy')
        ax.plot(steps, policy, color='orange', linewidth=2,
                linestyle='--', label='Policy Entropy')

        # Secondary axis for coherence
        ax2 = ax.twinx()
        ax2.plot(steps, coherence, color=self.config.coherence_color,
                linewidth=2, linestyle=':', label='Coherence')
        ax2.set_ylabel('Coherence', color=self.config.coherence_color)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor=self.config.coherence_color)

        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy Dynamics\n(Semantic vs Policy)')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_infrastructure_roi(self, ax: plt.Axes):
        """Plot infrastructure return on investment"""
        steps = range(len(self.infrastructure_built))

        infra = list(self.infrastructure_built)
        saved = list(self.energy_saved_by_infra)
        spent = list(self.info_spent_on_infra)

        # Compute ROI
        roi = []
        for s, sp in zip(saved, spent):
            if sp > 0:
                roi.append((s - sp) / sp * 100)  # Percentage ROI
            else:
                roi.append(0)

        # Plot infrastructure count
        ax.plot(steps, infra, 'b-', linewidth=2, label='Infrastructure Units')

        # ROI on secondary axis
        ax2 = ax.twinx()
        colors = [self.config.roi_positive_color if r >= 0
                 else self.config.roi_negative_color for r in roi]
        ax2.bar(steps, roi, color=colors, alpha=0.5, width=1.0)
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_ylabel('ROI (%)', color='gray')

        ax.set_xlabel('Step')
        ax.set_ylabel('Infrastructure Count')
        ax.set_title('Infrastructure ROI\n(Energy Saved vs Info Spent)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_agent_info_distribution(self, ax: plt.Axes):
        """Plot distribution of info energy across agents"""
        if not self.agent_info_energies or not self.agent_info_energies[-1]:
            ax.text(0.5, 0.5, 'No agent data', ha='center', va='center')
            return

        current_energies = self.agent_info_energies[-1]

        # Histogram of current distribution
        ax.hist(current_energies, bins=20, color='steelblue',
               edgecolor='black', alpha=0.7)

        # Statistics
        mean_energy = np.mean(current_energies)
        std_energy = np.std(current_energies)
        min_energy = np.min(current_energies)
        max_energy = np.max(current_energies)

        ax.axvline(mean_energy, color='red', linestyle='--',
                  label=f'Mean: {mean_energy:.1f}')
        ax.axvline(mean_energy - std_energy, color='orange', linestyle=':',
                  alpha=0.7)
        ax.axvline(mean_energy + std_energy, color='orange', linestyle=':',
                  alpha=0.7, label=f'Std: {std_energy:.1f}')

        # Info deficit warning zone
        ax.axvspan(0, 20, color='red', alpha=0.1, label='Info Deficit Zone')

        ax.set_xlabel('Info Energy')
        ax.set_ylabel('Agent Count')
        ax.set_title(f'Agent Info Energy Distribution\n(Range: {min_energy:.0f} - {max_energy:.0f})')
        ax.legend(loc='upper right', fontsize=8)

    def _plot_free_energy(self, ax: plt.Axes):
        """
        Plot free energy landscape.

        Free Energy = Entropy - log(Coherence)
        Lower free energy = more organized system
        """
        steps = range(len(self.semantic_entropy_history))

        entropy = np.array(list(self.semantic_entropy_history))
        coherence = np.array(list(self.coherence_history))

        # Compute free energy (simplified Helmholtz)
        # F = E - TS, approximated as F = entropy - log(coherence)
        free_energy = entropy - np.log(coherence + 1e-10)

        # Smooth for visualization
        if len(free_energy) >= self.config.smoothing_window:
            kernel = np.ones(self.config.smoothing_window) / self.config.smoothing_window
            free_energy_smooth = np.convolve(free_energy, kernel, mode='valid')
            steps_smooth = range(self.config.smoothing_window - 1, len(free_energy))
        else:
            free_energy_smooth = free_energy
            steps_smooth = steps

        ax.plot(steps, free_energy, 'b-', alpha=0.3, linewidth=1)
        ax.plot(steps_smooth, free_energy_smooth, 'b-', linewidth=2,
               label='Free Energy (smoothed)')

        # Trend line
        if len(free_energy) > 10:
            z = np.polyfit(range(len(free_energy)), free_energy, 1)
            trend = np.poly1d(z)
            ax.plot(steps, trend(list(steps)), 'r--', linewidth=2,
                   label=f'Trend (slope: {z[0]:.4f})')

        ax.set_xlabel('Step')
        ax.set_ylabel('Free Energy')
        ax.set_title('Free Energy Landscape\n(Lower = More Organized)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_temperature_dynamics(self, ax: plt.Axes):
        """Plot temperature (simulated annealing) dynamics"""
        steps = range(len(self.temperature_history))
        temp = list(self.temperature_history)

        ax.plot(steps, temp, color='red', linewidth=2)
        ax.fill_between(steps, temp, color='red', alpha=0.3)

        # Phase transition markers
        if len(temp) > 1:
            # Detect rapid changes
            temp_diff = np.abs(np.diff(temp))
            threshold = np.mean(temp_diff) + 2 * np.std(temp_diff)
            transitions = np.where(temp_diff > threshold)[0]

            for t in transitions:
                ax.axvline(t, color='purple', linestyle=':', alpha=0.5)

        ax.set_xlabel('Step')
        ax.set_ylabel('Temperature')
        ax.set_title('System Temperature\n(Exploration vs Exploitation)')
        ax.grid(True, alpha=0.3)


def plot_info_budget(env: 'MycoBeaverEnv',
                     visualizer: Optional[ThermodynamicsVisualizer] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot info energy budget.

    Quick diagnostic: Is the colony info-sustainable?
    """
    if visualizer is None:
        visualizer = ThermodynamicsVisualizer()

    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer._plot_info_budget(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_entropy_dynamics(env: 'MycoBeaverEnv',
                          visualizer: Optional[ThermodynamicsVisualizer] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot entropy dynamics.

    Quick diagnostic: Is the system converging or fragmenting?
    """
    if visualizer is None:
        visualizer = ThermodynamicsVisualizer()

    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer._plot_entropy_dynamics(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_infrastructure_roi(env: 'MycoBeaverEnv',
                            visualizer: Optional[ThermodynamicsVisualizer] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot infrastructure ROI.

    Quick diagnostic: Is building infrastructure worth the info cost?
    """
    if visualizer is None:
        visualizer = ThermodynamicsVisualizer()

    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer._plot_infrastructure_roi(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig
