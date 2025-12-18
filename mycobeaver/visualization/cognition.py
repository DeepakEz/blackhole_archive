"""
Communication & Consensus Visualization
========================================

PHASE 4: Visualization of distributed cognition and consensus formation.

This tells you whether:
- Consensus is real
- Or just noise alignment

Shows:
- Number of messages over time
- Vector clock skew
- Quorum formation times
- Failed vs successful projects
- Communication network topology
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict

if TYPE_CHECKING:
    from ..environment import MycoBeaverEnv
    from ..communication import CommunicationHub
    from ..projects import ProjectManager


@dataclass
class CognitionVisualizerConfig:
    """Configuration for communication/consensus visualization"""
    figsize: Tuple[int, int] = (14, 10)
    dpi: int = 100

    # Colors
    message_color: str = "#4169E1"  # Royal blue
    broadcast_color: str = "#32CD32"  # Lime green
    quorum_success_color: str = "#228B22"  # Forest green
    quorum_fail_color: str = "#DC143C"  # Crimson

    # Project colors
    project_colors: Dict[str, str] = field(default_factory=lambda: {
        "proposed": "#FFD700",   # Gold
        "active": "#4169E1",     # Royal blue
        "completed": "#228B22",  # Forest green
        "abandoned": "#DC143C",  # Crimson
    })

    # Time window for rolling statistics
    rolling_window: int = 50


class CommunicationVisualizer:
    """
    Visualization tools for communication patterns and consensus formation.

    Tracks and displays:
    - Message flow over time
    - Consensus formation dynamics
    - Project lifecycle and success rates
    - Communication network topology
    """

    def __init__(self, config: Optional[CognitionVisualizerConfig] = None):
        self.config = config or CognitionVisualizerConfig()

        # History tracking
        self.message_counts: List[int] = []
        self.broadcast_counts: List[int] = []
        self.quorum_attempts: List[int] = []
        self.quorum_successes: List[int] = []

        self.project_proposed: List[int] = []
        self.project_active: List[int] = []
        self.project_completed: List[int] = []
        self.project_abandoned: List[int] = []

        # Vector clock tracking
        self.vector_clock_skews: List[float] = []

        # Communication topology (for network analysis)
        self.communication_matrix: Optional[np.ndarray] = None

    def record_step(self, env: 'MycoBeaverEnv'):
        """Record communication statistics for current step"""
        # Message counts (if communication hub available)
        if hasattr(env, 'communication_hub') and env.communication_hub is not None:
            hub = env.communication_hub
            stats = hub.get_statistics()
            self.message_counts.append(stats.get('messages_this_step', 0))
            self.broadcast_counts.append(stats.get('broadcasts_this_step', 0))
            self.quorum_attempts.append(stats.get('quorum_attempts', 0))
            self.quorum_successes.append(stats.get('quorum_successes', 0))

            # Vector clock skew
            skew = self._compute_vector_clock_skew(hub)
            self.vector_clock_skews.append(skew)

            # Communication matrix
            if self.communication_matrix is None:
                n = len(env.agents)
                self.communication_matrix = np.zeros((n, n))
            # Update matrix based on recent communications
            self._update_communication_matrix(hub, env)
        else:
            # Placeholder values
            self.message_counts.append(0)
            self.broadcast_counts.append(0)
            self.quorum_attempts.append(0)
            self.quorum_successes.append(0)
            self.vector_clock_skews.append(0)

        # Project statistics
        if env.project_manager is not None:
            pm = env.project_manager
            self.project_proposed.append(pm.proposed_count if hasattr(pm, 'proposed_count') else 0)
            self.project_active.append(len(pm.active_projects))
            self.project_completed.append(pm.completed_count if hasattr(pm, 'completed_count') else 0)
            self.project_abandoned.append(pm.abandoned_count if hasattr(pm, 'abandoned_count') else 0)
        else:
            self.project_proposed.append(0)
            self.project_active.append(0)
            self.project_completed.append(0)
            self.project_abandoned.append(0)

    def _compute_vector_clock_skew(self, hub: 'CommunicationHub') -> float:
        """Compute the skew (variance) in vector clocks across agents"""
        if not hasattr(hub, 'vector_clocks'):
            return 0.0

        clocks = hub.vector_clocks
        if not clocks:
            return 0.0

        # Get maximum component from each agent's clock
        max_components = []
        for agent_id, clock in clocks.items():
            if clock:
                max_components.append(max(clock.values()))

        if len(max_components) < 2:
            return 0.0

        return np.std(max_components)

    def _update_communication_matrix(self, hub: 'CommunicationHub', env: 'MycoBeaverEnv'):
        """Update communication adjacency matrix"""
        if not hasattr(hub, 'recent_messages'):
            return

        for msg in hub.recent_messages:
            sender = msg.get('sender', -1)
            receiver = msg.get('receiver', -1)
            if 0 <= sender < self.communication_matrix.shape[0] and \
               0 <= receiver < self.communication_matrix.shape[1]:
                self.communication_matrix[sender, receiver] += 1

    def plot_overview(self, env: 'MycoBeaverEnv') -> plt.Figure:
        """
        Create overview of communication and consensus patterns.

        4-panel view:
        1. Message flow over time
        2. Consensus (quorum) formation
        3. Project lifecycle
        4. Communication network heatmap
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)

        step = env.current_step
        fig.suptitle(f'Communication & Consensus Dashboard - Step {step}', fontsize=14)

        # Panel 1: Message flow
        self._plot_message_flow(axes[0, 0])

        # Panel 2: Consensus formation
        self._plot_consensus_formation(axes[0, 1])

        # Panel 3: Project lifecycle
        self._plot_project_lifecycle(axes[1, 0])

        # Panel 4: Communication network
        self._plot_communication_network(axes[1, 1], env)

        plt.tight_layout()
        return fig

    def _plot_message_flow(self, ax: plt.Axes):
        """Plot message counts over time"""
        steps = range(len(self.message_counts))

        ax.plot(steps, self.message_counts, color=self.config.message_color,
                label='Direct Messages', linewidth=2, alpha=0.8)
        ax.plot(steps, self.broadcast_counts, color=self.config.broadcast_color,
                label='Broadcasts', linewidth=2, alpha=0.8)

        # Rolling average
        if len(self.message_counts) >= self.config.rolling_window:
            rolling_msg = np.convolve(self.message_counts,
                                      np.ones(self.config.rolling_window)/self.config.rolling_window,
                                      mode='valid')
            ax.plot(range(self.config.rolling_window-1, len(self.message_counts)),
                   rolling_msg, color=self.config.message_color,
                   linestyle='--', alpha=0.5, label='Msg Rolling Avg')

        ax.set_xlabel('Step')
        ax.set_ylabel('Count')
        ax.set_title('Message Flow Over Time')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_consensus_formation(self, ax: plt.Axes):
        """Plot quorum/consensus attempts and successes"""
        steps = range(len(self.quorum_attempts))

        # Cumulative quorum statistics
        cum_attempts = np.cumsum(self.quorum_attempts)
        cum_successes = np.cumsum(self.quorum_successes)

        ax.fill_between(steps, cum_attempts, color=self.config.quorum_fail_color,
                       alpha=0.3, label='Total Attempts')
        ax.fill_between(steps, cum_successes, color=self.config.quorum_success_color,
                       alpha=0.5, label='Successes')

        ax.plot(steps, cum_attempts, color=self.config.quorum_fail_color, linewidth=2)
        ax.plot(steps, cum_successes, color=self.config.quorum_success_color, linewidth=2)

        # Success rate annotation
        total_attempts = sum(self.quorum_attempts)
        total_successes = sum(self.quorum_successes)
        if total_attempts > 0:
            rate = 100 * total_successes / total_attempts
            ax.text(0.95, 0.05, f'Success Rate: {rate:.1f}%',
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Count')
        ax.set_title('Consensus Formation\n(Quorum Attempts vs Successes)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_project_lifecycle(self, ax: plt.Axes):
        """Plot project status over time"""
        steps = range(len(self.project_active))

        # Stacked area for project states
        ax.fill_between(steps, self.project_completed,
                       color=self.config.project_colors['completed'],
                       alpha=0.7, label='Completed')

        active_plus_completed = np.array(self.project_active) + np.array(self.project_completed)
        ax.fill_between(steps, self.project_completed, active_plus_completed,
                       color=self.config.project_colors['active'],
                       alpha=0.7, label='Active')

        all_projects = active_plus_completed + np.array(self.project_abandoned)
        ax.fill_between(steps, active_plus_completed, all_projects,
                       color=self.config.project_colors['abandoned'],
                       alpha=0.7, label='Abandoned')

        ax.set_xlabel('Step')
        ax.set_ylabel('Project Count')
        ax.set_title('Project Lifecycle')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_communication_network(self, ax: plt.Axes, env: 'MycoBeaverEnv'):
        """Plot communication network as heatmap"""
        if self.communication_matrix is not None and np.sum(self.communication_matrix) > 0:
            # Normalize
            norm_matrix = self.communication_matrix / (np.max(self.communication_matrix) + 1e-10)

            im = ax.imshow(norm_matrix, cmap='YlOrRd', aspect='auto')
            plt.colorbar(im, ax=ax, label='Communication Intensity')

            ax.set_xlabel('Receiver Agent')
            ax.set_ylabel('Sender Agent')
            ax.set_title('Communication Network\n(Who talks to whom)')

            # Add agent count
            n_agents = self.communication_matrix.shape[0]
            ax.set_xticks(range(0, n_agents, max(1, n_agents//10)))
            ax.set_yticks(range(0, n_agents, max(1, n_agents//10)))
        else:
            ax.text(0.5, 0.5, 'No communication data yet',
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Communication Network')

    def plot_vector_clock_analysis(self) -> plt.Figure:
        """
        Detailed analysis of vector clock synchronization.

        Shows:
        - Clock skew over time
        - Synchronization events
        - Causality violations (if any)
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=self.config.dpi)

        steps = range(len(self.vector_clock_skews))

        # Top: Skew over time
        ax1 = axes[0]
        ax1.plot(steps, self.vector_clock_skews, 'b-', linewidth=2)
        ax1.fill_between(steps, self.vector_clock_skews, alpha=0.3)
        ax1.axhline(y=np.mean(self.vector_clock_skews) if self.vector_clock_skews else 0,
                   color='red', linestyle='--', label='Mean Skew')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Vector Clock Skew (Std Dev)')
        ax1.set_title('Vector Clock Synchronization Over Time\n(Lower = Better Synchronized)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Skew distribution
        ax2 = axes[1]
        if self.vector_clock_skews:
            ax2.hist(self.vector_clock_skews, bins=30, color='steelblue',
                    edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(self.vector_clock_skews), color='red',
                       linestyle='--', label=f'Mean: {np.mean(self.vector_clock_skews):.2f}')
            ax2.axvline(np.median(self.vector_clock_skews), color='green',
                       linestyle='--', label=f'Median: {np.median(self.vector_clock_skews):.2f}')
            ax2.set_xlabel('Skew Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Vector Clock Skew')
            ax2.legend()

        plt.tight_layout()
        return fig


def plot_message_flow(env: 'MycoBeaverEnv',
                      visualizer: Optional[CommunicationVisualizer] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot message flow.

    Quick diagnostic: Is communication happening?
    """
    if visualizer is None:
        visualizer = CommunicationVisualizer()

    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer._plot_message_flow(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_consensus_formation(env: 'MycoBeaverEnv',
                             visualizer: Optional[CommunicationVisualizer] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot consensus formation.

    Quick diagnostic: Is consensus forming or failing?
    """
    if visualizer is None:
        visualizer = CommunicationVisualizer()

    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer._plot_consensus_formation(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_project_lifecycle(env: 'MycoBeaverEnv',
                           visualizer: Optional[CommunicationVisualizer] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot project lifecycle.

    Quick diagnostic: Are projects succeeding or failing?
    """
    if visualizer is None:
        visualizer = CommunicationVisualizer()

    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer._plot_project_lifecycle(ax)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig
