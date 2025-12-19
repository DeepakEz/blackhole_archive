"""
Unified Live Dashboard
======================

PHASE 4: This lets you interrogate the colony.

Provides a comprehensive real-time dashboard using Streamlit or Dash.
Shows all visualization components in a unified interface:
- Live grid with water/dams/physarum
- Live semantic graph
- Scalar metrics panel
- Overmind signals
- Time-series plots

Usage:
    # Run from parent directory
    cd C:\\
    streamlit run mycobeaver/visualization/dashboard.py

    # Or integrate programmatically
    from mycobeaver.visualization.dashboard import LiveDashboard
    dashboard = LiveDashboard(env)
    dashboard.run()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
import threading
import time
import io
import base64
import sys
import os

# Fix imports for both module and direct execution
if __name__ == "__main__" or "streamlit" in sys.modules:
    # Running directly or via streamlit - add parent to path
    _viz_dir = os.path.dirname(os.path.abspath(__file__))
    _mycobeaver_dir = os.path.dirname(_viz_dir)
    _parent_dir = os.path.dirname(_mycobeaver_dir)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)

    from mycobeaver.visualization.grid import GridVisualizer
    from mycobeaver.visualization.networks import SemanticGraphVisualizer
    from mycobeaver.visualization.cognition import CommunicationVisualizer
    from mycobeaver.visualization.thermodynamics import ThermodynamicsVisualizer
else:
    # Running as module import
    from .grid import GridVisualizer
    from .networks import SemanticGraphVisualizer
    from .cognition import CommunicationVisualizer
    from .thermodynamics import ThermodynamicsVisualizer

if TYPE_CHECKING:
    from ..environment import MycoBeaverEnv

# Try to import Streamlit
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None


@dataclass
class DashboardConfig:
    """Configuration for the live dashboard"""
    title: str = "MycoBeaver Colony Dashboard"
    update_interval: float = 0.5  # seconds
    grid_height: int = 400
    show_grid: bool = True
    show_semantic: bool = True
    show_metrics: bool = True
    show_thermodynamics: bool = True
    show_overmind: bool = True
    auto_refresh: bool = True


class LiveDashboard:
    """
    Unified live dashboard for interrogating the colony.

    Combines all visualization components into a single interface
    with real-time updates.
    """

    def __init__(self, env: 'MycoBeaverEnv',
                 config: Optional[DashboardConfig] = None):
        self.env = env
        self.config = config or DashboardConfig()

        # Initialize visualizers
        self.grid_viz = GridVisualizer()
        self.semantic_viz = None  # Lazy init (requires networkx)
        self.comm_viz = CommunicationVisualizer()
        self.thermo_viz = ThermodynamicsVisualizer()

        # State
        self.is_running = False
        self.step_count = 0

    def _init_semantic_viz(self):
        """Lazily initialize semantic visualizer"""
        if self.semantic_viz is None:
            try:
                self.semantic_viz = SemanticGraphVisualizer()
            except ImportError:
                pass  # networkx not available

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get current metrics as a dictionary"""
        env = self.env
        n_alive = sum(1 for a in env.agents if a.alive)

        metrics = {
            "step": env.current_step,
            "n_alive": n_alive,
            "n_agents_total": len(env.agents),
            "survival_rate": n_alive / max(1, len(env.agents)),
            "avg_water": float(np.mean(env.grid_state.water_depth)),
            "max_water": float(np.max(env.grid_state.water_depth)),
            "total_vegetation": float(np.sum(env.grid_state.vegetation)),
            "n_structures": int(np.sum(env.grid_state.dam_permeability < 0.9)),
            "episode_reward": env.episode_reward,
        }

        # Agent info energy
        if n_alive > 0:
            info_energies = [a.info_energy for a in env.agents if a.alive]
            metrics["avg_info_energy"] = float(np.mean(info_energies))
            metrics["min_info_energy"] = float(np.min(info_energies))
        else:
            metrics["avg_info_energy"] = 0.0
            metrics["min_info_energy"] = 0.0

        # Semantic metrics
        if env.semantic_system is not None:
            graph = env.semantic_system.shared_graph
            metrics["semantic_vertices"] = len(graph.vertices)
            metrics["semantic_edges"] = len(graph.edges)
            metrics["semantic_entropy"] = float(graph.compute_semantic_entropy())
            metrics["coherence"] = float(graph.compute_coherence())
            metrics["contradictions"] = len(graph.contradictions)

        # Overmind metrics
        if env.overmind is not None:
            signals = env.overmind.get_current_signals()
            metrics["overmind_lr_scale"] = signals.get("lr_scale", 1.0)
            metrics["overmind_entropy_scale"] = signals.get("entropy_scale", 1.0)
            metrics["overmind_comm_budget"] = signals.get("comm_budget", 1.0)

        # Time-scale stats
        if hasattr(env, '_physarum_update_count'):
            metrics["physarum_updates"] = env._physarum_update_count
            metrics["overmind_updates"] = env._overmind_update_count
            metrics["semantic_consolidations"] = env._semantic_consolidation_count

        return metrics

    def get_overmind_signals(self) -> Dict[str, float]:
        """Get current Overmind signals"""
        if self.env.overmind is None:
            return {}
        return self.env.overmind.get_current_signals()

    def fig_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64 for web display"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str

    def render_grid_image(self) -> np.ndarray:
        """Render grid as RGB image"""
        return self.env._render_rgb()

    def record_step(self, policy_entropy: float = 0.0):
        """Record current step data in all visualizers"""
        self.grid_viz.capture_state(self.env)
        self.comm_viz.record_step(self.env)
        self.thermo_viz.record_step(self.env, policy_entropy)
        self.step_count += 1


def create_dashboard_app():
    """
    Create and run the Streamlit dashboard application.

    Run with: streamlit run mycobeaver/visualization/dashboard.py
    """
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Install with: pip install streamlit")
        print("Then run: streamlit run mycobeaver/visualization/dashboard.py")
        return

    st.set_page_config(
        page_title="MycoBeaver Colony Dashboard",
        page_icon="🦫",
        layout="wide"
    )

    st.title("🦫 MycoBeaver Colony Dashboard")
    st.markdown("*Real-time visualization of the distributed beaver colony simulation*")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Check if environment is available in session state
    if 'env' not in st.session_state:
        st.warning("No environment loaded. Please initialize the environment first.")
        st.code("""
# Example: Initialize environment
from mycobeaver.config import SimulationConfig
from mycobeaver.environment import MycoBeaverEnv

config = SimulationConfig()
env = MycoBeaverEnv(config)
env.reset()

# Then pass to dashboard
import streamlit as st
st.session_state.env = env
        """)
        return

    env = st.session_state.env
    dashboard = LiveDashboard(env)

    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh Rate (s)", 0.1, 2.0, 0.5)

    # Panels selection
    st.sidebar.header("Panels")
    show_grid = st.sidebar.checkbox("Grid View", value=True)
    show_metrics = st.sidebar.checkbox("Metrics", value=True)
    show_semantic = st.sidebar.checkbox("Semantic Graph", value=True)
    show_thermo = st.sidebar.checkbox("Thermodynamics", value=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        if show_grid:
            st.subheader("🗺️ Environment Grid")
            grid_img = dashboard.render_grid_image()
            st.image(grid_img, caption=f"Step {env.current_step}", use_column_width=True)

        if show_semantic:
            st.subheader("🧠 Semantic Knowledge Graph")
            if env.semantic_system is not None:
                try:
                    dashboard._init_semantic_viz()
                    if dashboard.semantic_viz is not None:
                        fig = dashboard.semantic_viz.plot_knowledge_graph(
                            env.semantic_system.shared_graph
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Error rendering semantic graph: {e}")
            else:
                st.info("Semantic system not initialized")

    with col2:
        if show_metrics:
            st.subheader("📊 Colony Metrics")
            metrics = dashboard.get_metrics_dict()

            # Key metrics in columns
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Step", metrics["step"])
                st.metric("Alive Agents", f"{metrics['n_alive']}/{metrics['n_agents_total']}")
                st.metric("Structures", metrics["n_structures"])

            with m2:
                st.metric("Avg Info Energy", f"{metrics.get('avg_info_energy', 0):.1f}")
                st.metric("Coherence", f"{metrics.get('coherence', 1.0):.2f}")
                st.metric("Contradictions", metrics.get("contradictions", 0))

            # Overmind signals
            st.subheader("🔮 Overmind Signals")
            signals = dashboard.get_overmind_signals()
            if signals:
                for key, value in signals.items():
                    st.progress(min(1.0, value), text=f"{key}: {value:.3f}")
            else:
                st.info("Overmind not active")

    if show_thermo:
        st.subheader("🌡️ Information Thermodynamics")
        # Record current step for visualization
        dashboard.record_step()

        thermo_col1, thermo_col2 = st.columns(2)

        with thermo_col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            dashboard.thermo_viz._plot_info_budget(ax)
            st.pyplot(fig)
            plt.close(fig)

        with thermo_col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            dashboard.thermo_viz._plot_entropy_dynamics(ax)
            st.pyplot(fig)
            plt.close(fig)

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


def create_minimal_dashboard(env: 'MycoBeaverEnv') -> Dict[str, Any]:
    """
    Create a minimal dashboard output suitable for non-Streamlit environments.

    Returns a dictionary with all visualization data that can be
    displayed in any framework (Jupyter, Flask, etc.)
    """
    dashboard = LiveDashboard(env)
    dashboard.record_step()

    output = {
        "metrics": dashboard.get_metrics_dict(),
        "overmind_signals": dashboard.get_overmind_signals(),
        "grid_rgb": dashboard.render_grid_image(),
    }

    # Generate figures
    try:
        fig = dashboard.grid_viz.plot_overview(env)
        output["grid_fig_b64"] = dashboard.fig_to_base64(fig)
    except Exception:
        output["grid_fig_b64"] = None

    if env.semantic_system is not None:
        try:
            dashboard._init_semantic_viz()
            if dashboard.semantic_viz is not None:
                fig = dashboard.semantic_viz.plot_knowledge_graph(
                    env.semantic_system.shared_graph
                )
                output["semantic_fig_b64"] = dashboard.fig_to_base64(fig)
        except Exception:
            output["semantic_fig_b64"] = None

    return output


# Jupyter-friendly display function
def display_dashboard(env: 'MycoBeaverEnv', show_all: bool = True):
    """
    Display dashboard in Jupyter notebook.

    Usage:
        from mycobeaver.visualization.dashboard import display_dashboard
        display_dashboard(env)
    """
    dashboard = LiveDashboard(env)
    dashboard.record_step()

    # Print metrics
    metrics = dashboard.get_metrics_dict()
    print("=" * 50)
    print(f"MycoBeaver Colony - Step {metrics['step']}")
    print("=" * 50)
    print(f"Agents: {metrics['n_alive']}/{metrics['n_agents_total']} alive")
    print(f"Structures: {metrics['n_structures']}")
    print(f"Avg Info Energy: {metrics.get('avg_info_energy', 0):.1f}")
    print(f"Coherence: {metrics.get('coherence', 1.0):.3f}")
    print(f"Episode Reward: {metrics['episode_reward']:.2f}")
    print("=" * 50)

    # Overmind signals
    signals = dashboard.get_overmind_signals()
    if signals:
        print("\nOvermind Signals:")
        for key, value in signals.items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {key}: [{bar}] {value:.3f}")

    if show_all:
        # Show grid overview
        fig = dashboard.grid_viz.plot_overview(env)
        plt.show()

        # Show semantic graph
        if env.semantic_system is not None:
            try:
                dashboard._init_semantic_viz()
                if dashboard.semantic_viz is not None:
                    fig = dashboard.semantic_viz.plot_knowledge_graph(
                        env.semantic_system.shared_graph
                    )
                    plt.show()
            except Exception as e:
                print(f"Could not display semantic graph: {e}")

        # Show thermodynamics
        fig = dashboard.thermo_viz.plot_dashboard(env)
        plt.show()


# Entry point for Streamlit
if __name__ == "__main__":
    create_dashboard_app()
