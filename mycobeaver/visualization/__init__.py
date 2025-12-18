"""
MycoBeaver Visualization Layer
==============================

PHASE 4: Advanced Visualizations - Debugging Instruments

These are not eye candy — they are debugging instruments that let you
interrogate the colony and understand emergent behavior.

Modules:
- grid: Environment & Infrastructure View (water, dams, physarum)
- networks: Cognitive Network View (semantic graph, entropy, contradictions)
- cognition: Communication & Consensus View (messages, quorum, projects)
- thermodynamics: Information Thermodynamics Dashboard (info energy, entropy, ROI)
- dashboard: Unified Live Dashboard (Streamlit/Dash integration)
"""

from .grid import (
    GridVisualizer,
    plot_water_flow,
    plot_infrastructure,
    plot_physarum_network,
    animate_simulation,
)

from .networks import (
    SemanticGraphVisualizer,
    plot_knowledge_graph,
    plot_entropy_heatmap,
    plot_contradiction_clusters,
)

from .cognition import (
    CommunicationVisualizer,
    plot_message_flow,
    plot_consensus_formation,
    plot_project_lifecycle,
)

from .thermodynamics import (
    ThermodynamicsVisualizer,
    plot_info_budget,
    plot_entropy_dynamics,
    plot_infrastructure_roi,
)

from .dashboard import (
    LiveDashboard,
    create_dashboard_app,
)

__all__ = [
    # Grid
    "GridVisualizer",
    "plot_water_flow",
    "plot_infrastructure",
    "plot_physarum_network",
    "animate_simulation",
    # Networks
    "SemanticGraphVisualizer",
    "plot_knowledge_graph",
    "plot_entropy_heatmap",
    "plot_contradiction_clusters",
    # Cognition
    "CommunicationVisualizer",
    "plot_message_flow",
    "plot_consensus_formation",
    "plot_project_lifecycle",
    # Thermodynamics
    "ThermodynamicsVisualizer",
    "plot_info_budget",
    "plot_entropy_dynamics",
    "plot_infrastructure_roi",
    # Dashboard
    "LiveDashboard",
    "create_dashboard_app",
]
