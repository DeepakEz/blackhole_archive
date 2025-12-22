"""
Cognitive Network Visualization - Semantic Graph View
======================================================

PHASE 4: Visualization of the semantic/cognitive layer.

This is VERY IMPORTANT - it answers:
"Is knowledge converging or fragmenting?"

Shows:
- Semantic graph nodes (beliefs, goals, observations)
- Edge weights (confidence/pheromone)
- Entropy per node
- Contradiction clusters
- Knowledge flow patterns

Tools: networkx + matplotlib (optional: pygraphviz for better layouts)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from typing import Optional, List, Tuple, Dict, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict

# Try to import networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

if TYPE_CHECKING:
    from ..semantic import SemanticGraph, ColonySemanticSystem, Vertex, Edge
    from ..environment import MycoBeaverEnv


@dataclass
class NetworkVisualizerConfig:
    """Configuration for semantic network visualization"""
    figsize: Tuple[int, int] = (14, 10)
    dpi: int = 100

    # Node styling by type
    node_colors: Dict[str, str] = field(default_factory=lambda: {
        "observation": "#87CEEB",   # Sky blue
        "belief": "#90EE90",        # Light green
        "goal": "#FFD700",          # Gold
        "plan": "#DDA0DD",          # Plum
        "contradiction": "#FF6B6B", # Light red
    })
    node_size_base: int = 300
    node_size_scale: float = 500  # Scale by confidence

    # Edge styling
    edge_cmap: str = "YlOrRd"  # Yellow-Orange-Red for pheromone
    edge_width_min: float = 0.5
    edge_width_max: float = 4.0
    edge_alpha: float = 0.6

    # Layout
    layout_algorithm: str = "spring"  # spring, kamada_kawai, spectral
    layout_k: float = 2.0  # Spring constant for spring layout
    layout_iterations: int = 50

    # Entropy visualization
    entropy_cmap: str = "RdYlGn_r"  # Red=high entropy, Green=low

    # Labels
    show_labels: bool = True
    label_font_size: int = 8


class SemanticGraphVisualizer:
    """
    Comprehensive visualization for semantic knowledge graphs.

    Provides insights into:
    - Knowledge structure and connectivity
    - Belief coherence and contradictions
    - Information flow patterns
    - Entropy distribution
    """

    def __init__(self, config: Optional[NetworkVisualizerConfig] = None):
        self.config = config or NetworkVisualizerConfig()

        if not HAS_NETWORKX:
            raise ImportError("networkx is required for semantic graph visualization. "
                            "Install with: pip install networkx")

        # History for tracking evolution
        self.entropy_history: List[float] = []
        self.coherence_history: List[float] = []
        self.n_vertices_history: List[int] = []
        self.n_contradictions_history: List[int] = []

    def build_networkx_graph(self, semantic_graph: 'SemanticGraph') -> 'nx.DiGraph':
        """Convert SemanticGraph to NetworkX graph for visualization"""
        G = nx.DiGraph()

        # Add nodes
        for vid, vertex in semantic_graph.vertices.items():
            G.add_node(vid,
                      vertex_type=vertex.vertex_type.value,
                      content=str(vertex.content)[:50],  # Truncate for display
                      confidence=vertex.confidence,
                      activation=vertex.activation,
                      position=vertex.position)

        # Add edges (using correct attribute names: from_vertex, to_vertex)
        for eid, edge in semantic_graph.edges.items():
            G.add_edge(edge.from_vertex, edge.to_vertex,
                      edge_type=edge.edge_type.value,
                      weight=edge.weight,
                      pheromone=edge.pheromone,
                      confidence=edge.confidence)

        return G

    def plot_knowledge_graph(self, semantic_graph: 'SemanticGraph',
                             title: Optional[str] = None,
                             highlight_contradictions: bool = True) -> plt.Figure:
        """
        Main visualization of the semantic knowledge graph.

        Shows:
        - Nodes colored by type, sized by confidence
        - Edges colored by pheromone level, weighted by strength
        - Contradiction clusters highlighted
        """
        G = self.build_networkx_graph(semantic_graph)

        if len(G.nodes) == 0:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            ax.text(0.5, 0.5, "No vertices in semantic graph",
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Compute layout
        pos = self._compute_layout(G)

        # Get node attributes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            vtype = G.nodes[node].get('vertex_type', 'observation')
            conf = G.nodes[node].get('confidence', 0.5)
            node_colors.append(self.config.node_colors.get(vtype, '#808080'))
            node_sizes.append(self.config.node_size_base +
                            self.config.node_size_scale * conf)

        # Get edge attributes
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            pheromone = G.edges[u, v].get('pheromone', 0.1)
            weight = G.edges[u, v].get('weight', 0.5)
            edge_colors.append(pheromone)
            edge_widths.append(self.config.edge_width_min +
                              (self.config.edge_width_max - self.config.edge_width_min) * weight)

        # Draw edges
        if len(G.edges) > 0:
            edges = nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=edge_colors,
                edge_cmap=plt.cm.get_cmap(self.config.edge_cmap),
                width=edge_widths,
                alpha=self.config.edge_alpha,
                arrows=True,
                arrowsize=10,
                connectionstyle="arc3,rad=0.1"
            )

        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=1
        )

        # Highlight contradictions
        if highlight_contradictions and semantic_graph.contradictions:
            contradiction_nodes = set()
            for v1, v2 in semantic_graph.contradictions:
                contradiction_nodes.add(v1)
                contradiction_nodes.add(v2)

            if contradiction_nodes:
                contra_pos = {n: pos[n] for n in contradiction_nodes if n in pos}
                nx.draw_networkx_nodes(
                    G.subgraph(contradiction_nodes), contra_pos, ax=ax,
                    node_color='none',
                    node_size=[s * 1.5 for n, s in zip(G.nodes(), node_sizes)
                              if n in contradiction_nodes],
                    edgecolors='red',
                    linewidths=3,
                    alpha=0.8
                )

        # Labels
        if self.config.show_labels:
            labels = {n: f"{n}" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax,
                                   font_size=self.config.label_font_size)

        # Legend
        legend_elements = [
            mpatches.Patch(color=color, label=vtype.capitalize())
            for vtype, color in self.config.node_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        # Title and stats
        if title is None:
            title = "Semantic Knowledge Graph"
        stats = (f"Vertices: {len(G.nodes)}, Edges: {len(G.edges)}, "
                f"Contradictions: {len(semantic_graph.contradictions)}")
        ax.set_title(f"{title}\n{stats}", fontsize=12)

        ax.axis('off')
        return fig

    def _compute_layout(self, G: 'nx.Graph') -> Dict[int, Tuple[float, float]]:
        """Compute node positions using specified layout algorithm"""
        if len(G.nodes) == 0:
            return {}

        if self.config.layout_algorithm == "spring":
            return nx.spring_layout(G,
                                   k=self.config.layout_k,
                                   iterations=self.config.layout_iterations)
        elif self.config.layout_algorithm == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(G)
            except (nx.NetworkXError, ValueError, np.linalg.LinAlgError) as e:
                # Fall back to spring layout if kamada_kawai fails
                # (can fail on disconnected graphs or numerical issues)
                return nx.spring_layout(G)
        elif self.config.layout_algorithm == "spectral":
            try:
                return nx.spectral_layout(G)
            except (nx.NetworkXError, ValueError, np.linalg.LinAlgError) as e:
                # Fall back to spring layout if spectral fails
                # (can fail on graphs without enough structure)
                return nx.spring_layout(G)
        else:
            return nx.spring_layout(G)

    def plot_entropy_heatmap(self, semantic_graph: 'SemanticGraph') -> plt.Figure:
        """
        Visualize entropy distribution across the graph.

        High entropy regions indicate uncertainty/incoherence.
        """
        G = self.build_networkx_graph(semantic_graph)

        if len(G.nodes) == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No vertices for entropy analysis",
                   ha='center', va='center')
            return fig

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=self.config.dpi)

        # Compute local entropy for each node
        local_entropies = self._compute_local_entropies(semantic_graph)

        # Left: Graph with entropy coloring
        ax1 = axes[0]
        pos = self._compute_layout(G)

        node_colors = [local_entropies.get(n, 0) for n in G.nodes()]
        node_sizes = [self.config.node_size_base +
                     self.config.node_size_scale * G.nodes[n].get('confidence', 0.5)
                     for n in G.nodes()]

        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax1,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.get_cmap(self.config.entropy_cmap),
            alpha=0.9,
            edgecolors='black',
            linewidths=1
        )

        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, arrows=True)

        if self.config.show_labels:
            nx.draw_networkx_labels(G, pos, ax=ax1,
                                   font_size=self.config.label_font_size)

        plt.colorbar(nodes, ax=ax1, label='Local Entropy')
        ax1.set_title('Entropy Distribution\n(Red=High, Green=Low)')
        ax1.axis('off')

        # Right: Entropy histogram
        ax2 = axes[1]
        if local_entropies:
            entropies = list(local_entropies.values())
            ax2.hist(entropies, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(entropies), color='red', linestyle='--',
                       label=f'Mean: {np.mean(entropies):.3f}')
            ax2.axvline(np.median(entropies), color='green', linestyle='--',
                       label=f'Median: {np.median(entropies):.3f}')
            ax2.set_xlabel('Local Entropy')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Entropy Distribution Histogram')
            ax2.legend()

        plt.tight_layout()
        return fig

    def _compute_local_entropies(self, semantic_graph: 'SemanticGraph') -> Dict[int, float]:
        """Compute local entropy for each vertex based on neighbor confidence distribution"""
        entropies = {}

        for vid, vertex in semantic_graph.vertices.items():
            # Get neighbor confidences
            neighbors = semantic_graph.get_neighbors(vid)
            if not neighbors:
                entropies[vid] = 0.0
                continue

            # Compute entropy from confidence distribution
            confidences = []
            for nid, eid in neighbors:
                if nid in semantic_graph.vertices:
                    confidences.append(semantic_graph.vertices[nid].confidence)
                edge = semantic_graph.edges.get(eid)
                if edge:
                    confidences.append(edge.confidence)

            if confidences:
                # Normalize to probability distribution
                probs = np.array(confidences) / (np.sum(confidences) + 1e-10)
                probs = probs[probs > 0]  # Remove zeros for log
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies[vid] = entropy
            else:
                entropies[vid] = 0.0

        return entropies

    def plot_contradiction_clusters(self, semantic_graph: 'SemanticGraph') -> plt.Figure:
        """
        Visualize contradiction clusters in the knowledge graph.

        Shows which beliefs are in conflict and their relationship structure.
        """
        G = self.build_networkx_graph(semantic_graph)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=self.config.dpi)

        # Left: Full graph with contradictions highlighted
        ax1 = axes[0]
        pos = self._compute_layout(G)

        # Draw base graph faded
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightgray',
                              node_size=100, alpha=0.3)
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.1)

        # Highlight contradiction pairs
        contradiction_edges = []
        contradiction_nodes = set()

        for v1, v2 in semantic_graph.contradictions:
            if v1 in G.nodes and v2 in G.nodes:
                contradiction_edges.append((v1, v2))
                contradiction_nodes.add(v1)
                contradiction_nodes.add(v2)

        if contradiction_nodes:
            # Draw contradiction nodes
            contra_colors = [self.config.node_colors.get(
                G.nodes[n].get('vertex_type', 'observation'), 'red')
                for n in contradiction_nodes]

            nx.draw_networkx_nodes(
                G.subgraph(contradiction_nodes), pos, ax=ax1,
                node_color=contra_colors,
                node_size=400,
                edgecolors='red',
                linewidths=3,
                alpha=0.9
            )

            # Draw contradiction edges (as red dashed lines)
            for v1, v2 in contradiction_edges:
                ax1.annotate("", xy=pos[v2], xytext=pos[v1],
                           arrowprops=dict(arrowstyle="<->",
                                         color='red',
                                         lw=2,
                                         ls='--'))

            nx.draw_networkx_labels(
                G.subgraph(contradiction_nodes), pos, ax=ax1,
                font_size=9, font_weight='bold'
            )

        ax1.set_title(f'Contradiction Network\n({len(semantic_graph.contradictions)} conflicts)')
        ax1.axis('off')

        # Right: Contradiction statistics
        ax2 = axes[1]

        if semantic_graph.contradictions:
            # Analyze contradiction structure
            contra_graph = nx.Graph()
            for v1, v2 in semantic_graph.contradictions:
                contra_graph.add_edge(v1, v2)

            # Find connected components (clusters of related contradictions)
            clusters = list(nx.connected_components(contra_graph))

            # Plot cluster size distribution
            cluster_sizes = [len(c) for c in clusters]
            ax2.bar(range(len(cluster_sizes)), sorted(cluster_sizes, reverse=True),
                   color='indianred', edgecolor='black')
            ax2.set_xlabel('Cluster Index')
            ax2.set_ylabel('Cluster Size')
            ax2.set_title(f'Contradiction Cluster Sizes\n({len(clusters)} clusters)')

            # Add text summary
            summary = (f"Total contradictions: {len(semantic_graph.contradictions)}\n"
                      f"Unique nodes involved: {len(contradiction_nodes)}\n"
                      f"Number of clusters: {len(clusters)}\n"
                      f"Largest cluster: {max(cluster_sizes) if cluster_sizes else 0}")
            ax2.text(0.95, 0.95, summary, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        else:
            ax2.text(0.5, 0.5, "No contradictions detected\n(Knowledge is coherent)",
                    ha='center', va='center', fontsize=14,
                    color='green', fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def plot_evolution(self, semantic_graph: 'SemanticGraph') -> plt.Figure:
        """
        Plot the evolution of knowledge graph metrics over time.

        Call this periodically to track:
        - Graph growth
        - Entropy trends
        - Coherence evolution
        - Contradiction accumulation
        """
        # Record current state
        self.entropy_history.append(semantic_graph.compute_semantic_entropy())
        self.coherence_history.append(semantic_graph.compute_coherence())
        self.n_vertices_history.append(len(semantic_graph.vertices))
        self.n_contradictions_history.append(len(semantic_graph.contradictions))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.config.dpi)

        steps = range(len(self.entropy_history))

        # Entropy over time
        ax1 = axes[0, 0]
        ax1.plot(steps, self.entropy_history, 'b-', linewidth=2, label='Semantic Entropy')
        ax1.fill_between(steps, self.entropy_history, alpha=0.3)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Semantic Entropy Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Coherence over time
        ax2 = axes[0, 1]
        ax2.plot(steps, self.coherence_history, 'g-', linewidth=2, label='Coherence')
        ax2.fill_between(steps, self.coherence_history, alpha=0.3, color='green')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Coherence')
        ax2.set_title('Knowledge Coherence Evolution')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Graph size
        ax3 = axes[1, 0]
        ax3.plot(steps, self.n_vertices_history, 'purple', linewidth=2, label='Vertices')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Count')
        ax3.set_title('Knowledge Graph Growth')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Contradictions
        ax4 = axes[1, 1]
        ax4.plot(steps, self.n_contradictions_history, 'r-', linewidth=2, label='Contradictions')
        ax4.fill_between(steps, self.n_contradictions_history, alpha=0.3, color='red')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Count')
        ax4.set_title('Contradiction Accumulation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def plot_knowledge_graph(env: 'MycoBeaverEnv',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot the semantic knowledge graph.

    Quick diagnostic: Is knowledge converging or fragmenting?
    """
    if env.semantic_system is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Semantic system not initialized", ha='center', va='center')
        return fig

    viz = SemanticGraphVisualizer()
    fig = viz.plot_knowledge_graph(env.semantic_system.shared_graph,
                                   title=f"Knowledge Graph - Step {env.current_step}")

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_entropy_heatmap(env: 'MycoBeaverEnv',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot entropy distribution.

    Quick diagnostic: Where is uncertainty concentrated?
    """
    if env.semantic_system is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Semantic system not initialized", ha='center', va='center')
        return fig

    viz = SemanticGraphVisualizer()
    fig = viz.plot_entropy_heatmap(env.semantic_system.shared_graph)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_contradiction_clusters(env: 'MycoBeaverEnv',
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot contradiction clusters.

    Quick diagnostic: Are beliefs conflicting?
    """
    if env.semantic_system is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Semantic system not initialized", ha='center', va='center')
        return fig

    viz = SemanticGraphVisualizer()
    fig = viz.plot_contradiction_clusters(env.semantic_system.shared_graph)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig
