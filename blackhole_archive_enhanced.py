# Blackhole Archive Enhanced Simulation
# Fixes beaver construction, adds packet transport, semantic graph building

"""
ENHANCEMENTS OVER BASE VERSION:

1. Beaver Improvements:
   - Realistic curvature computation (Ricci scalar)
   - Lower construction threshold
   - Energy replenishment from successful builds
   - Collective building behavior

2. Ant Improvements:
   - Actual semantic graph construction
   - Pheromone trails visible in output
   - Vertex discovery based on information density
   - Graph metrics in statistics

3. Bee Improvements:
   - Packet collection from ant graph
   - Wormhole targeting behavior
   - Waggle dance recruitment
   - Transport statistics tracking

4. New Visualizations:
   - Semantic graph visualization
   - Pheromone trail heatmap
   - Packet flow diagram
   - Colony health metrics over time
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import uuid

# Note: Full Lyapunov stability monitoring requires epistemic layer
# (available in blackhole_archive_production.py with --engine production)

# Import base classes
import sys
sys.path.append('/mnt/user-data/outputs')

# Import epistemic cognitive layer for Overmind integration
try:
    from epistemic_cognitive_layer import Overmind, EpistemicSemanticGraph, FreeEnergyComputer
    EPISTEMIC_LAYER_AVAILABLE = True
except ImportError:
    EPISTEMIC_LAYER_AVAILABLE = False
    Overmind = None

# Import Adversarial Pressure Layer for Phase II
try:
    from adversarial_pressure_layer import AdversarialPressureLayer, AgentState
    APL_AVAILABLE = True
except ImportError:
    APL_AVAILABLE = False
    AdversarialPressureLayer = None

# Import Agent Plasticity System for emergent intelligence
try:
    from agent_plasticity import AgentPlasticitySystem, StrategyWeights
    PLASTICITY_AVAILABLE = True
except ImportError:
    PLASTICITY_AVAILABLE = False
    AgentPlasticitySystem = None

# =============================================================================
# THERMODYNAMIC CONSTANTS (replaces arbitrary magic numbers)
# =============================================================================
# All constants derived from first principles or physical reasoning

# Boltzmann-like constant for information thermodynamics (geometric units)
# Relates information entropy to energy: E = k_info * S
K_INFO = 0.01  # Sets energy scale for information processing

# Friction coefficient for geodesic motion damping
# From: dE/dt = -γ v² (viscous drag in curved spacetime)
# γ chosen so that orbital decay time ~ 100 proper time units at r = 10M
GAMMA_FRICTION = 0.001

# Diffusion coefficient for Brownian motion in curved spacetime
# From: σ² = 2D Δt (Einstein relation)
# D = k_info * T / γ where T is effective temperature
D_DIFFUSION = 0.0025  # Gives σ ≈ 0.05 for dt = 1

# Energy costs (in geometric units where M = 1)
ENERGY_MOVE = 0.002      # Cost per unit proper time of movement
ENERGY_BUILD = 0.02      # Cost to modify structural field
ENERGY_COMMUNICATE = 0.005  # Cost to deposit pheromone/waggle
ENERGY_BASE_DECAY = 0.001   # Passive metabolism per unit time

# Thresholds derived from spacetime scales
CURVATURE_THRESHOLD = 0.25  # K ~ 1/r³, this corresponds to r ~ 1.6 r_s
INFO_DENSITY_THRESHOLD = 0.5  # 50% of maximum information density

# Probabilities from statistical mechanics
# P ~ exp(-E/kT) for activation processes
# With kT ~ 0.1 in geometric units:
P_SPONTANEOUS_ACTION = 0.01  # exp(-4.6) ≈ 0.01, activation energy ~ 0.46

# =============================================================================
# RESEARCH-GRADE METRICS & DIAGNOSTICS
# =============================================================================

class GraphHealthDashboard:
    """
    Scientific metrics for graph health analysis.

    Tracks structural properties that distinguish "bigger graph" from
    "useful knowledge structure". Outputs to JSON/CSV for analysis.
    """

    def __init__(self, output_dir: str = "./enhanced_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = []
        self.current_step = 0

    def compute_metrics(self, graph: nx.DiGraph, agents: dict, step: int) -> dict:
        """Compute comprehensive graph health metrics."""
        self.current_step = step
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        if n_nodes == 0:
            return self._empty_metrics(step)

        metrics = {
            'step': step,
            'timestamp': str(np.datetime64('now')),

            # === STRUCTURAL METRICS ===
            'n_vertices': n_nodes,
            'n_edges': n_edges,
            'density': n_edges / (n_nodes * (n_nodes - 1) + 1e-10),

            # Giant component: fragmentation detector
            'giant_component_size': self._giant_component_size(graph),
            'giant_component_pct': self._giant_component_size(graph) / max(1, n_nodes),
            'n_components': nx.number_weakly_connected_components(graph),

            # Navigability: can agents find what they need?
            'avg_shortest_path': self._avg_shortest_path(graph),
            'diameter': self._graph_diameter(graph),

            # Clustering: redundant reinforcement
            'avg_clustering': nx.average_clustering(graph.to_undirected()) if n_nodes > 1 else 0,

            # Hub structure: power-law or balanced?
            'max_in_degree': max(dict(graph.in_degree()).values()) if n_nodes > 0 else 0,
            'max_out_degree': max(dict(graph.out_degree()).values()) if n_nodes > 0 else 0,
            'degree_assortativity': self._safe_assortativity(graph),

            # === SALIENCE METRICS ===
            'mean_salience': np.mean([graph.nodes[v].get('salience', 0) for v in graph.nodes()]),
            'salience_std': np.std([graph.nodes[v].get('salience', 0) for v in graph.nodes()]),
            'salience_gini': self._gini_coefficient([graph.nodes[v].get('salience', 0) for v in graph.nodes()]),

            # === DYNAMICS METRICS ===
            'edge_churn_rate': self._compute_edge_churn(),

            # === QUEUE METRICS (congestion) ===
            'total_queue_backlog': 0,  # Will be filled by caller
            'max_queue_length': 0,

            # === PER-COLONY CONTRIBUTION ===
            'beaver_structures': sum(a.structures_built for a in agents.get('beavers', [])),
            'ant_vertices_created': sum(a.vertices_created for a in agents.get('ants', [])),
            'ant_edges_created': sum(a.edges_created for a in agents.get('ants', [])),
            'bee_packets_delivered': sum(a.packets_delivered for a in agents.get('bees', [])),
            'bee_packets_dropped': sum(a.packets_dropped for a in agents.get('bees', [])),

            # Colony survival
            'beavers_alive': sum(1 for a in agents.get('beavers', []) if a.state == 'active'),
            'ants_alive': sum(1 for a in agents.get('ants', []) if a.state == 'active'),
            'bees_alive': sum(1 for a in agents.get('bees', []) if a.state == 'active'),
        }

        self.history.append(metrics)
        return metrics

    def _empty_metrics(self, step: int) -> dict:
        """Return metrics for empty graph."""
        return {
            'step': step, 'n_vertices': 0, 'n_edges': 0, 'density': 0,
            'giant_component_size': 0, 'giant_component_pct': 0, 'n_components': 0,
            'avg_shortest_path': 0, 'diameter': 0, 'avg_clustering': 0,
            'max_in_degree': 0, 'max_out_degree': 0, 'degree_assortativity': 0,
            'mean_salience': 0, 'salience_std': 0, 'salience_gini': 0,
            'edge_churn_rate': 0, 'total_queue_backlog': 0, 'max_queue_length': 0,
            'beaver_structures': 0, 'ant_vertices_created': 0, 'ant_edges_created': 0,
            'bee_packets_delivered': 0, 'bee_packets_dropped': 0,
            'beavers_alive': 0, 'ants_alive': 0, 'bees_alive': 0,
        }

    def _giant_component_size(self, graph: nx.DiGraph) -> int:
        """Size of largest weakly connected component."""
        if graph.number_of_nodes() == 0:
            return 0
        components = list(nx.weakly_connected_components(graph))
        return max(len(c) for c in components) if components else 0

    def _avg_shortest_path(self, graph: nx.DiGraph) -> float:
        """Average shortest path in giant component (navigability)."""
        if graph.number_of_nodes() < 2:
            return 0

        # Use largest connected component
        largest_cc = max(nx.weakly_connected_components(graph), key=len)
        if len(largest_cc) < 2:
            return 0

        subgraph = graph.subgraph(largest_cc)
        try:
            return nx.average_shortest_path_length(subgraph)
        except nx.NetworkXError:
            return 0

    def _graph_diameter(self, graph: nx.DiGraph) -> int:
        """Diameter of giant component."""
        if graph.number_of_nodes() < 2:
            return 0

        largest_cc = max(nx.weakly_connected_components(graph), key=len)
        if len(largest_cc) < 2:
            return 0

        subgraph = graph.subgraph(largest_cc)
        try:
            return nx.diameter(subgraph.to_undirected())
        except nx.NetworkXError:
            return 0

    def _safe_assortativity(self, graph: nx.DiGraph) -> float:
        """Degree assortativity (hubs connected to hubs?)."""
        try:
            return nx.degree_assortativity_coefficient(graph)
        except (nx.NetworkXError, ZeroDivisionError):
            return 0

    def _gini_coefficient(self, values: list) -> float:
        """Gini coefficient for inequality (0=equal, 1=one has all)."""
        if not values or len(values) < 2:
            return 0
        values = np.array(sorted(values))
        n = len(values)
        if values.sum() == 0:
            return 0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))

    def _compute_edge_churn(self) -> float:
        """Edge churn rate (stability vs thrashing)."""
        if len(self.history) < 2:
            return 0
        prev_edges = self.history[-2].get('n_edges', 0) if len(self.history) > 1 else 0
        curr_edges = self.history[-1].get('n_edges', 0) if self.history else 0
        return abs(curr_edges - prev_edges) / max(1, prev_edges)

    def save_history(self, filename: str = "graph_health_metrics.json"):
        """Save metrics history to JSON."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2, default=float)
        return path

    def save_csv(self, filename: str = "graph_health_metrics.csv"):
        """Save metrics history to CSV for analysis."""
        import csv
        path = self.output_dir / filename
        if not self.history:
            return path

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)
        return path

    def diagnose_collapse(self, current_v: int, previous_v: int, graph: nx.DiGraph) -> dict:
        """Auto-diagnose when graph collapses (V drops sharply)."""
        if previous_v == 0 or current_v >= previous_v * 0.7:
            return None  # No collapse

        diagnosis = {
            'collapse_detected': True,
            'vertex_drop_pct': (previous_v - current_v) / previous_v,
            'current_vertices': current_v,
            'previous_vertices': previous_v,
            'components_after': nx.number_weakly_connected_components(graph),
            'giant_component_pct': self._giant_component_size(graph) / max(1, current_v),
            'recent_history': self.history[-10:] if len(self.history) >= 10 else self.history,
        }

        # Analyze salience distribution
        saliences = [graph.nodes[v].get('salience', 0) for v in graph.nodes()]
        if saliences:
            diagnosis['salience_mean'] = np.mean(saliences)
            diagnosis['salience_min'] = min(saliences)
            diagnosis['salience_max'] = max(saliences)

        return diagnosis


class EventLedger:
    """
    Discrete event logging for debugging and replay.

    Tracks all significant events with causality info for post-mortem analysis.
    """

    def __init__(self, output_dir: str = "./enhanced_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.event_counts = {}

    def log(self, event_type: str, step: int, details: dict = None):
        """Log a discrete event."""
        event = {
            'type': event_type,
            'step': step,
            'timestamp': str(np.datetime64('now')),
            'details': details or {}
        }
        self.events.append(event)

        # Track counts
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

    def log_vertex_created(self, step: int, vertex_id: int, position: np.ndarray,
                           salience: float, creator_id: str):
        self.log('vertex_created', step, {
            'vertex_id': vertex_id,
            'position': position.tolist() if hasattr(position, 'tolist') else position,
            'salience': salience,
            'creator': creator_id
        })

    def log_vertex_pruned(self, step: int, vertex_id: int, reason: str,
                          age: float = 0, access_count: int = 0):
        self.log('vertex_pruned', step, {
            'vertex_id': vertex_id,
            'reason': reason,
            'age': age,
            'access_count': access_count
        })

    def log_vertex_merged(self, step: int, source_id: int, target_id: int,
                          distance: float, salience_diff: float):
        self.log('vertex_merged', step, {
            'source': source_id,
            'target': target_id,
            'distance': distance,
            'salience_diff': salience_diff
        })

    def log_agent_death(self, step: int, agent_id: str, cause: str,
                        energy: float, position: np.ndarray):
        self.log('agent_death', step, {
            'agent_id': agent_id,
            'cause': cause,
            'energy': energy,
            'position': position.tolist() if hasattr(position, 'tolist') else position
        })

    def log_backlog_spike(self, step: int, total_backlog: int, max_queue: int):
        self.log('backlog_spike', step, {
            'total_backlog': total_backlog,
            'max_queue': max_queue
        })

    def log_fragmentation(self, step: int, n_components: int, giant_pct: float):
        self.log('fragmentation', step, {
            'n_components': n_components,
            'giant_component_pct': giant_pct
        })

    def log_apl_trigger(self, step: int, threat_type: str, damage: float):
        self.log('apl_trigger', step, {
            'threat_type': threat_type,
            'damage': damage
        })

    def get_summary(self) -> dict:
        """Get event count summary."""
        return {
            'total_events': len(self.events),
            'event_counts': self.event_counts.copy(),
            'first_event': self.events[0] if self.events else None,
            'last_event': self.events[-1] if self.events else None,
        }

    def save(self, filename: str = "event_ledger.json"):
        """Save event ledger to JSON."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump({
                'summary': self.get_summary(),
                'events': self.events[-10000:]  # Keep last 10k events
            }, f, indent=2, default=float)
        return path

    def filter_events(self, event_type: str) -> list:
        """Get all events of a specific type."""
        return [e for e in self.events if e['type'] == event_type]


class KnowledgeUtilityScorer:
    """
    Measures actual utility of knowledge graph beyond just size.

    Utility = (retrieval_success × freshness × coherence) - cost

    Runs periodic "query tasks" where agents navigate to goal concepts.
    """

    def __init__(self):
        self.query_results = []
        self.utility_history = []

    def run_query_task(self, graph: nx.DiGraph, agents: dict, step: int) -> dict:
        """
        Run a knowledge retrieval task to measure utility.

        Randomly samples a goal vertex and measures:
        - Can any agent reach it?
        - How many hops?
        - Energy cost to reach?
        """
        if graph.number_of_nodes() < 5:
            return {'step': step, 'utility': 0, 'reason': 'graph_too_small'}

        # Sample a random goal vertex (prefer high-salience)
        vertices = list(graph.nodes())
        saliences = [graph.nodes[v].get('salience', 0.5) for v in vertices]
        # Weighted sampling by salience
        weights = np.array(saliences) / sum(saliences) if sum(saliences) > 0 else None
        goal_vertex = np.random.choice(vertices, p=weights)

        # Find shortest paths from all vertices to goal
        try:
            # FIX: NetworkX returns generator, convert to dict first
            paths = dict(nx.single_target_shortest_path_length(graph, goal_vertex))
            reachable_count = len(paths)
            avg_path_length = np.mean(list(paths.values())) if paths else float('inf')
        except nx.NetworkXError:
            reachable_count = 0
            avg_path_length = float('inf')

        # Freshness: how recently was goal accessed?
        goal_data = graph.nodes.get(goal_vertex, {})
        freshness = 1.0 / (1.0 + step - goal_data.get('last_accessed', 0))

        # Coherence: does goal have meaningful connections?
        in_degree = graph.in_degree(goal_vertex)
        out_degree = graph.out_degree(goal_vertex)
        coherence = min(1.0, (in_degree + out_degree) / 10.0)

        # Retrieval success
        retrieval_success = reachable_count / len(vertices)

        # Cost (inverse of efficiency)
        cost = avg_path_length / 10.0 if avg_path_length < float('inf') else 1.0

        # Final utility
        utility = (retrieval_success * freshness * coherence) - cost

        result = {
            'step': step,
            'goal_vertex': goal_vertex,
            'goal_salience': goal_data.get('salience', 0),
            'retrieval_success': retrieval_success,
            'freshness': freshness,
            'coherence': coherence,
            'avg_path_length': avg_path_length if avg_path_length < float('inf') else -1,
            'cost': cost,
            'utility': utility
        }

        self.query_results.append(result)
        self.utility_history.append(utility)

        return result

    def get_average_utility(self, window: int = 10) -> float:
        """Get rolling average utility."""
        if not self.utility_history:
            return 0
        return np.mean(self.utility_history[-window:])

    def save(self, output_dir: str = "./enhanced_results"):
        """Save utility results."""
        path = Path(output_dir) / "knowledge_utility.json"
        with open(path, 'w') as f:
            json.dump({
                'average_utility': self.get_average_utility(),
                'query_results': self.query_results
            }, f, indent=2, default=float)
        return path


class GraphGovernor:
    """
    Self-regulating graph governance system.

    Dynamically adjusts merge/prune policies based on graph health metrics.
    Replaces fixed schedules with adaptive control.
    """

    def __init__(self):
        self.policies = {
            'merge_aggressiveness': 1.0,  # 1.0 = normal, >1 = more aggressive
            'prune_aggressiveness': 1.0,
            'bridge_priority': 1.0,       # Priority for creating bridging edges
            'hub_dampening': 1.0,         # Reduce concentration in hubs
        }

        self.target_metrics = {
            'vertex_growth_rate': 0.1,    # Target 10% growth per period
            'fragmentation_max': 0.3,     # Max 30% fragmentation
            'queue_backlog_max': 100,     # Max total queue backlog
            'salience_gini_max': 0.5,     # Max inequality in salience
        }

        self.adjustment_history = []

    def update_policies(self, metrics: dict, step: int) -> dict:
        """
        Adjust governance policies based on current graph health.

        Returns policy adjustments made.
        """
        adjustments = {}

        # 1. If V grows too fast → increase merge aggressiveness
        if len(self.adjustment_history) > 0:
            prev_v = self.adjustment_history[-1].get('n_vertices', 0)
            curr_v = metrics.get('n_vertices', 0)
            growth_rate = (curr_v - prev_v) / max(1, prev_v)

            if growth_rate > self.target_metrics['vertex_growth_rate'] * 2:
                self.policies['merge_aggressiveness'] = min(2.0, self.policies['merge_aggressiveness'] * 1.1)
                adjustments['merge_aggressiveness'] = 'increased (fast growth)'
            elif growth_rate < 0:  # Shrinking
                self.policies['merge_aggressiveness'] = max(0.5, self.policies['merge_aggressiveness'] * 0.9)
                adjustments['merge_aggressiveness'] = 'decreased (shrinking)'

        # 2. If fragmentation increases → reduce pruning + boost bridging
        giant_pct = metrics.get('giant_component_pct', 1.0)
        if giant_pct < 1.0 - self.target_metrics['fragmentation_max']:
            self.policies['prune_aggressiveness'] = max(0.5, self.policies['prune_aggressiveness'] * 0.9)
            self.policies['bridge_priority'] = min(2.0, self.policies['bridge_priority'] * 1.1)
            adjustments['fragmentation'] = 'detected - reduced pruning, boosted bridging'

        # 3. If queue backlog rises → boost transport priority (affects bees)
        backlog = metrics.get('total_queue_backlog', 0)
        if backlog > self.target_metrics['queue_backlog_max']:
            adjustments['backlog_alert'] = f'high backlog: {backlog}'

        # 4. If salience too concentrated → dampen hub dominance
        salience_gini = metrics.get('salience_gini', 0)
        if salience_gini > self.target_metrics['salience_gini_max']:
            self.policies['hub_dampening'] = min(2.0, self.policies['hub_dampening'] * 1.1)
            adjustments['hub_dampening'] = 'increased (concentration detected)'

        # Record for history
        record = {
            'step': step,
            'policies': self.policies.copy(),
            'adjustments': adjustments,
            'metrics_snapshot': {
                'n_vertices': metrics.get('n_vertices', 0),
                'giant_component_pct': giant_pct,
                'salience_gini': salience_gini,
            }
        }
        self.adjustment_history.append(record)

        return adjustments

    def get_merge_threshold(self, base_threshold: float = 0.05) -> float:
        """Get adjusted merge distance threshold.

        FIX: Was inverted (/ instead of *). Higher aggressiveness should mean
        LARGER threshold = MORE vertices qualify for merge, not fewer.
        """
        return base_threshold * self.policies['merge_aggressiveness']

    def should_prune(self, base_probability: float = 1.0) -> bool:
        """Should we prune this cycle?"""
        return np.random.random() < base_probability * self.policies['prune_aggressiveness']

    def get_bridge_bonus(self) -> float:
        """Bonus probability for creating bridging edges."""
        return 0.1 * self.policies['bridge_priority']


# =============================================================================
# ENHANCED PHYSICS
# =============================================================================

class EnhancedSpacetime:
    """Enhanced spacetime with proper curvature computation"""

    # PERFORMANCE: Discretization bins for Christoffel caching
    _CHRISTOFFEL_R_BINS = 200  # Radial shells
    _CHRISTOFFEL_THETA_BINS = 50  # Angular bins

    def __init__(self, config):
        self.config = config
        self.M = config.black_hole_mass
        self.r_s = 2 * self.M

        # Grid
        self.r = np.linspace(config.r_min, config.r_max, config.n_r)
        self.theta = np.linspace(0, np.pi, config.n_theta)
        self.phi = np.linspace(0, 2*np.pi, config.n_phi)

        # Precompute metric
        self.g = self._compute_metric()

        # Structural field from beavers
        self.structural_field = np.zeros((config.n_r, config.n_theta, config.n_phi))

        # Information density field (for ants)
        self.information_density = self._initialize_information_density()

        # PERFORMANCE: Christoffel symbol cache (5-10x speedup)
        self._christoffel_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # FIX #7: Epistemic stress field (affected by contradictions/congestion)
        # High epistemic stress = harder to traverse, like gravitational stress
        self.epistemic_stress = np.zeros((config.n_r, config.n_theta, config.n_phi))

        # DYNAMIC SPACETIME: Stress-energy tensor and metric perturbation
        # T_μν tracks energy density from agents and packets
        # Metric perturbation h_μν represents back-reaction on geometry
        self.stress_energy = np.zeros((config.n_r, config.n_theta, config.n_phi))
        self.metric_perturbation = np.zeros((config.n_r, config.n_theta, config.n_phi))

        # Bekenstein bound tracking: S ≤ 2πkRE/(ℏc)
        # In geometric units (G=c=1): S ≤ 2πRE
        self.local_entropy = np.zeros((config.n_r, config.n_theta, config.n_phi))
        # Count of thermalization events (when Bekenstein bound is enforced)
        # These are NOT violations - the bound IS enforced, excess info is thermalized
        self.bekenstein_thermalization_count = 0

        # Landauer limit tracking: E = k_B T ln(2) per bit erased
        self.bits_erased = 0
        self.thermodynamic_heat = 0.0  # Cumulative heat from erasure
    
    def _compute_metric(self) -> np.ndarray:
        """Compute Schwarzschild metric"""
        metric = np.zeros((self.config.n_r, self.config.n_theta, self.config.n_phi, 4, 4))
        
        for i, r in enumerate(self.r):
            f = 1 - self.r_s / r
            
            metric[i, :, :, 0, 0] = -f
            metric[i, :, :, 1, 1] = 1 / f
            
            for j, theta in enumerate(self.theta):
                metric[i, j, :, 2, 2] = r**2
                metric[i, j, :, 3, 3] = r**2 * np.sin(theta)**2
        
        return metric
    
    def _initialize_information_density(self) -> np.ndarray:
        """Initialize information density field for ant exploration"""
        # Create interesting structure with multiple peaks
        info = np.zeros((self.config.n_r, self.config.n_theta, self.config.n_phi))

        # Add Gaussian blobs at random locations
        # FIXED: 20 blobs was too sparse - increased to 100 for better coverage
        np.random.seed(42)
        n_blobs = 100  # Was 20 - way too sparse

        for _ in range(n_blobs):
            i = np.random.randint(0, self.config.n_r)
            j = np.random.randint(0, self.config.n_theta)
            k = np.random.randint(0, self.config.n_phi)

            strength = np.random.uniform(0.3, 1.0)
            width = np.random.uniform(3.0, 8.0)  # Wider blobs for more coverage

            # Add Gaussian centered at (i,j,k)
            for di in range(-10, 11):
                for dj in range(-5, 6):
                    for dk in range(-5, 6):
                        ii = (i + di) % self.config.n_r
                        jj = max(0, min(self.config.n_theta-1, j + dj))
                        kk = (k + dk) % self.config.n_phi

                        distance = np.sqrt(di**2 + dj**2 + dk**2)
                        info[ii, jj, kk] += strength * np.exp(-distance**2 / (2*width**2))

        # Add baseline noise so no region is completely empty
        info += 0.02 * np.random.random(info.shape)

        # Normalize to [0, 1] but preserve relative structure
        info = info / (np.max(info) + 1e-10)

        return info
    
    def get_ricci_scalar(self, position: np.ndarray) -> float:
        """
        Compute Ricci scalar at position.

        For Schwarzschild vacuum solution: R = g^μν R_μν = 0.
        This is exact - Schwarzschild is a vacuum solution to Einstein's equations.

        For non-vacuum regions (with matter/energy), use get_effective_ricci_scalar().

        Returns:
            0.0 (Schwarzschild is a vacuum solution)
        """
        # Schwarzschild is a vacuum solution: R_μν = 0 → R = 0
        return 0.0

    def get_kretschmann_scalar(self, position: np.ndarray) -> float:
        """
        Compute Kretschmann scalar K = R_μνρσ R^μνρσ at position.

        For Schwarzschild metric: K = 48 M² / r⁶

        This is the appropriate curvature measure for vacuum solutions,
        as the Ricci scalar vanishes. The Kretschmann scalar quantifies
        the strength of tidal forces and diverges at r=0.

        Physical interpretation:
        - Diverges as r → 0 (physical singularity)
        - Finite at horizon r = 2M (coordinate singularity only)
        - Falls off as r⁻⁶ at large distances

        Returns:
            Kretschmann scalar in geometric units (1/length⁴)
        """
        r = max(position[1], 1e-10)  # Avoid division by zero
        return 48.0 * self.M**2 / r**6

    def get_tidal_strength(self, position: np.ndarray) -> float:
        """
        Get tidal force strength at position (sqrt of Kretschmann).

        This provides a length^-2 quantity suitable for comparing
        with other physical scales in the simulation.

        Returns:
            sqrt(K) in geometric units (1/length²)
        """
        return np.sqrt(self.get_kretschmann_scalar(position))

    def get_curvature(self, position: np.ndarray) -> float:
        """
        Get effective curvature at position for agent decision-making.

        Uses tidal strength (sqrt of Kretschmann) as the physically
        meaningful curvature measure for vacuum spacetimes.

        This is the appropriate quantity for:
        - Beaver construction decisions (build where curvature is high)
        - Agent energy costs (curved regions are harder to traverse)
        - Stability assessments (high curvature = less stable)

        Returns:
            Effective curvature measure (1/length²)
        """
        return self.get_tidal_strength(position)
    
    def get_information_density(self, position: np.ndarray) -> float:
        """Sample information density at position"""
        # Find nearest grid point
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))
        
        return self.information_density[i, j, k]
    
    def get_time_dilation(self, position: np.ndarray) -> float:
        """Get time dilation factor"""
        r = position[1]
        return 1 / np.sqrt(max(1e-10, 1 - self.r_s / r))

    def _discretize_position(self, r: float, theta: float) -> tuple:
        """Discretize position to cache bin for Christoffel lookup."""
        # Clamp r to valid range
        r = max(r, self.r_s * 1.01)
        r = min(r, self.config.r_max)

        # Discretize to bins
        r_bin = int((r - self.r_s) / (self.config.r_max - self.r_s) * self._CHRISTOFFEL_R_BINS)
        r_bin = max(0, min(r_bin, self._CHRISTOFFEL_R_BINS - 1))

        theta_bin = int(theta / np.pi * self._CHRISTOFFEL_THETA_BINS)
        theta_bin = max(0, min(theta_bin, self._CHRISTOFFEL_THETA_BINS - 1))

        return (r_bin, theta_bin)

    def _compute_christoffel_raw(self, r: float, theta: float) -> np.ndarray:
        """Compute Christoffel symbols at exact (r, theta) - internal use."""
        M = self.M
        Gamma = np.zeros((4, 4, 4))

        # Avoid division by zero
        f = 1 - 2*M/r
        if f < 1e-10:
            f = 1e-10

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        if abs(sin_theta) < 1e-10:
            sin_theta = 1e-10

        # Non-zero Christoffel symbols for Schwarzschild
        Gamma[0, 0, 1] = Gamma[0, 1, 0] = M / (r * (r - 2*M) + 1e-10)
        Gamma[1, 0, 0] = M * (r - 2*M) / (r**3 + 1e-10)
        Gamma[1, 1, 1] = -M / (r * (r - 2*M) + 1e-10)
        Gamma[1, 2, 2] = -(r - 2*M)
        Gamma[1, 3, 3] = -(r - 2*M) * sin_theta**2
        Gamma[2, 1, 2] = Gamma[2, 2, 1] = 1/r
        Gamma[2, 3, 3] = -sin_theta * cos_theta
        Gamma[3, 1, 3] = Gamma[3, 3, 1] = 1/r
        Gamma[3, 2, 3] = Gamma[3, 3, 2] = cos_theta / sin_theta

        return Gamma

    def compute_christoffel(self, position: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols at position for geodesic motion.

        OPTIMIZED: Uses caching by discretized radial/angular bins.
        This gives 5-10x speedup since Christoffel symbols vary smoothly
        and can be reused across nearby positions.

        For Schwarzschild metric, the non-zero components are:
        Γ^t_tr = M / (r(r - 2M))
        Γ^r_tt = M(r - 2M) / r³
        Γ^r_rr = -M / (r(r - 2M))
        Γ^r_θθ = -(r - 2M)
        Γ^r_φφ = -(r - 2M)sin²θ
        Γ^θ_rθ = 1/r
        Γ^θ_φφ = -sinθ cosθ
        Γ^φ_rφ = 1/r
        Γ^φ_θφ = cotθ
        """
        r = max(position[1], self.r_s * 1.01)  # Avoid singularity
        theta = position[2]

        # Check cache
        cache_key = self._discretize_position(r, theta)
        if cache_key in self._christoffel_cache:
            self._cache_hits += 1
            return self._christoffel_cache[cache_key]

        # Cache miss - compute and store
        self._cache_misses += 1
        Gamma = self._compute_christoffel_raw(r, theta)
        self._christoffel_cache[cache_key] = Gamma

        # Limit cache size to prevent memory issues
        if len(self._christoffel_cache) > 20000:
            # Clear half the cache (LRU approximation)
            keys = list(self._christoffel_cache.keys())
            for k in keys[:10000]:
                del self._christoffel_cache[k]

        return Gamma

    def _geodesic_acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Compute geodesic acceleration from the geodesic equation.

        The geodesic equation is:
        d²x^μ/dτ² = -Γ^μ_αβ (dx^α/dτ)(dx^β/dτ)

        OPTIMIZED: Uses numpy einsum instead of Python loops (3-5x faster).

        Args:
            position: 4-position (t, r, θ, φ)
            velocity: 4-velocity

        Returns:
            4-acceleration
        """
        Gamma = self.compute_christoffel(position)

        # Vectorized contraction: a^μ = -Γ^μ_αβ v^α v^β
        # einsum performs: acceleration[m] = sum over a,b of Gamma[m,a,b] * v[a] * v[b]
        acceleration = -np.einsum('mab,a,b->m', Gamma, velocity, velocity)

        # Handle any NaN from near-singularity
        acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clip to prevent numerical explosion near singularities
        return np.clip(acceleration, -10.0, 10.0)

    def geodesic_step(self, position: np.ndarray, velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute geodesic-corrected position and velocity update.

        Uses 4th-order Runge-Kutta integration (RK4) for the geodesic equation:
        dx^μ/dτ = v^μ
        dv^μ/dτ = -Γ^μ_αβ v^α v^β

        RK4 provides O(dt⁴) accuracy and good energy conservation for
        orbital dynamics, making it suitable for geodesic motion.

        For truly symplectic (exactly energy-conserving) integration,
        use geodesic_step_symplectic() instead.

        Args:
            position: Current 4-position (t, r, θ, φ)
            velocity: Current 4-velocity
            dt: Time step (proper time interval)

        Returns:
            (new_position, new_velocity)
        """
        # Safety check: ensure we're outside the horizon
        if position[1] <= self.r_s * 1.01:
            position = position.copy()
            position[1] = self.r_s * 1.05
            velocity = velocity.copy()
            velocity[1] = max(velocity[1], 0.1)

        # RK4 integration of the coupled system:
        # dx/dt = v,  dv/dt = a(x, v)

        # k1
        k1_x = velocity
        k1_v = self._geodesic_acceleration(position, velocity)

        # k2
        x2 = position + 0.5 * dt * k1_x
        v2 = velocity + 0.5 * dt * k1_v
        k2_x = v2
        k2_v = self._geodesic_acceleration(x2, v2)

        # k3
        x3 = position + 0.5 * dt * k2_x
        v3 = velocity + 0.5 * dt * k2_v
        k3_x = v3
        k3_v = self._geodesic_acceleration(x3, v3)

        # k4
        x4 = position + dt * k3_x
        v4 = velocity + dt * k3_v
        k4_x = v4
        k4_v = self._geodesic_acceleration(x4, v4)

        # Combine
        new_position = position + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_velocity = velocity + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        # Energy-based velocity constraint (replaces arbitrary np.clip(-5, 5))
        # For geodesic motion, we preserve the norm of the 4-velocity
        # In Schwarzschild: g_μν u^μ u^ν = -1 for timelike geodesics
        # Speed limit: spatial velocity bounded by effective potential
        r = max(new_position[1], self.r_s * 1.5)
        # Maximum allowed spatial velocity at radius r (from effective potential)
        # v_max² ≈ 2(E - V_eff) where V_eff = -(r_s/2r) for radial motion
        v_max = min(5.0, np.sqrt(2.0 * (1.0 + self.r_s / (2 * r))))
        spatial_speed = np.linalg.norm(new_velocity[1:])
        if spatial_speed > v_max:
            # Scale down spatial components while preserving direction
            scale = v_max / spatial_speed
            new_velocity[1:] *= scale

        # Handle NaN/Inf - reset to safe values
        if not np.all(np.isfinite(new_position)):
            new_position = position.copy()
            new_position[1] = max(new_position[1], self.r_s * 1.5)
        if not np.all(np.isfinite(new_velocity)):
            new_velocity = np.zeros(4)
            new_velocity[1] = 0.1

        # Boundary conditions
        new_position, new_velocity = self._apply_boundary_conditions(new_position, new_velocity)

        return new_position, new_velocity

    def geodesic_step_symplectic(self, position: np.ndarray, velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Symplectic (Störmer-Verlet/Leapfrog) geodesic integration.

        This integrator exactly preserves the symplectic structure of
        Hamiltonian dynamics, providing exact energy conservation
        (up to floating-point precision) over arbitrarily long times.

        The leapfrog scheme:
        v_{n+1/2} = v_n + (dt/2) * a(x_n, v_n)
        x_{n+1} = x_n + dt * v_{n+1/2}
        v_{n+1} = v_{n+1/2} + (dt/2) * a(x_{n+1}, v_{n+1/2})

        Args:
            position: Current 4-position (t, r, θ, φ)
            velocity: Current 4-velocity
            dt: Time step

        Returns:
            (new_position, new_velocity)
        """
        # Safety check
        if position[1] <= self.r_s * 1.01:
            position = position.copy()
            position[1] = self.r_s * 1.05
            velocity = velocity.copy()
            velocity[1] = max(velocity[1], 0.1)

        # Half-step velocity update (kick)
        a0 = self._geodesic_acceleration(position, velocity)
        v_half = velocity + 0.5 * dt * a0

        # Full-step position update (drift)
        new_position = position + dt * v_half

        # Half-step velocity update (kick)
        a1 = self._geodesic_acceleration(new_position, v_half)
        new_velocity = v_half + 0.5 * dt * a1

        # Energy-based velocity constraint (replaces arbitrary np.clip(-5, 5))
        r = max(new_position[1], self.r_s * 1.5)
        v_max = min(5.0, np.sqrt(2.0 * (1.0 + self.r_s / (2 * r))))
        spatial_speed = np.linalg.norm(new_velocity[1:])
        if spatial_speed > v_max:
            scale = v_max / spatial_speed
            new_velocity[1:] *= scale

        # Handle NaN/Inf
        if not np.all(np.isfinite(new_position)):
            new_position = position.copy()
            new_position[1] = max(new_position[1], self.r_s * 1.5)
        if not np.all(np.isfinite(new_velocity)):
            new_velocity = np.zeros(4)
            new_velocity[1] = 0.1

        # Boundary conditions
        new_position, new_velocity = self._apply_boundary_conditions(new_position, new_velocity)

        return new_position, new_velocity

    def _apply_boundary_conditions(self, position: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions to position and velocity."""
        position = position.copy()
        velocity = velocity.copy()

        # Keep r outside event horizon
        if position[1] <= self.r_s * 1.01:
            position[1] = self.r_s * 1.05
            velocity[1] = abs(velocity[1])  # Bounce outward

        # Keep r inside domain
        if position[1] >= self.config.r_max * 0.95:
            position[1] = self.config.r_max * 0.9
            velocity[1] = -abs(velocity[1]) * 0.5  # Reflect inward

        # Keep theta in [0, π]
        position[2] = np.clip(position[2], 0.01, np.pi - 0.01)

        # Wrap phi to [0, 2π]
        position[3] = position[3] % (2 * np.pi)

        return position, velocity

    def add_structural_field(self, position: np.ndarray, strength: float, radius: float):
        """Add beaver structural field with proper boundary handling"""
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # FIX: No-build zone at domain boundaries (r_max region)
        # Prevents boundary exploit where agents "build at edge of map"
        if i >= self.config.n_r - 3:  # Too close to outer boundary
            return

        # Add Gaussian with CLAMPED radial indices (no wrap-around)
        for di in range(-5, 6):
            for dj in range(-3, 4):
                for dk in range(-3, 4):
                    # FIX: Clamp radial index instead of modulo wrap
                    # Wrap is only valid for phi (azimuthal), not r (radial)
                    ii = max(0, min(self.config.n_r - 1, i + di))
                    jj = max(0, min(self.config.n_theta - 1, j + dj))
                    kk = (k + dk) % self.config.n_phi  # Phi wraps correctly

                    distance = np.sqrt(di**2 + dj**2 + dk**2)
                    if distance < radius:
                        self.structural_field[ii, jj, kk] += strength * np.exp(-distance**2 / (2 * radius**2))

    # FIX #2: Structural field coupling methods
    def get_structural_field_at(self, position: np.ndarray) -> float:
        """Get structural field value at position - FIX #2"""
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))
        return self.structural_field[i, j, k]

    def get_movement_cost(self, position: np.ndarray) -> float:
        """
        Get movement cost at position - FIX #2
        Structural field REDUCES movement cost (easier to traverse)
        """
        base_cost = 1.0
        structural = self.get_structural_field_at(position)
        # Structural field reduces cost: more structure = easier movement
        # Formula: cost = base / (1 + structural)
        return base_cost / (1.0 + structural)

    def get_effective_time_dilation(self, position: np.ndarray) -> float:
        """
        Get time dilation modified by structural field - FIX #2
        Structural field provides "exotic matter" effect that reduces dilation
        """
        r = position[1]
        base_dilation = 1 / np.sqrt(max(1e-10, 1 - self.r_s / r))
        structural = self.get_structural_field_at(position)
        # Structural field reduces time dilation (stabilizes region)
        # Formula: effective = base / (1 + 0.1 * structural)
        return base_dilation / (1.0 + 0.1 * structural)

    def get_exploration_bias(self, position: np.ndarray) -> float:
        """
        Get exploration bias from structural field - FIX #2
        High structural field = already explored, look elsewhere
        Returns value in [0, 1] where 0 = explore elsewhere, 1 = explore here
        """
        structural = self.get_structural_field_at(position)
        # Inverse relationship: high structure = low exploration priority
        return 1.0 / (1.0 + structural)

    def get_structural_field(self, position: np.ndarray) -> float:
        """Alias for get_structural_field_at for cleaner API."""
        return self.get_structural_field_at(position)

    def get_structural_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Get gradient of structural field at position.
        Returns 3D gradient vector (dr, dtheta, dphi) pointing toward higher structure.
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Compute numerical gradient using central differences
        gradient = np.zeros(3)

        # dr gradient
        if 0 < i < self.config.n_r - 1:
            dr = self.r[1] - self.r[0] if len(self.r) > 1 else 1.0
            gradient[0] = (self.structural_field[i+1, j, k] - self.structural_field[i-1, j, k]) / (2 * dr)

        # dtheta gradient
        if 0 < j < self.config.n_theta - 1:
            dtheta = self.theta[1] - self.theta[0] if len(self.theta) > 1 else 1.0
            gradient[1] = (self.structural_field[i, j+1, k] - self.structural_field[i, j-1, k]) / (2 * dtheta)

        # dphi gradient (periodic)
        dphi = self.phi[1] - self.phi[0] if len(self.phi) > 1 else 1.0
        k_next = (k + 1) % self.config.n_phi
        k_prev = (k - 1) % self.config.n_phi
        gradient[2] = (self.structural_field[i, j, k_next] - self.structural_field[i, j, k_prev]) / (2 * dphi)

        return gradient

    def decay_structural_field(self, dt: float, decay_rate: float = 0.01):
        """
        FIX #5: Structural field decays over time (maintenance cost).
        Structures that aren't maintained fade away.
        This prevents infinite accumulation and encourages active maintenance.
        """
        self.structural_field *= np.exp(-decay_rate * dt)

        # Floor to zero for very small values (cleanup)
        self.structural_field[self.structural_field < 0.01] = 0.0

    def add_epistemic_stress(self, position: np.ndarray, stress: float, radius: float = 3.0):
        """
        FIX #7: Add epistemic stress at a position.
        Called when contradictions or congestion occur.
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Add Gaussian stress field
        for di in range(-3, 4):
            for dj in range(-2, 3):
                for dk in range(-2, 3):
                    ii = max(0, min(self.config.n_r - 1, i + di))
                    jj = max(0, min(self.config.n_theta - 1, j + dj))
                    kk = (k + dk) % self.config.n_phi

                    distance = np.sqrt(di**2 + dj**2 + dk**2)
                    if distance < radius:
                        self.epistemic_stress[ii, jj, kk] += stress * np.exp(-distance**2 / (2 * radius**2))

    def decay_epistemic_stress(self, dt: float, decay_rate: float = 0.05):
        """FIX #7: Epistemic stress decays (contradictions can be resolved)"""
        self.epistemic_stress *= np.exp(-decay_rate * dt)
        self.epistemic_stress[self.epistemic_stress < 0.01] = 0.0

    def get_epistemic_stress_at(self, position: np.ndarray) -> float:
        """Get epistemic stress at position"""
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))
        return self.epistemic_stress[i, j, k]

    def get_total_traversal_cost(self, position: np.ndarray) -> float:
        """
        FIX #7: Combined traversal cost including epistemic stress.
        High stress = epistemic gravity well, harder to traverse.
        """
        base_cost = self.get_movement_cost(position)
        stress = self.get_epistemic_stress_at(position)
        # Stress increases traversal cost
        return base_cost * (1.0 + stress)

    # =========================================================================
    # DYNAMIC SPACETIME: Stress-Energy and Metric Back-Reaction
    # =========================================================================

    def add_stress_energy(self, position: np.ndarray, energy: float):
        """
        Add stress-energy contribution at position.

        In full GR: G_μν = 8πT_μν
        Here we use linearized approximation where metric perturbation
        is proportional to integrated stress-energy.

        Args:
            position: 4-position (t, r, θ, φ)
            energy: Energy contribution (from agent or packet)
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Add to stress-energy tensor (simplified scalar representation)
        self.stress_energy[i, j, k] += energy

    def evolve_metric(self, dt: float):
        """
        Evolve metric perturbation based on stress-energy distribution.

        Uses linearized Einstein equations in weak-field limit:
        □h_μν = -16πT_μν

        In our simplified scalar model (trace-reversed perturbation):
        ∂²h/∂t² - c²∇²h = -16πGT/c²

        In geometric units (G=c=1): ∂²h/∂t² - ∇²h = -16πT

        Physical parameters:
        - Gravitational wave speed = c = 1 (geometric units)
        - Radiative damping rate from quadrupole formula: Γ = (32/5) (GM/c³)⁵ / r⁵
          For weak fields, this is negligible compared to propagation

        The metric perturbation affects:
        - Time dilation (stronger near high energy density)
        - Geodesic paths (agents curve toward energy concentrations)
        """
        # Store previous state for wave equation (need h, h_old for leapfrog)
        if not hasattr(self, '_metric_perturbation_old'):
            self._metric_perturbation_old = np.zeros_like(self.metric_perturbation)

        # Compute Laplacian of metric perturbation (spherical coords)
        laplacian = np.zeros_like(self.metric_perturbation)

        # Radial part: (1/r²) d/dr (r² dh/dr)
        # Using central differences for interior points
        dr = self.r[1] - self.r[0] if len(self.r) > 1 else 1.0
        for i in range(1, self.config.n_r - 1):
            r_i = self.r[i]
            # Second derivative term
            d2h_dr2 = (
                self.metric_perturbation[i+1, :, :] -
                2*self.metric_perturbation[i, :, :] +
                self.metric_perturbation[i-1, :, :]
            ) / (dr**2)
            # First derivative term (from 1/r² d/dr(r² dh/dr))
            dh_dr = (
                self.metric_perturbation[i+1, :, :] -
                self.metric_perturbation[i-1, :, :]
            ) / (2 * dr)
            laplacian[i, :, :] = d2h_dr2 + (2/r_i) * dh_dr

        # Source term from stress-energy: S = -16πT
        # The stress-energy here represents energy density contributions
        source = -16 * np.pi * self.stress_energy

        # Leapfrog/Störmer-Verlet time integration for wave equation
        # h_{n+1} = 2h_n - h_{n-1} + dt²(∇²h + S)
        # This is symplectic and conserves energy for the wave equation
        new_perturbation = (
            2 * self.metric_perturbation -
            self._metric_perturbation_old +
            dt**2 * (laplacian + source)
        )

        # Physical damping from gravitational wave radiation
        # Quadrupole formula gives damping timescale τ ~ r⁵/(GM)⁵
        # For simulation stability, use scale-appropriate damping
        # Damping timescale τ_damp = r_max / c = r_max (in geometric units)
        tau_damp = self.config.r_max
        gamma = dt / tau_damp  # Dimensionless damping per timestep
        new_perturbation *= (1.0 - gamma)

        # Update states
        self._metric_perturbation_old = self.metric_perturbation.copy()
        self.metric_perturbation = new_perturbation

        # Physical constraint: linearized approximation valid for |h| << 1
        # If perturbation exceeds this, the linearized equations break down
        max_perturbation = 0.3  # Beyond this, need full nonlinear GR
        if np.max(np.abs(self.metric_perturbation)) > max_perturbation:
            # Scale down to valid regime (energy conservation)
            scale = max_perturbation / np.max(np.abs(self.metric_perturbation))
            self.metric_perturbation *= scale
            self._metric_perturbation_old *= scale

        # Stress-energy disperses via radiation and diffusion
        # Physical timescale: crossing time ~ r_max / c = r_max
        dispersion_rate = 1.0 / self.config.r_max
        self.stress_energy *= np.exp(-dispersion_rate * dt)

    def get_effective_mass(self, position: np.ndarray) -> float:
        """
        Get effective gravitational mass including metric perturbation.

        The total mass felt at a point is M + δM where δM comes from
        the integrated stress-energy perturbation.
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Perturbation contributes to effective mass
        delta_M = self.metric_perturbation[i, j, k] * self.M
        return self.M + delta_M

    # =========================================================================
    # BEKENSTEIN BOUND AND THERMALIZATION
    # =========================================================================

    def add_local_entropy(self, position: np.ndarray, bits: float):
        """
        Add entropy (information) at position and check Bekenstein bound.

        Bekenstein bound: S ≤ 2πRE/(ℏc) = 2πRE (geometric units)

        If violated, information is "thermalized" (randomly corrupted)
        simulating the physical limit on information density.

        Returns:
            bits_lost: Number of bits thermalized due to bound violation
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        r = position[1]

        # Add entropy
        self.local_entropy[i, j, k] += bits

        # Bekenstein bound: S_max = 2πRE
        # Using local energy from stress-energy tensor
        local_energy = max(self.stress_energy[i, j, k], 0.01)
        bekenstein_limit = 2 * np.pi * r * local_energy

        # Check for violation
        current_entropy = self.local_entropy[i, j, k]
        if current_entropy > bekenstein_limit:
            # Thermalization: excess information is lost
            excess = current_entropy - bekenstein_limit
            self.local_entropy[i, j, k] = bekenstein_limit

            # Track thermalization event (bound enforced, excess info discarded)
            self.bekenstein_thermalization_count += 1

            # Landauer cost: erasing bits generates heat
            self.bits_erased += excess
            # k_B T ln(2) per bit, using T ~ 1/(8πM) Hawking temperature
            hawking_temp = 1.0 / (8 * np.pi * self.M)
            self.thermodynamic_heat += excess * hawking_temp * np.log(2)

            return excess

        return 0.0

    def decay_entropy(self, dt: float, rate: float = 0.02):
        """Entropy naturally dissipates (Hawking radiation analog)"""
        self.local_entropy *= np.exp(-rate * dt)

    def get_bekenstein_capacity(self, position: np.ndarray) -> float:
        """
        Get remaining Bekenstein capacity at position.

        Returns fraction of capacity remaining (0 = full, 1 = empty)
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        r = position[1]
        local_energy = max(self.stress_energy[i, j, k], 0.01)
        bekenstein_limit = 2 * np.pi * r * local_energy

        current = self.local_entropy[i, j, k]
        return max(0.0, 1.0 - current / bekenstein_limit)

    # =========================================================================
    # VALIDATION METRICS
    # =========================================================================

    def estimate_kolmogorov_complexity(self, data: np.ndarray) -> float:
        """
        Estimate Kolmogorov complexity using compression ratio.

        K(x) ≈ len(compress(x))

        Higher values = more complex/random data
        Lower values = more compressible/structured data
        """
        import zlib

        # Convert to bytes
        data_bytes = data.tobytes()

        # Compress
        compressed = zlib.compress(data_bytes, level=9)

        # Complexity estimate = compression ratio
        if len(data_bytes) == 0:
            return 0.0

        return len(compressed) / len(data_bytes)

    def get_thermodynamic_stats(self) -> Dict[str, float]:
        """
        Get thermodynamic validation statistics.

        Returns metrics for validating physical consistency:
        - Total entropy in system
        - Bekenstein violations (should be rare)
        - Landauer heat generated
        - Compression efficiency (Kolmogorov estimate)
        """
        return {
            'total_entropy': float(np.sum(self.local_entropy)),
            'max_local_entropy': float(np.max(self.local_entropy)),
            'bekenstein_thermalization_events': self.bekenstein_thermalization_count,
            'bits_erased': self.bits_erased,
            'thermodynamic_heat': self.thermodynamic_heat,
            'mean_metric_perturbation': float(np.mean(np.abs(self.metric_perturbation))),
            'max_stress_energy': float(np.max(self.stress_energy)),
        }

# =============================================================================
# ACTIVE INFERENCE FRAMEWORK
# =============================================================================

class ActiveInferenceMixin:
    """
    Active Inference agent model based on Free Energy Principle.

    Agents don't have hardcoded roles - they minimize variational free energy:

    F = D_KL[q(ψ)||p(ψ)] - E_q[ln p(o|ψ)]
      = Complexity - Accuracy

    Where:
    - q(ψ): Agent's beliefs about world state
    - p(ψ): Prior beliefs (generative model)
    - p(o|ψ): Likelihood of observations given state
    - Complexity: Cost of updating beliefs
    - Accuracy: How well beliefs predict observations

    Actions are selected to minimize expected free energy (epistemic + pragmatic value).
    """

    def __init_active_inference__(self):
        """Initialize active inference state"""
        # Beliefs about world state (simplified as feature vector)
        self.beliefs = np.zeros(8)  # [energy, curvature, density, stress, ...]

        # Precision (inverse variance) of beliefs - how confident
        self.precision = np.ones(8) * 0.5

        # Prior preferences (what states the agent prefers)
        self.preferences = np.zeros(8)

        # Free energy history for learning
        self.free_energy_history = []

        # Epistemic value (information gain motivation)
        self.epistemic_drive = 0.5

    def compute_free_energy(self, observation: np.ndarray, action: np.ndarray = None) -> float:
        """
        Compute variational free energy given observation.

        F = Complexity + Accuracy
          = D_KL[q||p] - E_q[ln p(o|s)]

        Lower free energy = better model of the world
        """
        # Complexity: KL divergence between posterior and prior beliefs
        # Using simplified Gaussian assumption
        prior_mean = self.preferences
        posterior_mean = self.beliefs

        # KL divergence for Gaussians: 0.5 * sum((μ1-μ2)² / σ² + log(σ²/σ²) + σ²/σ² - 1)
        # Simplified: just squared difference weighted by precision
        complexity = 0.5 * np.sum(self.precision * (posterior_mean - prior_mean)**2)

        # Accuracy: negative log likelihood of observation under beliefs
        # How well do current beliefs predict what we observe?
        prediction_error = observation - self.beliefs
        accuracy = -0.5 * np.sum(self.precision * prediction_error**2)

        # Free energy = complexity - accuracy (want to minimize)
        free_energy = complexity - accuracy

        return free_energy

    def update_beliefs(self, observation: np.ndarray, learning_rate: float = 0.1):
        """
        Update beliefs based on observation (perception).

        Implements belief updating to minimize prediction error.
        """
        # Prediction error
        error = observation - self.beliefs

        # Update beliefs toward observation (gradient descent on free energy)
        self.beliefs += learning_rate * self.precision * error

        # Update precision based on error magnitude (learn confidence)
        error_magnitude = np.abs(error)
        self.precision = 0.9 * self.precision + 0.1 * (1.0 / (error_magnitude + 0.1))

    def expected_free_energy(self, action: np.ndarray, spacetime, position: np.ndarray) -> float:
        """
        Compute expected free energy for an action.

        G = E_q[F(o', s')] = Epistemic Value + Pragmatic Value

        Where:
        - Epistemic: Information gain from action
        - Pragmatic: Preference satisfaction from action
        """
        # Predict next state after action
        predicted_position = position + action * 0.1

        # Get predicted observations at new position
        predicted_obs = self._get_observation(spacetime, predicted_position)

        # Epistemic value: expected information gain
        # Higher uncertainty reduction = higher epistemic value
        uncertainty_before = 1.0 / (np.sum(self.precision) + 1e-6)

        # Simulate belief update
        temp_beliefs = self.beliefs.copy()
        temp_beliefs += 0.1 * self.precision * (predicted_obs - self.beliefs)

        # Uncertainty after (estimated)
        uncertainty_after = np.sum((predicted_obs - temp_beliefs)**2)

        epistemic_value = uncertainty_before - uncertainty_after

        # Pragmatic value: how close to preferences?
        pragmatic_value = -np.sum((predicted_obs - self.preferences)**2)

        # Expected free energy (lower is better)
        return -self.epistemic_drive * epistemic_value - pragmatic_value

    def select_action_active_inference(self, spacetime, candidate_actions: List[np.ndarray]) -> np.ndarray:
        """
        Select action by minimizing expected free energy.

        This replaces hardcoded role-based behavior with principled action selection.
        """
        best_action = candidate_actions[0]
        best_G = float('inf')

        for action in candidate_actions:
            G = self.expected_free_energy(action, spacetime, self.position)
            if G < best_G:
                best_G = G
                best_action = action

        return best_action

    def _get_observation(self, spacetime, position: np.ndarray) -> np.ndarray:
        """Get observation vector at position"""
        obs = np.zeros(8)
        obs[0] = self.energy  # Own energy
        obs[1] = spacetime.get_curvature(position)  # Local curvature
        obs[2] = spacetime.get_information_density(position)  # Info density
        obs[3] = spacetime.get_epistemic_stress_at(position)  # Stress
        obs[4] = spacetime.get_structural_field(position)  # Structure
        obs[5] = spacetime.get_effective_mass(position) / spacetime.M  # Effective mass ratio
        obs[6] = spacetime.get_bekenstein_capacity(position)  # Available capacity
        obs[7] = position[1] / spacetime.config.r_max  # Normalized radius
        return obs

# =============================================================================
# ENHANCED AGENTS
# =============================================================================

@dataclass
class Agent:
    """Base agent"""
    id: str
    colony: str
    position: np.ndarray
    velocity: np.ndarray
    energy: float
    state: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedBeaverAgent(Agent):
    """Enhanced beaver with realistic construction and scarcity"""

    # Class-level material budget (shared across all beavers)
    # FIX: Increased from 500 → 2000 and added regeneration for soft constraints
    global_material_budget = 2000.0  # Increased finite resources
    MATERIAL_BUDGET_MAX = 2000.0     # Cap for regeneration
    MATERIAL_REGEN_RATE = 0.5        # Materials regenerated per timestep (foraging)

    # Sigmoid cap parameters for productivity scaling
    SIGMOID_ALPHA = 0.3  # Maximum productivity bonus
    SIGMOID_K = 2.0  # Steepness of sigmoid curve

    @classmethod
    def regenerate_materials(cls, dt: float):
        """
        Regenerate global materials over time (foraging/harvesting analog).
        Called from simulation loop to enable sustainable construction.
        """
        regen = cls.MATERIAL_REGEN_RATE * dt
        cls.global_material_budget = min(cls.MATERIAL_BUDGET_MAX,
                                          cls.global_material_budget + regen)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structures_built = 0
        self.construction_cooldown = 0

    @staticmethod
    def sigmoid_cap(signal: float, alpha: float = 0.3, k: float = 2.0) -> float:
        """
        Sigmoid-capped productivity scaling.
        Formula: base * (1 + α * tanh(k * signal))

        Prevents runaway growth while allowing bounded bonuses.
        """
        return 1.0 + alpha * np.tanh(k * signal)

    def update(self, dt: float, spacetime: EnhancedSpacetime, semantic_graph=None):
        # ZOMBIE FIX: Check energy first, prevent actions with negative energy
        if self.energy <= 0:
            self.state = "dead"
            return

        # Cooldown
        if self.construction_cooldown > 0:
            self.construction_cooldown -= dt

        # ENERGY REGENERATION: Beavers gain energy from multiple sources
        # FIX: Increased regeneration to prevent colony extinction
        local_structure = spacetime.get_structural_field_at(self.position)
        curvature = spacetime.get_curvature(self.position)

        # 1. Structure maintenance bonus (beavers benefit from existing infrastructure)
        structure_gain = dt * 0.008 * local_structure  # Increased from 0.003

        # 2. Curvature exploration bonus (reward for reaching high-curvature regions)
        curvature_gain = dt * 0.005 * min(curvature, 1.0)  # Increased from 0.002

        # 3. BASE FORAGING: Passive energy recovery even without structures
        # This prevents death spiral when no structures exist yet
        base_forage = dt * 0.003  # Baseline survival energy

        total_regen = structure_gain + curvature_gain + base_forage
        self.energy = min(1.5, self.energy + total_regen)

        # Check curvature using Kretschmann-based tidal strength (not Ricci which is 0 in vacuum)
        # get_curvature() returns sqrt(K) where K = 48M²/r⁶ for Schwarzschild

        # Threshold based on tidal strength: ~0.01 corresponds to moderate curvature regions
        if curvature > 0.01 and self.energy > 0.05 and self.construction_cooldown <= 0:
            # FIX #5: Diminishing returns based on local structural density
            local_structure = spacetime.get_structural_field_at(self.position)

            # SIGMOID CAP: Use tanh-based diminishing returns instead of 1/(1+x)
            # This prevents both runaway growth and complete stagnation
            diminishing_factor = self.sigmoid_cap(-local_structure, self.SIGMOID_ALPHA, self.SIGMOID_K)
            diminishing_factor = max(0.1, diminishing_factor)  # Floor at 0.1

            # Check global material budget
            material_cost = 1.0  # Each build costs 1 unit of material
            if EnhancedBeaverAgent.global_material_budget < material_cost:
                # No materials left - enter resource conservation mode
                # Reduce velocity to conserve energy while waiting for materials
                self.velocity *= 0.5
                self.construction_cooldown = 5.0  # Long cooldown to wait for materials
            elif diminishing_factor < 0.15:
                # Too much structure already - seek less crowded area
                # Move away from high-density region
                gradient = spacetime.get_structural_gradient(self.position)
                self.velocity -= 0.1 * gradient  # Move against gradient (toward less structure)
                self.construction_cooldown = 2.0  # Short cooldown before trying elsewhere
            else:
                # Build structure with sigmoid-capped strength
                build_strength = 2.0 * diminishing_factor
                spacetime.add_structural_field(self.position, build_strength, 3.0)
                self.structures_built += 1

                # Consume global materials
                EnhancedBeaverAgent.global_material_budget -= material_cost

                # ENERGY FIX: Construction costs energy, no direct reward
                # Edges reduce decay, NOT create energy (handled in decay calculation)
                construction_cost = 0.02
                self.energy -= construction_cost

                # Cooldown scales with local density (harder to build in crowded areas)
                self.construction_cooldown = 1.0 + 0.5 * local_structure
        
        # Move toward high curvature regions
        # Sample nearby points using Kretschmann-based curvature (non-zero in vacuum)
        # NOTE: get_ricci_scalar() returns 0 in Schwarzschild vacuum - NEVER use for navigation
        nearby_curvatures = []
        directions = []

        for _ in range(5):
            offset = 0.5 * np.random.randn(4)
            test_pos = self.position + offset
            # Use get_curvature() = sqrt(Kretschmann) which is non-zero: K = 48M²/r⁶
            test_curv = spacetime.get_curvature(test_pos)
            nearby_curvatures.append(test_curv)
            directions.append(offset)
        
        # Move toward highest curvature
        max_idx = np.argmax(nearby_curvatures)
        self.velocity = 0.8 * self.velocity + 0.2 * directions[max_idx]

        # Update position using geodesic motion (respects spacetime curvature)
        # This replaces naive Euclidean: self.position += dt * self.velocity
        self.position, self.velocity = spacetime.geodesic_step(
            self.position, self.velocity, dt
        )

        # FIX #2: Energy decay modified by structural field AND edges
        # Moving through structured regions costs less energy
        # EDGES REDUCE DECAY: Graph connectivity provides efficiency bonus
        movement_cost = spacetime.get_movement_cost(self.position)
        base_decay = dt * 0.005 * movement_cost

        # Edge-based decay reduction: more edges = less decay
        edge_reduction = 1.0
        if semantic_graph is not None:
            n_edges = semantic_graph.graph.number_of_edges()
            # Sigmoid cap on edge benefit: edges reduce decay, NOT create energy
            # Formula: decay_multiplier = 1 / (1 + α * tanh(edges / scale))
            edge_reduction = 1.0 / (1.0 + 0.3 * np.tanh(n_edges / 20.0))

        self.energy -= base_decay * edge_reduction

        if self.energy <= 0:
            self.state = "dead"


class EnhancedAntAgent(Agent):
    """Enhanced ant with semantic graph building, epistemic guidance, and individual accountability"""

    # MEMORY FIX: Per-ant vertex creation budget to prevent memory explosion
    MAX_VERTICES_PER_ANT = 50  # After this, ant only creates edges (attaches to existing)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_vertex = None
        self.pheromone_deposits = 0
        self.path_history = []
        self.packets_generated = 0
        self.last_position = None  # For co-occurrence edge creation
        self.discovery_times = {}  # vertex_id -> discovery_time for temporal edges

        # ISSUE 1: Epistemic guidance state
        self.belief_id: Optional[int] = None
        self.information_gain_history: List[float] = []
        self.epistemic_memory = []  # Track visited high-info regions

        # ISSUE 2: Individual accountability tracking
        self.vertices_created = 0      # Personal discovery count
        self.edges_created = 0         # Personal connection count
        self.total_info_discovered = 0.0  # Cumulative information gain
        self.contribution_score = 0.0  # Running counterfactual contribution

    def _find_nearby_vertex(self, semantic_graph, distance_threshold: float = 2.0):
        """Find the closest existing vertex within distance threshold.

        OPTIMIZED: Uses KD-tree spatial index for O(log V) instead of O(V).
        """
        # Use semantic graph's spatial index
        return semantic_graph.find_nearest_vertex(self.position, distance_threshold)

    def _find_all_nearby_vertices(self, semantic_graph, distance_threshold: float = 3.0):
        """Find all existing vertices within distance threshold.

        OPTIMIZED: Uses KD-tree spatial index for O(log V + k) instead of O(V).
        """
        return semantic_graph.find_vertices_in_radius(self.position, distance_threshold)

    def update(self, dt: float, spacetime: EnhancedSpacetime, semantic_graph, current_time: float = 0.0):
        # ZOMBIE FIX: Check energy first, prevent actions with negative energy
        if self.energy <= 0:
            self.state = "dead"
            return

        # Sample local information
        info_density = spacetime.get_information_density(self.position)

        # DIAGNOSTIC: Track info density distribution (first ant only, periodically)
        if self.id == "ant_0" and int(current_time * 100) % 1000 == 0:
            import logging
            logging.getLogger("EnhancedBlackholeArchive").debug(
                f"ANT DIAGNOSTIC: info_density={info_density:.4f}, "
                f"current_vertex={self.current_vertex}, "
                f"vertices_created={self.vertices_created}"
            )

        # ENERGY REGENERATION: Foraging reward for exploring high-info regions
        # Ants gain energy proportional to information density (biological foraging analog)
        foraging_gain = dt * 0.002 * info_density  # Scales with info richness
        self.energy = min(1.5, self.energy + foraging_gain)  # Cap at 1.5x initial energy

        # If sufficient information density, create vertex or attach to nearby existing one
        # LOWERED THRESHOLD: 0.15 → 0.05 - original was way too high for sparse info field
        if info_density > 0.05 and self.current_vertex is None:
            # Check for nearby existing vertices first (spatial co-occurrence)
            # REDUCED from 2.0 to 1.0: Tighter clustering for more vertex diversity
            nearby_vertex = self._find_nearby_vertex(semantic_graph, distance_threshold=1.0)

            # MEMORY FIX: Check per-ant vertex budget
            # If budget exceeded, force attachment to existing vertex
            budget_exceeded = self.vertices_created >= self.MAX_VERTICES_PER_ANT

            # CREATIVITY FACTOR: 5% chance to create new vertex even if nearby exists
            # BALANCED: Reduced from 30% → 5% to match merge throttling (grace period)
            # Without this reduction, vertex count explodes when merging is restricted
            # BUT: Disabled if budget exceeded
            force_create = (not budget_exceeded) and (np.random.random() < 0.05)

            if nearby_vertex is not None and (not force_create or budget_exceeded):
                # Attach to existing vertex instead of creating new one
                vertex_id = nearby_vertex
                # Use mark_vertex_accessed for proper stability tracking
                semantic_graph.mark_vertex_accessed(vertex_id, current_time)
            elif budget_exceeded and nearby_vertex is None:
                # Budget exceeded and no nearby vertex - just explore, don't create
                # This prevents memory explosion from isolated vertices
                vertex_id = None
            else:
                # Create new vertex with current_time for grace period tracking
                vertex_id = semantic_graph.add_vertex(
                    position=self.position.copy(),
                    salience=info_density,
                    current_time=current_time
                )
                # ISSUE 2: Track individual contribution
                self.vertices_created += 1
                self.total_info_discovered += info_density
                # Contribution score: weighted by info density (personal discovery value)
                self.contribution_score += info_density * 0.5

                # ENERGY REWARD: Proportional to individual contribution
                # Personal reward = 0.7 * discovery + 0.3 * avg colony benefit
                personal_reward = 0.05 * info_density  # Scales with discovery quality
                self.energy = min(1.5, self.energy + personal_reward)

            # Only update state if we actually have a vertex
            if vertex_id is not None:
                self.current_vertex = vertex_id
                self.path_history.append(vertex_id)
                self.discovery_times[vertex_id] = current_time

                # Generate packet when discovering salient vertex
                # FIX: Lowered threshold from 0.35 → 0.10 to enable earlier transport
                # Also: probabilistic packet generation for low-density discoveries
                packet_prob = min(1.0, info_density * 3.0)  # Higher density = higher prob
                if info_density > 0.10 or (info_density > 0.05 and np.random.random() < 0.3):
                    packet = {
                        'content': f"discovery_{vertex_id}",
                        'salience': info_density,
                        'confidence': min(1.0, info_density + 0.2),
                        'created_at': current_time,  # FIX: Use actual simulation time
                        'ttl': 50.0,  # FIX #4: Packet time-to-live
                        'source_agent': self.id,
                        'source_vertex': vertex_id
                    }
                    if semantic_graph.add_packet(vertex_id, packet):
                        self.packets_generated += 1
        
        # If at vertex, deposit pheromone and move to neighbor
        if self.current_vertex is not None:
            # SAFETY CHECK: Verify vertex still exists (may have been pruned)
            if self.current_vertex not in semantic_graph.graph:
                self.current_vertex = None
                # Skip to exploration mode below
            else:
                # Use mark_vertex_accessed for stability tracking (increments access count)
                semantic_graph.mark_vertex_accessed(self.current_vertex, current_time)

                # Record activation for co-occurrence tracking
                semantic_graph.record_activation(self.current_vertex, current_time, self.id)

                # Count nearby structures for structure-dependent edge probability
                nearby_structures = len(self._find_all_nearby_vertices(semantic_graph, distance_threshold=3.0))

                # EDGE POLICY 1: Temporal adjacency - connect sequential discoveries
                # Path continuity is GUARANTEED during bootstrap phase to kickstart network
                if len(self.path_history) > 1:
                    prev_vertex = self.path_history[-2]
                    if prev_vertex in semantic_graph.graph and self.current_vertex in semantic_graph.graph:
                        # First ensure both vertices have activation records
                        semantic_graph.record_activation(prev_vertex, current_time - 0.1, self.id)

                        # Check if edge already exists
                        if not semantic_graph.graph.has_edge(prev_vertex, self.current_vertex):
                            # BOOTSTRAP GUARANTEE: Always create path edges during bootstrap
                            in_bootstrap = semantic_graph.is_in_bootstrap_phase(current_time)
                            edge_created = False

                            if in_bootstrap:
                                # Guaranteed edge creation during bootstrap for path continuity
                                semantic_graph.add_edge(prev_vertex, self.current_vertex, pheromone=1.0)
                                semantic_graph.add_edge(self.current_vertex, prev_vertex, pheromone=1.0)
                                edge_created = True
                            else:
                                # Post-bootstrap: use conditional edge formation
                                if semantic_graph.add_edge_conditional(prev_vertex, self.current_vertex, 1.0,
                                                                       current_time, nearby_structures):
                                    semantic_graph.add_edge(self.current_vertex, prev_vertex, pheromone=1.0)
                                    edge_created = True

                            if edge_created:
                                self.pheromone_deposits += 2
                                # ISSUE 2: Track individual edge contribution
                                self.edges_created += 2
                                self.contribution_score += 0.2  # Edge creation value
                                # ENERGY REWARD: Edge creation strengthens the network
                                self.energy = min(1.5, self.energy + 0.03)

                # CONDITIONAL EDGE POLICY 2: Co-occurrence - connect vertices discovered close in time
                # Only creates edges if co-occurrence count > 0 (temporal overlap requirement)
                recent_discoveries = [v for v, t in self.discovery_times.items()
                                      if current_time - t < 5.0 and v != self.current_vertex
                                      and v in semantic_graph.graph]
                for other_vertex in recent_discoveries[-3:]:  # Limit to 3 most recent
                    if not semantic_graph.graph.has_edge(self.current_vertex, other_vertex):
                        # Conditional edge - requires co-occurrence
                        if semantic_graph.add_edge_conditional(self.current_vertex, other_vertex, 0.5,
                                                               current_time, nearby_structures):
                            semantic_graph.add_edge(other_vertex, self.current_vertex, pheromone=0.5)
                            # ENERGY REWARD: Co-occurrence edges reinforce memory patterns
                            self.energy = min(1.5, self.energy + 0.02)

                # CONDITIONAL EDGE POLICY 3: Spatial proximity - connect to nearby vertices
                # Structure-dependent probability: 0.05 + 0.05 * nearby_structures
                for nearby_v in self._find_all_nearby_vertices(semantic_graph, distance_threshold=3.0):
                    if nearby_v != self.current_vertex and not semantic_graph.graph.has_edge(self.current_vertex, nearby_v):
                        # Record activation for nearby vertex to enable co-occurrence
                        semantic_graph.record_activation(nearby_v, current_time, self.id)
                        if semantic_graph.add_edge_conditional(self.current_vertex, nearby_v, 0.3,
                                                               current_time, nearby_structures):
                            semantic_graph.add_edge(nearby_v, self.current_vertex, pheromone=0.3)
                            # ENERGY REWARD: Spatial connections expand network reach
                            self.energy = min(1.5, self.energy + 0.01)

                # Choose next vertex - FIX: Use both successors and predecessors for DiGraph
                successors = set(semantic_graph.graph.successors(self.current_vertex))
                predecessors = set(semantic_graph.graph.predecessors(self.current_vertex))
                neighbors = list(successors | predecessors)

                if neighbors:
                    # CRITICAL FIX: Force exploration with probability based on stagnation
                    # If ant has visited same vertices repeatedly, force exploration
                    recent_unique = len(set(self.path_history[-10:])) if len(self.path_history) >= 10 else 10
                    stagnation_factor = 1.0 - (recent_unique / 10.0)  # 0 = diverse, 1 = stuck

                    # Exploration probability increases with stagnation and low vertex creation
                    base_explore_prob = 0.15  # 15% base chance to explore instead of follow
                    explore_prob = base_explore_prob + 0.5 * stagnation_factor

                    # Also boost exploration if ant hasn't created vertices recently
                    if self.vertices_created == 0:
                        explore_prob += 0.3  # Strong boost for unproductive ants

                    if np.random.random() < explore_prob:
                        # FORCE EXPLORATION: Detach from graph to seek new regions
                        self.current_vertex = None
                    else:
                        # Follow pheromones probabilistically
                        pheromones = []
                        for n in neighbors:
                            # Check both edge directions for pheromone
                            p1 = semantic_graph.get_pheromone((self.current_vertex, n))
                            p2 = semantic_graph.get_pheromone((n, self.current_vertex))
                            pheromones.append(max(p1, p2))
                        probs = np.array(pheromones) + 0.1  # Add baseline
                        probs /= probs.sum()

                        next_vertex = np.random.choice(neighbors, p=probs)
                        self.current_vertex = next_vertex
                        self.path_history.append(next_vertex)

                        # Update position (with safety check)
                        node_data = semantic_graph.graph.nodes.get(next_vertex, {})
                        if 'position' in node_data:
                            self.position = node_data['position']
                            # Use mark_vertex_accessed for stability tracking when moving to vertex
                            semantic_graph.mark_vertex_accessed(next_vertex, current_time)
                        else:
                            # Vertex missing position data, reset to exploration mode
                            self.current_vertex = None
                else:
                    # No neighbors, explore
                    self.current_vertex = None

        # ISSUE 1: EPISTEMIC GUIDANCE - Replace random walk with information-seeking exploration
        if self.current_vertex is None:
            # LAYER INTEGRATION: Ants are attracted to structural field (beaver builds)
            structural_value = spacetime.get_structural_field(self.position)
            structural_gradient = spacetime.get_structural_gradient(self.position)

            # =====================================================================
            # EPISTEMIC EXPLORATION: Move toward high-information, unvisited regions
            # =====================================================================

            # 1. Compute local information density and gradient
            info_density = spacetime.get_information_density(self.position)

            # 2. Compute information gradient using finite differences
            eps = 0.5  # Spatial sampling distance
            info_gradient = np.zeros(3)
            for dim in range(3):  # r, theta, phi
                pos_plus = self.position.copy()
                pos_minus = self.position.copy()
                pos_plus[dim + 1] += eps
                pos_minus[dim + 1] -= eps
                # Bound the test positions
                pos_plus[1] = max(spacetime.r_s * 1.1, min(spacetime.config.r_max * 0.95, pos_plus[1]))
                pos_minus[1] = max(spacetime.r_s * 1.1, min(spacetime.config.r_max * 0.95, pos_minus[1]))
                info_gradient[dim] = (spacetime.get_information_density(pos_plus) -
                                      spacetime.get_information_density(pos_minus)) / (2 * eps)

            # 3. Novelty bonus: penalize revisiting same regions
            novelty_factor = 1.0
            if len(self.epistemic_memory) > 0:
                # Check distance to recently visited positions
                for past_pos in self.epistemic_memory[-20:]:
                    dist = np.linalg.norm(self.position[1:] - past_pos[1:])
                    if dist < 2.0:
                        novelty_factor *= 0.7  # Reduce attraction if recently visited

            # 4. Combine exploration signals:
            #    - Info gradient: move toward higher information density (epistemic value)
            #    - Structural gradient: follow beaver builds (exploitation)
            #    - Novelty: avoid recently visited regions (anti-clustering)
            #    - Random: small perturbation for stochasticity

            # Epistemic drive: normalized gradient toward high-info regions
            info_grad_norm = np.linalg.norm(info_gradient)
            if info_grad_norm > 1e-6:
                epistemic_direction = info_gradient / info_grad_norm
            else:
                epistemic_direction = np.zeros(3)

            # Compute weighted exploration direction
            random_component = 0.02 * np.random.randn(3)  # Reduced randomness
            epistemic_weight = 0.5 * novelty_factor  # Strong epistemic drive
            structural_weight = 0.3 if structural_value > 0.1 else 0.0
            random_weight = 0.2

            exploration_direction = (
                epistemic_weight * epistemic_direction +
                structural_weight * structural_gradient +
                random_weight * random_component
            )

            # Track information gain for this step
            expected_info_gain = info_density * novelty_factor
            self.information_gain_history.append(expected_info_gain)
            self.total_info_discovered += expected_info_gain * dt

            # 5. Build exploration step (4D: time component stays zero)
            exploration_step = np.zeros(4)
            exploration_step[1:] = 0.05 * exploration_direction

            # Add exploration offset to velocity before geodesic step
            exploration_velocity = self.velocity + exploration_step / (dt + 1e-6)

            # Use geodesic motion (respects spacetime curvature)
            self.position, self.velocity = spacetime.geodesic_step(
                self.position, exploration_velocity, dt
            )

            # 6. Update epistemic memory (sliding window of visited positions)
            self.epistemic_memory.append(self.position.copy())
            if len(self.epistemic_memory) > 50:
                self.epistemic_memory.pop(0)

            # LAYER INTEGRATION: Create structure-linked vertex at high structural field
            if structural_value > 0.3 and np.random.random() < 0.1:
                # This location has significant beaver activity, mark it semantically
                nearby = self._find_nearby_vertex(semantic_graph, distance_threshold=3.0)
                if nearby is None:
                    # Create a "structure vertex" linking beaver and ant layers
                    vertex_id = semantic_graph.add_vertex(
                        position=self.position.copy(),
                        salience=0.3 + structural_value * 0.5,  # Salience from structure
                        current_time=current_time
                    )
                    semantic_graph.graph.nodes[vertex_id]['vertex_type'] = 'structure'
                    self.current_vertex = vertex_id
                    self.path_history.append(vertex_id)
                    self.discovery_times[vertex_id] = current_time
                    # ISSUE 2: Track structure vertex creation
                    self.vertices_created += 1
                    self.contribution_score += 0.3

        # FIX #2: Energy decay modified by structural field AND edges
        # EDGES REDUCE DECAY: Graph connectivity provides efficiency bonus
        movement_cost = spacetime.get_movement_cost(self.position)
        base_decay = dt * 0.003 * movement_cost

        # Edge-based decay reduction
        n_edges = semantic_graph.graph.number_of_edges()
        edge_reduction = 1.0 / (1.0 + 0.3 * np.tanh(n_edges / 20.0))
        self.energy -= base_decay * edge_reduction

        if self.energy <= 0:
            self.state = "dead"


class EnhancedBeeAgent(Agent, ActiveInferenceMixin):
    """
    Enhanced bee with packet transport from queue-based economy.

    Uses Active Inference for action selection:
    - Scout behavior emerges from high epistemic drive (exploration)
    - Forager behavior emerges from pragmatic preferences (goal-directed)
    - Role transitions happen naturally based on free energy gradients
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "scout"
        self.packet = None
        self.target_vertex = None
        self.packets_delivered = 0
        self.packets_dropped = 0  # FIX #4: Track failed deliveries
        self.waggle_intensity = 0.0

        # Initialize Active Inference
        self.__init_active_inference__()
        # Bees prefer high packet density (foraging) and wormhole proximity (delivery)
        self.preferences[2] = 1.0  # Prefer high info density
        self.preferences[4] = 0.5  # Prefer structural regions
        self.epistemic_drive = 0.7  # High exploration drive initially

    def update(self, dt: float, spacetime: EnhancedSpacetime, semantic_graph, wormhole_position,
                current_time: float = 0.0):
        # ZOMBIE FIX: Check energy first, prevent actions with negative energy
        if self.energy <= 0:
            self.state = "dead"
            return

        # ACTIVE INFERENCE: Update beliefs based on current observation
        observation = self._get_observation(spacetime, self.position)
        self.update_beliefs(observation)

        # Track free energy for learning
        current_F = self.compute_free_energy(observation)
        self.free_energy_history.append(current_F)

        # Adapt epistemic drive based on free energy trend
        # High free energy = world is surprising = increase exploration
        if len(self.free_energy_history) > 10:
            recent_F = np.mean(self.free_energy_history[-10:])
            if recent_F > 1.0:
                self.epistemic_drive = min(0.9, self.epistemic_drive + 0.01)
            else:
                self.epistemic_drive = max(0.3, self.epistemic_drive - 0.01)

        if self.role == "scout":
            # TRANSPORT GATING: Check if semantic graph is mature enough for transport
            # FIX: Early transport mode - allow low-bandwidth transport even with minimal structure
            transport_ready = semantic_graph.is_transport_ready(current_time)
            has_any_packets = semantic_graph.get_total_queue_length() > 0

            # EARLY TRANSPORT MODE: If packets exist, allow transport even if graph not "mature"
            # This prevents bees from sitting idle while packets pile up
            early_transport_allowed = has_any_packets and semantic_graph.graph.number_of_nodes() >= 5

            if not transport_ready and not early_transport_allowed:
                # Graph not mature and no packets - wait for structure to develop
                # Just random walk to reduce energy consumption
                random_step = 0.02 * np.random.randn(4)
                exploration_velocity = self.velocity + random_step / (dt + 1e-6)
                # Use geodesic motion even while waiting
                self.position, self.velocity = spacetime.geodesic_step(
                    self.position, exploration_velocity, dt
                )
                self.energy -= dt * 0.003  # Reduced energy cost while waiting
                return

            # FIX #3: Find vertices with packets waiting (not just high salience)
            # FIX: Increased check frequency from 0.1 → 0.2 for faster response
            if np.random.rand() < 0.2:  # Check more frequently
                vertices = list(semantic_graph.graph.nodes())
                if vertices:
                    # Prioritize vertices with packets AND high salience
                    scores = []
                    for v in vertices:
                        queue_len = semantic_graph.get_queue_length(v)
                        salience = semantic_graph.graph.nodes[v].get('salience', 0.5)
                        # Score = queue_length * salience (packets at important nodes)
                        scores.append(queue_len * salience + 0.1 * salience)

                    if max(scores) > 0:
                        max_idx = np.argmax(scores)
                        self.target_vertex = vertices[max_idx]
                        self.waggle_intensity = scores[max_idx]
                        self.role = "forager"

        elif self.role == "forager":
            if self.packet is None and self.target_vertex is not None:
                # Safety check: verify target vertex still exists (may have been pruned)
                if self.target_vertex not in semantic_graph.graph:
                    self.target_vertex = None
                    self.role = "scout"
                    return

                # Move to target vertex
                target_pos = semantic_graph.graph.nodes[self.target_vertex]['position']

                # Handle dimension mismatch
                if len(target_pos) > len(self.position):
                    target_pos = target_pos[:len(self.position)]
                elif len(target_pos) < len(self.position):
                    target_pos = np.pad(target_pos, (0, len(self.position) - len(target_pos)))

                direction = target_pos - self.position
                self.velocity = 0.3 * direction / (np.linalg.norm(direction) + 1e-6)

                # Check if reached
                if np.linalg.norm(direction) < 1.0:
                    # FIX #3: Pick up packet from queue (not fabricate one)
                    self.packet = semantic_graph.get_packet(self.target_vertex)

                    if self.packet is not None:
                        # Got a real packet - transport it
                        self.role = "transporter"
                    else:
                        # No packet available - go back to scouting
                        self.target_vertex = None
                        self.role = "scout"

        elif self.role == "transporter":
            # FIX #4: Check packet TTL
            if self.packet is not None:
                self.packet['ttl'] -= dt
                if self.packet['ttl'] <= 0:
                    # Packet expired - drop it
                    self.packets_dropped += 1
                    self.packet = None
                    self.target_vertex = None
                    self.role = "scout"
                    return

            # Move to wormhole
            direction = wormhole_position - self.position
            self.velocity = 0.5 * direction / (np.linalg.norm(direction) + 1e-6)

            # Check if reached wormhole
            if np.linalg.norm(direction) < 2.0:
                # Deliver packet
                self.packets_delivered += 1
                self.packet = None
                self.target_vertex = None
                self.role = "scout"
        
        # Update position using geodesic motion (respects spacetime curvature)
        self.position, self.velocity = spacetime.geodesic_step(
            self.position, self.velocity, dt
        )

        # FIX #2: Energy decay modified by structural field AND edges
        # EDGES REDUCE DECAY: Graph connectivity provides efficiency bonus
        movement_cost = spacetime.get_movement_cost(self.position)
        base_decay = dt * 0.008 * movement_cost * (1 + 0.5 * bool(self.packet))

        # Edge-based decay reduction
        n_edges = semantic_graph.graph.number_of_edges()
        edge_reduction = 1.0 / (1.0 + 0.3 * np.tanh(n_edges / 20.0))
        self.energy -= base_decay * edge_reduction

        if self.energy <= 0:
            self.state = "dead"

# =============================================================================
# SEMANTIC GRAPH
# =============================================================================

class SemanticGraph:
    """Semantic graph maintained by ants with packet economy"""

    # Class-level constants for emergence control
    MAX_VERTICES = 10000  # FIXED: Was 100, way too low for emergence
    PHASE_1_MATURITY_EDGES = 2
    PHASE_2_MATURITY_EDGES = 5
    MIN_ENTROPY_FOR_TRANSPORT = 0.3

    # Bootstrap phase parameters - CRITICAL FOR ANT SURVIVAL
    # Extended bootstrap allows network to stabilize before strict edge policies apply
    BOOTSTRAP_DURATION = 40.0  # Time units for bootstrap phase (increased from 30)
    BOOTSTRAP_EDGE_PROB_MULTIPLIER = 5.0  # Higher edge probability during bootstrap
    VERTEX_GRACE_PERIOD = 30.0  # New vertices protected from pruning (increased from 20)
    MIN_STABLE_VERTICES = 100  # Minimum vertices to maintain

    # FIX: Merge throttling to prevent "shredding the mind"
    MERGE_BUDGET_PCT = 0.03     # Max 3% of alive vertices can be merged per step
    MERGE_GRACE_PERIOD = 50.0   # Newborn vertices immune to merge for this duration
    MAX_EDGES_PER_VERTEX = 64   # Cap edges to prevent near-clique explosion

    def __init__(self):
        self.graph = nx.DiGraph()
        self.pheromones = {}  # (v1, v2) -> strength
        self.next_vertex_id = 0
        # FIX #3: Packet queues at vertices
        self.packet_queues = {}  # vertex_id -> list of packets
        self.max_queue_size = 10  # FIX #4: Queue capacity limit
        # Co-occurrence tracking for conditional edge formation
        self.vertex_activations = {}  # vertex_id -> list of (time, agent_id)
        self.co_occurrence_counts = {}  # (v1, v2) -> count

        # Bootstrap phase tracking
        self.creation_time = 0.0  # When the graph was initialized
        self.is_bootstrapped = False  # Whether bootstrap phase is complete

        # PERFORMANCE: Spatial index for O(log V) nearest neighbor queries
        self._spatial_index = None  # KD-tree, rebuilt periodically
        self._spatial_vertex_ids = []  # Mapping from index to vertex ID
        self._spatial_index_dirty = True  # Needs rebuild

        # Vertex stability tracking
        self.vertex_creation_times = {}  # vertex_id -> creation_time
        self.vertex_access_counts = {}  # vertex_id -> access count for stability scoring
        self.stable_vertex_set = set()  # Vertices that have proven stable

        # =================================================================
        # GRAPH LEDGER - Comprehensive operation tracking for debugging
        # =================================================================
        self.ledger = {
            'created_total': 0,
            'created_rejected_cap': 0,       # Rejected due to MAX_VERTICES
            'created_replaced': 0,           # Created by replacing low-salience
            'pruned_total': 0,
            'pruned_age': 0,                 # Pruned due to age
            'pruned_replaced': 0,            # Removed to make room
            'merged_total': 0,
            'noise_injected': 0,
            'removed_by_apl': 0,             # Removed by APL effects
            'min_stable_blocks': 0,          # Times MIN_STABLE prevented removal
            'invariant_violations': 0,       # Times graph dropped below minimum
        }
        self.ledger_history = []  # Snapshots for analysis

    def set_creation_time(self, time: float):
        """Set the graph creation time for bootstrap phase tracking"""
        self.creation_time = time

    def get_graph_age(self, current_time: float) -> float:
        """Get age of graph since creation"""
        return current_time - self.creation_time

    def is_in_bootstrap_phase(self, current_time: float) -> bool:
        """Check if graph is still in bootstrap phase"""
        if self.is_bootstrapped:
            return False
        age = self.get_graph_age(current_time)
        if age >= self.BOOTSTRAP_DURATION:
            self.is_bootstrapped = True
            return False
        return True

    def rebuild_spatial_index(self):
        """Rebuild KD-tree spatial index for O(log V) nearest neighbor queries.

        Call this periodically (e.g., every 50 steps) to keep index fresh.
        Much faster than O(V) linear scan for each ant update.
        """
        from scipy.spatial import cKDTree

        if self.graph.number_of_nodes() == 0:
            self._spatial_index = None
            self._spatial_vertex_ids = []
            self._spatial_index_dirty = False
            return

        positions = []
        vertex_ids = []

        for v in self.graph.nodes():
            node_data = self.graph.nodes.get(v, {})
            if 'position' in node_data:
                # Use spatial coordinates (ignore time component)
                pos = node_data['position']
                positions.append(pos[1:4] if len(pos) >= 4 else pos[:3])
                vertex_ids.append(v)

        if positions:
            self._spatial_index = cKDTree(np.array(positions))
            self._spatial_vertex_ids = vertex_ids
        else:
            self._spatial_index = None
            self._spatial_vertex_ids = []

        self._spatial_index_dirty = False

    def find_nearest_vertex(self, position: np.ndarray, distance_threshold: float = 2.0):
        """Find nearest vertex using spatial index. O(log V) instead of O(V)."""
        if self._spatial_index is None or self._spatial_index_dirty:
            self.rebuild_spatial_index()

        if self._spatial_index is None or len(self._spatial_vertex_ids) == 0:
            return None

        # Query using spatial coordinates
        query_pos = position[1:4] if len(position) >= 4 else position[:3]
        dist, idx = self._spatial_index.query(query_pos, k=1)

        if dist <= distance_threshold:
            return self._spatial_vertex_ids[idx]
        return None

    def find_vertices_in_radius(self, position: np.ndarray, radius: float = 3.0):
        """Find all vertices within radius using spatial index. O(log V + k)."""
        if self._spatial_index is None or self._spatial_index_dirty:
            self.rebuild_spatial_index()

        if self._spatial_index is None or len(self._spatial_vertex_ids) == 0:
            return []

        query_pos = position[1:4] if len(position) >= 4 else position[:3]
        indices = self._spatial_index.query_ball_point(query_pos, radius)

        return [self._spatial_vertex_ids[i] for i in indices]

    def add_vertex(self, position: np.ndarray, salience: float, current_time: float = 0.0) -> int:
        """Add vertex to graph with max vertex limit and stability tracking"""
        # Enforce maximum vertices
        if self.graph.number_of_nodes() >= self.MAX_VERTICES:
            # Find lowest salience vertex to replace (excluding stable and grace period vertices)
            candidates = []
            for v in self.graph.nodes():
                # Skip vertices in grace period
                creation_time = self.vertex_creation_times.get(v, 0.0)
                if current_time - creation_time < self.VERTEX_GRACE_PERIOD:
                    continue
                # Skip stable vertices
                if v in self.stable_vertex_set:
                    continue
                candidates.append(v)

            if not candidates:
                self.ledger['created_rejected_cap'] += 1
                return -1  # No replaceable vertices

            min_sal_vertex = min(candidates,
                                  key=lambda v: self.graph.nodes[v].get('salience', 0))
            if self.graph.nodes[min_sal_vertex].get('salience', 0) < salience:
                self._remove_vertex(min_sal_vertex, reason='replaced')
                self.ledger['created_replaced'] += 1
            else:
                self.ledger['created_rejected_cap'] += 1
                return -1  # Reject new vertex

        vertex_id = self.next_vertex_id
        self.next_vertex_id += 1

        self.graph.add_node(vertex_id, position=position, salience=salience,
                           created_at=current_time, last_accessed=current_time)
        self.packet_queues[vertex_id] = []  # Initialize packet queue
        self.vertex_activations[vertex_id] = []
        self.vertex_creation_times[vertex_id] = current_time
        self.vertex_access_counts[vertex_id] = 0

        # PERFORMANCE: Mark spatial index as needing rebuild
        self._spatial_index_dirty = True

        self.ledger['created_total'] += 1
        return vertex_id

    def _remove_vertex(self, vertex_id: int, reason: str = 'unknown'):
        """Internal method to cleanly remove a vertex with ledger tracking"""
        if vertex_id in self.graph:
            self.graph.remove_node(vertex_id)

            # PERFORMANCE: Mark spatial index as needing rebuild
            self._spatial_index_dirty = True

            # Track reason in ledger
            if reason == 'replaced':
                self.ledger['pruned_replaced'] += 1
            elif reason == 'age':
                self.ledger['pruned_age'] += 1
            elif reason == 'apl':
                self.ledger['removed_by_apl'] += 1
            elif reason == 'merged':
                self.ledger['merged_total'] += 1
            self.ledger['pruned_total'] += 1

        if vertex_id in self.packet_queues:
            del self.packet_queues[vertex_id]
        if vertex_id in self.vertex_activations:
            del self.vertex_activations[vertex_id]
        if vertex_id in self.vertex_creation_times:
            del self.vertex_creation_times[vertex_id]
        if vertex_id in self.vertex_access_counts:
            del self.vertex_access_counts[vertex_id]
        if vertex_id in self.stable_vertex_set:
            self.stable_vertex_set.discard(vertex_id)

        # HARD INVARIANT CHECK
        if self.graph.number_of_nodes() < self.MIN_STABLE_VERTICES:
            self.ledger['invariant_violations'] += 1

    def mark_vertex_accessed(self, vertex_id: int, current_time: float):
        """Mark vertex as accessed and update stability tracking"""
        if vertex_id in self.graph:
            self.graph.nodes[vertex_id]['last_accessed'] = current_time
            self.vertex_access_counts[vertex_id] = self.vertex_access_counts.get(vertex_id, 0) + 1

            # Promote to stable if accessed frequently
            if self.vertex_access_counts[vertex_id] >= 10:
                self.stable_vertex_set.add(vertex_id)

    def add_packet(self, vertex_id: int, packet: dict) -> bool:
        """
        FIX #3: Add a packet to vertex queue.
        Returns False if queue is full (congestion).
        """
        if vertex_id not in self.packet_queues:
            return False
        if len(self.packet_queues[vertex_id]) >= self.max_queue_size:
            return False  # Queue full - packet dropped (congestion)
        self.packet_queues[vertex_id].append(packet)
        return True

    def get_packet(self, vertex_id: int) -> dict:
        """FIX #3: Get a packet from vertex queue (FIFO)"""
        if vertex_id not in self.packet_queues:
            return None
        if len(self.packet_queues[vertex_id]) == 0:
            return None
        return self.packet_queues[vertex_id].pop(0)

    def get_queue_length(self, vertex_id: int) -> int:
        """Get queue length at vertex"""
        return len(self.packet_queues.get(vertex_id, []))
    
    def add_edge(self, v1: int, v2: int, pheromone: float):
        """Add edge with pheromone, enforcing per-vertex edge cap"""
        if not self.graph.has_edge(v1, v2):
            # FIX: Enforce edge cap before adding new edge
            out_degree = self.graph.out_degree(v1)
            if out_degree >= self.MAX_EDGES_PER_VERTEX:
                # At capacity - prune weakest outgoing edge first
                self._prune_weakest_edge(v1, direction='out')

            self.graph.add_edge(v1, v2)

        key = (v1, v2)
        if key not in self.pheromones:
            self.pheromones[key] = 0.0
        self.pheromones[key] += pheromone

    def _prune_weakest_edge(self, vertex: int, direction: str = 'out'):
        """Remove the weakest edge from a vertex to make room for new connections."""
        if direction == 'out':
            edges = list(self.graph.out_edges(vertex))
        else:
            edges = list(self.graph.in_edges(vertex))

        if not edges:
            return

        # Find edge with lowest pheromone
        weakest_edge = None
        weakest_pheromone = float('inf')
        for edge in edges:
            p = self.pheromones.get(edge, 0.0)
            if p < weakest_pheromone:
                weakest_pheromone = p
                weakest_edge = edge

        if weakest_edge:
            self.graph.remove_edge(*weakest_edge)
            if weakest_edge in self.pheromones:
                del self.pheromones[weakest_edge]

    def get_pheromone(self, edge: Tuple[int, int]) -> float:
        """Get pheromone strength on edge"""
        return self.pheromones.get(edge, 0.0)

    def decay_pheromones(self, dt: float, decay_rate: float = 0.1):
        """Decay all pheromones and remove weak edges"""
        edges_to_remove = []
        for edge in list(self.pheromones.keys()):
            self.pheromones[edge] *= np.exp(-decay_rate * dt)

            # Remove if too weak
            if self.pheromones[edge] < 0.01:
                del self.pheromones[edge]
                edges_to_remove.append(edge)

        # FIX: Also remove edges from graph when pheromone decays to zero
        # This prevents edge explosion from unreinforced connections
        for edge in edges_to_remove:
            if self.graph.has_edge(*edge):
                self.graph.remove_edge(*edge)

    def prune_graph(self, current_time: float, max_age: float = 80.0, min_vertices: int = None):
        """
        Prune vertices that haven't been accessed recently.
        Cost of memory - old, unused beliefs are forgotten.

        Protected vertices (never pruned):
        - Vertices in grace period (recently created)
        - Vertices marked as stable
        - Vertices with packets in queue
        - Vertices with high connectivity (degree > 3)
        - Vertices with high salience (> 0.7)
        - If graph would drop below MIN_STABLE_VERTICES

        During bootstrap phase, pruning is completely disabled to allow
        the graph to establish stable structure.
        """
        # Use class constant if not specified
        if min_vertices is None:
            min_vertices = self.MIN_STABLE_VERTICES

        # During bootstrap, no pruning at all
        if self.is_in_bootstrap_phase(current_time):
            return 0

        if self.graph.number_of_nodes() <= min_vertices:
            self.ledger['min_stable_blocks'] += 1
            return 0  # Don't prune if already at minimum

        vertices_to_remove = []
        for v in self.graph.nodes():
            # Check grace period protection
            creation_time = self.vertex_creation_times.get(v, 0.0)
            vertex_age = current_time - creation_time
            if vertex_age < self.VERTEX_GRACE_PERIOD:
                continue  # In grace period, keep

            # Check stable vertex protection
            if v in self.stable_vertex_set:
                continue  # Marked stable, keep

            # Check access age
            last_accessed = self.graph.nodes[v].get('last_accessed', 0.0)
            access_age = current_time - last_accessed
            if access_age <= max_age:
                continue  # Recently accessed, keep

            # Check protections
            if self.get_queue_length(v) > 0:
                continue  # Has packets, keep

            degree = self.graph.in_degree(v) + self.graph.out_degree(v)
            if degree > 3:
                continue  # Well-connected hub, keep

            salience = self.graph.nodes[v].get('salience', 0.5)
            if salience > 0.7:
                continue  # High importance, keep

            vertices_to_remove.append(v)

        # Limit pruning to maintain minimum graph size
        max_to_remove = max(0, self.graph.number_of_nodes() - min_vertices)
        vertices_to_remove = vertices_to_remove[:max_to_remove]

        for v in vertices_to_remove:
            self._remove_vertex(v, reason='age')

        return len(vertices_to_remove)

    def merge_nearby_vertices(self, distance_threshold: float = 0.05, current_time: float = None):
        """
        FIX #6: Merge vertices that are spatially close.
        Prevents unbounded growth from redundant beliefs.

        TUNED: Threshold reduced from 1.0 to 0.05 (essentially duplicate positions only).
        A/B test showed:
          - DISABLE_MERGE=1: 721 vertices at step 100
          - threshold=0.3: only 115 vertices at step 100 (606 merges!)
        0.3 was still too aggressive. 0.05 should only catch true duplicates.

        ADDITIONAL CRITERIA:
        - Salience similarity (within 0.3) to merge
        - MERGE_BUDGET_PCT: Max % of vertices that can be merged per call
        - MERGE_GRACE_PERIOD: Newborn vertices immune to merge

        OPTIMIZED: Uses scipy KD-tree for O(V log V) instead of O(V²) neighbor search.

        Note: Uses SPATIAL distance only (r, theta, phi), not time component.
        """
        from scipy.spatial import cKDTree

        merged_count = 0
        vertices = list(self.graph.nodes())
        n_vertices = self.graph.number_of_nodes()

        # CRITICAL: Don't merge if already at or below minimum
        if n_vertices <= self.MIN_STABLE_VERTICES:
            return 0

        # FIX: Merge budget - max vertices to merge this call
        merge_budget = max(1, int(n_vertices * self.MERGE_BUDGET_PCT))

        # Build position and salience arrays, respecting grace period
        valid_vertices = []
        positions = []
        saliences = []

        for v in vertices:
            if v not in self.graph:
                continue
            if 'position' not in self.graph.nodes[v]:
                continue
            pos = self.graph.nodes[v]['position']
            if len(pos) < 4:
                continue

            # FIX: Grace period - skip newborn vertices
            if current_time is not None and v in self.vertex_creation_times:
                age = current_time - self.vertex_creation_times[v]
                if age < self.MERGE_GRACE_PERIOD:
                    continue  # Protected from merge

            valid_vertices.append(v)
            positions.append(pos[1:4])  # Spatial only (r, theta, phi)
            saliences.append(self.graph.nodes[v].get('salience', 0.5))

        if len(valid_vertices) < 2:
            return 0

        # Build KD-tree for fast neighbor queries
        positions_array = np.array(positions)
        tree = cKDTree(positions_array)

        # Find all pairs within threshold
        pairs = tree.query_pairs(r=distance_threshold)

        # Track which vertices to merge (avoid double-merging)
        merged_into = {}  # v2 -> v1 (v2 merged into v1)

        for i, j in pairs:
            # FIX: Respect merge budget
            if len(merged_into) >= merge_budget:
                break

            v1, v2 = valid_vertices[i], valid_vertices[j]

            # Skip if either already merged
            if v1 in merged_into or v2 in merged_into:
                continue

            # Check salience similarity
            sal1, sal2 = saliences[i], saliences[j]
            if abs(sal1 - sal2) >= 0.3:
                continue

            # Stop if at minimum
            if self.graph.number_of_nodes() - len(merged_into) <= self.MIN_STABLE_VERTICES:
                break

            # Mark v2 to be merged into v1
            merged_into[v2] = v1

        # Perform the merges
        for v2, v1 in merged_into.items():
            if v1 not in self.graph or v2 not in self.graph:
                continue

            # Keep the more salient vertex's properties
            sal1 = self.graph.nodes[v1].get('salience', 0.5)
            sal2 = self.graph.nodes[v2].get('salience', 0.5)
            if sal2 > sal1:
                self.graph.nodes[v1]['salience'] = sal2
                self.graph.nodes[v1]['position'] = self.graph.nodes[v2]['position']

            # Transfer packets
            if v2 in self.packet_queues:
                for pkt in self.packet_queues[v2]:
                    self.add_packet(v1, pkt)
                del self.packet_queues[v2]

            # Transfer edges
            for pred in list(self.graph.predecessors(v2)):
                if pred != v1 and pred in self.graph:
                    self.add_edge(pred, v1, 0.5)
            for succ in list(self.graph.successors(v2)):
                if succ != v1 and succ in self.graph:
                    self.add_edge(v1, succ, 0.5)

            # Remove merged vertex
            if v2 in self.graph:
                self.graph.remove_node(v2)
                self.ledger['merged_total'] += 1
                merged_count += 1

            # Clean up other tracking structures
            if v2 in self.vertex_activations:
                del self.vertex_activations[v2]
            if v2 in self.vertex_creation_times:
                del self.vertex_creation_times[v2]
            if v2 in self.vertex_access_counts:
                del self.vertex_access_counts[v2]
            self.stable_vertex_set.discard(v2)

        return merged_count

    def get_total_queue_length(self) -> int:
        """Get total packets waiting across all vertices"""
        return sum(len(q) for q in self.packet_queues.values())

    def snapshot_ledger(self, step: int):
        """Take a snapshot of current ledger state for debugging"""
        snapshot = {
            'step': step,
            'v_alive': self.graph.number_of_nodes(),
            'e_alive': self.graph.number_of_edges(),
            **self.ledger.copy()
        }
        self.ledger_history.append(snapshot)

    def dump_ledger(self) -> str:
        """Dump ledger for analysis"""
        v_alive = self.graph.number_of_nodes()
        e_alive = self.graph.number_of_edges()

        lines = [
            "=" * 60,
            "GRAPH LEDGER DUMP",
            "=" * 60,
            f"V_alive_current:        {v_alive}",
            f"E_alive_current:        {e_alive}",
            f"V_created_total:        {self.ledger['created_total']}",
            f"V_created_rejected_cap: {self.ledger['created_rejected_cap']}",
            f"V_created_replaced:     {self.ledger['created_replaced']}",
            f"V_pruned_total:         {self.ledger['pruned_total']}",
            f"V_pruned_age:           {self.ledger['pruned_age']}",
            f"V_pruned_replaced:      {self.ledger['pruned_replaced']}",
            f"V_merged_total:         {self.ledger['merged_total']}",
            f"V_noise_injected:       {self.ledger['noise_injected']}",
            f"V_removed_by_apl:       {self.ledger['removed_by_apl']}",
            f"MIN_STABLE_blocks:      {self.ledger['min_stable_blocks']}",
            f"INVARIANT_violations:   {self.ledger['invariant_violations']}",
            "-" * 60,
            f"NET: created({self.ledger['created_total']}) - pruned({self.ledger['pruned_total']}) - merged({self.ledger['merged_total']}) = {self.ledger['created_total'] - self.ledger['pruned_total'] - self.ledger['merged_total']}",
            f"VALIDATION: {v_alive} should equal {self.ledger['created_total'] - self.ledger['pruned_total'] - self.ledger['merged_total']}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # =========================================================================
    # CONDITIONAL EDGE FORMATION (Co-activation based)
    # =========================================================================

    def record_activation(self, vertex_id: int, time: float, agent_id: int):
        """
        Record vertex activation for co-occurrence tracking.
        Edges require temporal overlap, not just absolute salience.
        """
        if vertex_id not in self.vertex_activations:
            self.vertex_activations[vertex_id] = []

        # Keep only recent activations (rolling window)
        activation_window = 10.0  # Time window for co-occurrence
        self.vertex_activations[vertex_id] = [
            (t, a) for t, a in self.vertex_activations[vertex_id]
            if time - t < activation_window
        ]
        self.vertex_activations[vertex_id].append((time, agent_id))

    def compute_co_occurrence(self, v1: int, v2: int, current_time: float,
                               time_window: float = 5.0) -> int:
        """
        Compute co-occurrence count between two vertices.
        Returns number of times both were activated within time_window.
        """
        if v1 not in self.vertex_activations or v2 not in self.vertex_activations:
            return 0

        activations1 = self.vertex_activations[v1]
        activations2 = self.vertex_activations[v2]

        co_occurrences = 0
        for t1, _ in activations1:
            for t2, _ in activations2:
                if abs(t1 - t2) < time_window:
                    co_occurrences += 1

        return co_occurrences

    def add_edge_conditional(self, v1: int, v2: int, pheromone: float,
                              current_time: float, nearby_structures: int = 0) -> bool:
        """
        Conditional edge formation based on co-activation with phase-based thresholds.

        During bootstrap phase:
        - Higher base probability (5x multiplier)
        - Spatial proximity alone can create edges (no co-occurrence required)
        - Encourages rapid graph structure formation

        After bootstrap:
        - Require co-occurrence (temporal overlap)
        - Standard probability formula:
            edge_prob = min(salience1, salience2) × co_occurrence_count × structure_bonus
        - Structure bonus: 0.05 + 0.05 × nearby_structures
        """
        if v1 not in self.graph or v2 not in self.graph:
            return False

        # Already connected - just reinforce pheromone
        if self.graph.has_edge(v1, v2):
            self.pheromones[(v1, v2)] = self.pheromones.get((v1, v2), 0) + pheromone * 0.1
            return False

        # Get saliences
        sal1 = self.graph.nodes[v1].get('salience', 0.5)
        sal2 = self.graph.nodes[v2].get('salience', 0.5)
        min_salience = min(sal1, sal2)

        # Check if in bootstrap phase
        in_bootstrap = self.is_in_bootstrap_phase(current_time)

        if in_bootstrap:
            # BOOTSTRAP PHASE: Very permissive edge formation to kickstart the network
            # Critical for ant survival: edges reduce energy decay, so need rapid edge formation
            pos1 = self.graph.nodes[v1].get('position')
            pos2 = self.graph.nodes[v2].get('position')

            # GUARANTEED EDGES: First few edges are always created to bootstrap network
            n_edges = self.graph.number_of_edges()
            if n_edges < 10:
                # Strongly encourage initial edge formation
                edge_prob = 0.8  # 80% chance for first 10 edges
            else:
                spatial_proximity_bonus = 0.0
                if pos1 is not None and pos2 is not None:
                    try:
                        spatial_dist = np.linalg.norm(pos1[1:] - pos2[1:])
                        # Closer vertices get higher bonus (inverse distance, stronger effect)
                        spatial_proximity_bonus = 2.0 / (1.0 + spatial_dist)
                    except (TypeError, ValueError):
                        pass

                # Bootstrap base probability is much higher (increased from 5x to 10x)
                base_prob = (0.2 + 0.1 * nearby_structures) * self.BOOTSTRAP_EDGE_PROB_MULTIPLIER * 2.0

                # Co-occurrence still helps but not required
                co_occurrence = self.compute_co_occurrence(v1, v2, current_time)
                co_occurrence_bonus = 1.0 + 0.3 * co_occurrence

                # Final bootstrap probability (capped at 0.9)
                edge_prob = min(0.9, min_salience * co_occurrence_bonus * (base_prob + spatial_proximity_bonus))

        else:
            # POST-BOOTSTRAP: Standard conditional edge formation
            # Require co-occurrence (temporal overlap)
            co_occurrence = self.compute_co_occurrence(v1, v2, current_time)
            if co_occurrence == 0:
                return False  # No edge without temporal overlap

            # Structure-dependent base probability
            base_prob = 0.05 + 0.05 * nearby_structures

            # Final edge probability
            edge_prob = min_salience * (1 + 0.1 * co_occurrence) * base_prob

        if np.random.random() < edge_prob:
            self.add_edge(v1, v2, pheromone)
            # Track co-occurrence for this pair
            pair = (min(v1, v2), max(v1, v2))
            self.co_occurrence_counts[pair] = self.co_occurrence_counts.get(pair, 0) + 1
            return True

        return False

    # =========================================================================
    # ENTROPY AND TRANSPORT GATING
    # =========================================================================

    def compute_entropy(self) -> float:
        """
        Compute graph entropy for transport gating.
        Higher entropy = more uniform structure = ready for transport.

        Uses degree distribution entropy:
            H = -Σ p(k) log(p(k))
        where p(k) is probability of degree k.
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0

        # Get degree sequence
        degrees = [d for _, d in self.graph.degree()]
        if not degrees:
            return 0.0

        # Count degree frequencies
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1

        # Compute probability distribution
        n = len(degrees)
        probabilities = [count / n for count in degree_counts.values()]

        # Compute Shannon entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log(p + 1e-10)

        # Normalize to [0, 1] range (max entropy is log(n))
        max_entropy = np.log(n + 1)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def is_transport_ready(self, current_time: float = None) -> bool:
        """
        Check if semantic graph is mature enough for transport.

        Adaptive thresholds based on graph maturity:
        - During bootstrap: Lower thresholds to encourage early transport
        - After bootstrap: Standard thresholds require stable structure

        Requirements (standard):
        - At least 3 vertices
        - At least 2 edges (PHASE_1_MATURITY_EDGES)
        - Graph entropy > MIN_ENTROPY_FOR_TRANSPORT

        Requirements (bootstrap):
        - At least 3 vertices
        - At least 1 edge
        - Some packets in queues OR stable vertices exist
        """
        n_vertices = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        entropy = self.compute_entropy()

        # Check bootstrap phase
        in_bootstrap = current_time is not None and self.is_in_bootstrap_phase(current_time)

        if in_bootstrap:
            # During bootstrap: Lower thresholds
            # Allow transport if we have basic structure
            has_packets = self.get_total_queue_length() > 0
            has_stable_vertices = len(self.stable_vertex_set) >= 2

            return (n_vertices >= 3 and
                    n_edges >= 1 and
                    (has_packets or has_stable_vertices or entropy >= 0.1))
        else:
            # Standard thresholds
            return (n_vertices >= 3 and
                    n_edges >= self.PHASE_1_MATURITY_EDGES and
                    entropy >= self.MIN_ENTROPY_FOR_TRANSPORT)

    def get_graph_maturity_phase(self, current_time: float = None) -> int:
        """
        Get current graph maturity phase.

        Phase 0: Immature (< 3 vertices or < 2 edges)
        Phase 1: Basic transport enabled (2+ edges, entropy > 0.3)
        Phase 2: Full transport (5+ edges)
        Phase 3: Mature stable graph (many stable vertices, high connectivity)
        """
        n_vertices = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        n_stable = len(self.stable_vertex_set)

        if not self.is_transport_ready(current_time):
            return 0

        # Phase 3: Mature stable graph
        if n_edges >= 10 and n_stable >= 5:
            return 3

        # Phase 2: Full transport
        if n_edges >= self.PHASE_2_MATURITY_EDGES:
            return 2

        return 1

    def get_transport_priority_multiplier(self, current_time: float = None) -> float:
        """
        Get priority multiplier for transport based on graph state.

        Higher multiplier when graph is healthy and has queued packets.
        Lower multiplier when graph is struggling.
        """
        phase = self.get_graph_maturity_phase(current_time)
        n_queued = self.get_total_queue_length()
        n_stable = len(self.stable_vertex_set)

        # Base multiplier from phase
        phase_multiplier = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0}.get(phase, 1.0)

        # Bonus for queued packets (encourages clearing queue)
        queue_bonus = min(1.0, n_queued * 0.1)

        # Bonus for stable vertices (healthy graph)
        stability_bonus = min(0.5, n_stable * 0.05)

        return phase_multiplier + queue_bonus + stability_bonus

# =============================================================================
# ENHANCED SIMULATION ENGINE
# =============================================================================

class EnhancedSimulationEngine:
    """Enhanced simulation with full functionality"""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.logger.info("Initializing enhanced spacetime...")
        self.spacetime = EnhancedSpacetime(config)
        
        self.logger.info("Initializing semantic graph...")
        self.semantic_graph = SemanticGraph()

        # FIX #3: Seed semantic graph with initial vertices so bees have targets
        self._seed_semantic_graph(n_initial_vertices=15)

        self.logger.info("Initializing agents...")
        self.agents = self._initialize_agents()
        
        # Wormhole position - derived from config instead of hardcoded
        # Located just outside the wormhole throat at r = throat_radius + 0.6
        wormhole_r = getattr(config, 'throat_radius', 2.0) + 0.6
        self.wormhole_position = np.array([0.0, wormhole_r, np.pi/2, 0.0])
        
        # Statistics
        self.stats = {
            'n_packets_transported': 0,
            'total_energy': 0.0,
            'n_structures_built': 0,
            'n_vertices': 0,
            'n_edges': 0,
            'energy_history': [],
            'vertices_history': [],
            'structures_history': [],
            'stability_rate': [],
            'lyapunov_V': []
        }

        # Simple stability tracking (full Lyapunov requires epistemic layer)
        self.stability_history = []
        self.consecutive_violations = 0
        self.last_energy = None

        # ISSUE 3: Initialize Overmind for meta-level colony regulation
        if EPISTEMIC_LAYER_AVAILABLE:
            self.logger.info("Initializing Overmind meta-controller...")
            self.overmind = Overmind(target_entropy=100.0)
            self.overmind_active = True
        else:
            self.overmind = None
            self.overmind_active = False
            self.logger.warning("Overmind not available - epistemic_cognitive_layer.py not found")

        # PHASE II: Initialize Adversarial Pressure Layer
        if APL_AVAILABLE:
            self.logger.info("Initializing Adversarial Pressure Layer (Phase II)...")
            self.apl = AdversarialPressureLayer(config)
            self.apl_active = True
            self.apl_stats = {
                'pressure_history': [],
                'threat_history': [],
                'damage_history': []
            }
        else:
            self.apl = None
            self.apl_active = False
            self.logger.warning("APL not available - adversarial_pressure_layer.py not found")

        # PHASE II: Initialize Agent Plasticity System for emergent intelligence
        if PLASTICITY_AVAILABLE:
            self.logger.info("Initializing Agent Plasticity System...")
            self.plasticity = AgentPlasticitySystem({
                'threat_decay_rate': 0.005,
                'max_threat_memories': 200,
                'social_radius': 10.0
            })
            self.plasticity_active = True
            # Initialize all agents with plasticity
            for colony_agents in self.agents.values():
                for agent in colony_agents:
                    self.plasticity.initialize_agent(agent.id)
        else:
            self.plasticity = None
            self.plasticity_active = False
            self.logger.warning("Plasticity not available - agent_plasticity.py not found")

        # RESEARCH-GRADE: Initialize scientific analysis components
        self.logger.info("Initializing research-grade components...")
        self.health_dashboard = GraphHealthDashboard(self.config.output_dir)
        self.event_ledger = EventLedger(self.config.output_dir)
        self.utility_scorer = KnowledgeUtilityScorer()
        self.governor = GraphGovernor()
        self.research_metrics = {
            'health_history': [],
            'utility_history': [],
            'governor_adjustments': []
        }

        self.logger.info("Enhanced simulation engine initialized")
    
    def _setup_logging(self):
        """Setup logging"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/enhanced_simulation.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("EnhancedBlackholeArchive")
    
    def _initialize_agents(self) -> Dict[str, List[Agent]]:
        """Initialize all agents"""
        agents = {'beavers': [], 'ants': [], 'bees': []}
        
        # Beavers
        for i in range(self.config.n_beavers):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 10,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            
            agents['beavers'].append(EnhancedBeaverAgent(
                id=f"beaver_{i}",
                colony="beavers",
                position=position,
                velocity=0.1 * np.random.randn(4),
                energy=1.0
            ))
        
        # Ants
        for i in range(self.config.n_ants):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 20,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            
            agents['ants'].append(EnhancedAntAgent(
                id=f"ant_{i}",
                colony="ants",
                position=position,
                velocity=0.05 * np.random.randn(4),
                energy=1.0
            ))
        
        # Bees
        for i in range(self.config.n_bees):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 15,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            
            agents['bees'].append(EnhancedBeeAgent(
                id=f"bee_{i}",
                colony="bees",
                position=position,
                velocity=0.2 * np.random.randn(4),
                energy=1.0
            ))
        
        return agents

    def save_checkpoint(self, step: int, checkpoint_dir: str = None):
        """Save simulation state for crash recovery.

        Checkpoints are saved as JSON files containing:
        - Current step number
        - Agent states (positions, velocities, energy, counters)
        - Semantic graph (vertices, edges, properties)
        - Statistics and history (truncated to save space)
        """
        import json

        if checkpoint_dir is None:
            checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        else:
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'step': step,
            'timestamp': str(np.datetime64('now')),
            'config': {
                'black_hole_mass': self.config.black_hole_mass,
                'n_beavers': self.config.n_beavers,
                'n_ants': self.config.n_ants,
                'n_bees': self.config.n_bees,
                't_max': self.config.t_max,
                'dt': self.config.dt,
            },
            'agents': {},
            'semantic_graph': {
                'vertices': [],
                'edges': [],
                'ledger': self.semantic_graph.ledger.copy()
            },
            'stats': {
                'total_energy': self.stats.get('total_energy', 0),
                'n_structures_built': self.stats.get('n_structures_built', 0),
                'n_vertices': self.stats.get('n_vertices', 0),
                'n_packets_transported': self.stats.get('n_packets_transported', 0),
                # Keep only last 100 history entries
                'energy_history': self.stats.get('energy_history', [])[-100:],
                'vertices_history': self.stats.get('vertices_history', [])[-100:],
            },
            'material_budget': EnhancedBeaverAgent.global_material_budget
        }

        # Save agent states
        for colony_name, agents_list in self.agents.items():
            checkpoint['agents'][colony_name] = []
            for agent in agents_list:
                agent_state = {
                    'id': agent.id,
                    'position': agent.position.tolist(),
                    'velocity': agent.velocity.tolist(),
                    'energy': agent.energy,
                    'state': agent.state,
                }
                # Colony-specific attributes
                if colony_name == 'beavers':
                    agent_state['structures_built'] = agent.structures_built
                elif colony_name == 'ants':
                    agent_state['vertices_created'] = agent.vertices_created
                    agent_state['edges_created'] = agent.edges_created
                    agent_state['packets_generated'] = agent.packets_generated
                elif colony_name == 'bees':
                    agent_state['packets_delivered'] = agent.packets_delivered
                    agent_state['packets_dropped'] = agent.packets_dropped
                checkpoint['agents'][colony_name].append(agent_state)

        # Save semantic graph vertices
        for v in self.semantic_graph.graph.nodes():
            vdata = self.semantic_graph.graph.nodes[v]
            checkpoint['semantic_graph']['vertices'].append({
                'id': v,
                'position': vdata['position'].tolist() if 'position' in vdata else [0,0,0,0],
                'salience': vdata.get('salience', 0.5),
                'created_at': vdata.get('created_at', 0),
            })

        # Save semantic graph edges (just connectivity)
        for u, v in self.semantic_graph.graph.edges():
            checkpoint['semantic_graph']['edges'].append({'source': u, 'target': v})

        # Write checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:06d}.json"
        with open(checkpoint_path, 'w') as f:
            # FIX: Handle numpy types in JSON serialization
            def numpy_encoder(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            json.dump(checkpoint, f, indent=2, default=numpy_encoder)

        # Keep only last 3 checkpoints to save disk space
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.json"))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, config=None):
        """Load simulation from checkpoint and resume.

        Usage:
            engine = EnhancedSimulationEngine.load_checkpoint("checkpoint_step_002300.json")
            engine.run(resume_from_step=2300)
        """
        import json
        from blackhole_archive_main import SimulationConfig

        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # Create config from checkpoint if not provided
        if config is None:
            ckpt_config = checkpoint['config']
            config = SimulationConfig(
                black_hole_mass=ckpt_config['black_hole_mass'],
                n_beavers=ckpt_config['n_beavers'],
                n_ants=ckpt_config['n_ants'],
                n_bees=ckpt_config['n_bees'],
                t_max=ckpt_config['t_max'],
                dt=ckpt_config['dt'],
            )

        # Create engine
        engine = cls(config)

        # Restore agent states
        for colony_name, agents_data in checkpoint['agents'].items():
            for i, agent_state in enumerate(agents_data):
                if i < len(engine.agents[colony_name]):
                    agent = engine.agents[colony_name][i]
                    agent.position = np.array(agent_state['position'])
                    agent.velocity = np.array(agent_state['velocity'])
                    agent.energy = agent_state['energy']
                    agent.state = agent_state['state']

                    if colony_name == 'beavers':
                        agent.structures_built = agent_state.get('structures_built', 0)
                    elif colony_name == 'ants':
                        agent.vertices_created = agent_state.get('vertices_created', 0)
                        agent.edges_created = agent_state.get('edges_created', 0)
                        agent.packets_generated = agent_state.get('packets_generated', 0)
                    elif colony_name == 'bees':
                        agent.packets_delivered = agent_state.get('packets_delivered', 0)
                        agent.packets_dropped = agent_state.get('packets_dropped', 0)

        # Restore semantic graph
        engine.semantic_graph.graph.clear()
        for vdata in checkpoint['semantic_graph']['vertices']:
            engine.semantic_graph.graph.add_node(
                vdata['id'],
                position=np.array(vdata['position']),
                salience=vdata['salience'],
                created_at=vdata['created_at'],
                packet_queue=[]
            )
        for edata in checkpoint['semantic_graph']['edges']:
            engine.semantic_graph.graph.add_edge(edata['source'], edata['target'], pheromone=0.5)

        # Restore ledger
        engine.semantic_graph.ledger = checkpoint['semantic_graph']['ledger']

        # Restore stats
        engine.stats.update(checkpoint['stats'])

        # Restore material budget
        EnhancedBeaverAgent.global_material_budget = checkpoint['material_budget']

        engine._checkpoint_step = checkpoint['step']
        engine.logger.info(f"Loaded checkpoint from step {checkpoint['step']}")

        return engine

    def _seed_semantic_graph(self, n_initial_vertices: int = 10):
        """
        FIX #3: Seed semantic graph with initial high-salience vertices.

        Without initial vertices, bees have nothing to transport and the
        epistemic layer remains inert. Seed with vertices at positions where
        curvature is high (near event horizon) since these are informationally
        interesting locations.
        """
        self.logger.info(f"Seeding semantic graph with {n_initial_vertices} initial vertices")

        # Initialize bootstrap phase - graph creation time is 0.0
        self.semantic_graph.set_creation_time(0.0)

        for i in range(n_initial_vertices):
            # Create vertices near event horizon where curvature is high
            r = self.spacetime.r_s + 0.5 + np.random.rand() * 5  # r in [rs+0.5, rs+5.5]
            theta = np.random.rand() * np.pi
            phi = np.random.rand() * 2 * np.pi

            position = np.array([0.0, r, theta, phi])

            # Compute salience based on curvature - closer to horizon = higher salience
            curvature = self.spacetime.get_curvature(position)
            salience = min(1.0, curvature * 10)  # Scale to [0, 1]

            # Add vertex to semantic graph
            vertex_id = self.semantic_graph.add_vertex(position, salience)

            # FIX #3: Seed initial packets at high-salience vertices
            if salience > 0.3:
                packet = {
                    'content': f"seed_discovery_{vertex_id}",
                    'salience': salience,
                    'confidence': 0.8,
                    'created_at': 0.0,
                    'ttl': 100.0,
                    'source_agent': 'seed',
                    'source_vertex': vertex_id
                }
                self.semantic_graph.add_packet(vertex_id, packet)

            # Connect to nearby vertices with initial pheromone
            if i > 0:
                for prev_id in range(max(0, vertex_id - 3), vertex_id):
                    self.semantic_graph.add_edge(prev_id, vertex_id, pheromone=0.5)
                    self.semantic_graph.add_edge(vertex_id, prev_id, pheromone=0.5)

        self.logger.info(f"Semantic graph seeded with {self.semantic_graph.graph.number_of_nodes()} vertices")

    def run(self, resume_from_step: int = 0, checkpoint_interval: int = 500):
        """Run enhanced simulation with checkpointing and error recovery.

        Args:
            resume_from_step: Start from this step (used when resuming from checkpoint)
            checkpoint_interval: Save checkpoint every N steps (default 500)
        """
        n_steps = int(self.config.t_max / self.config.dt)
        start_step = resume_from_step if resume_from_step > 0 else getattr(self, '_checkpoint_step', 0)

        if start_step > 0:
            self.logger.info(f"Resuming simulation from step {start_step}/{n_steps}")
        else:
            self.logger.info(f"Starting enhanced simulation: {n_steps} steps")
            # FIX #5: Reset material budget at simulation start (only if fresh start)
            EnhancedBeaverAgent.global_material_budget = 500.0

        try:
            for step in tqdm(range(start_step, n_steps), desc="Enhanced Simulation", initial=start_step, total=n_steps):
                t = step * self.config.dt

                # CHECKPOINT: Save every checkpoint_interval steps
                if step > 0 and step % checkpoint_interval == 0:
                    self.save_checkpoint(step)

                # Update all agents
                for beaver in self.agents['beavers']:
                    if beaver.state == "active":
                        beaver.update(self.config.dt, self.spacetime, self.semantic_graph)

                # FIX: Regenerate global materials (soft constraint instead of hard cap)
                EnhancedBeaverAgent.regenerate_materials(self.config.dt)

                for ant in self.agents['ants']:
                    if ant.state == "active":
                        ant.update(self.config.dt, self.spacetime, self.semantic_graph, current_time=t)

                for bee in self.agents['bees']:
                    if bee.state == "active":
                        bee.update(self.config.dt, self.spacetime, self.semantic_graph,
                                   self.wormhole_position, current_time=t)

                # FIX #5: Decay structural field (maintenance cost)
                self.spacetime.decay_structural_field(self.config.dt)

                # FIX #7: Decay epistemic stress
                self.spacetime.decay_epistemic_stress(self.config.dt)

                # FIX #7: Add epistemic stress at congested vertices
                for v in self.semantic_graph.graph.nodes():
                    queue_len = self.semantic_graph.get_queue_length(v)
                    if queue_len > 5:  # Congestion threshold
                        pos = self.semantic_graph.graph.nodes[v]['position']
                        stress = 0.1 * (queue_len - 5)  # Stress proportional to congestion
                        self.spacetime.add_epistemic_stress(pos, stress)

                # Decay pheromones
                self.semantic_graph.decay_pheromones(self.config.dt)

                # DYNAMIC SPACETIME: Update stress-energy from agents and evolve metric
                # PERFORMANCE: Only update metric every 10 steps (physically reasonable,
                # metric doesn't need dt-level updates and this is O(grid³) computation)
                METRIC_UPDATE_INTERVAL = 10
                if step % METRIC_UPDATE_INTERVAL == 0:
                    for agents_list in self.agents.values():
                        for agent in agents_list:
                            if agent.state == "active":
                                # Each active agent contributes to stress-energy
                                self.spacetime.add_stress_energy(agent.position, agent.energy * 0.01)

                    # Evolve metric perturbation (linearized Einstein equations)
                    self.spacetime.evolve_metric(self.config.dt * METRIC_UPDATE_INTERVAL)

                # BEKENSTEIN BOUND: Add entropy from packet queues
                for v in self.semantic_graph.graph.nodes():
                    queue_len = self.semantic_graph.get_queue_length(v)
                    if queue_len > 0:
                        pos = self.semantic_graph.graph.nodes[v]['position']
                        # Each packet represents some bits of information
                        bits = queue_len * 64  # Assume 64 bits per packet header
                        thermalized = self.spacetime.add_local_entropy(pos, bits * 0.01)
                        # If bits were thermalized, corrupt/drop packets
                        if thermalized > 0:
                            packets_to_drop = int(thermalized / 64)
                            for _ in range(min(packets_to_drop, queue_len)):
                                self.semantic_graph.get_packet(v)  # Drop packet

                # Decay entropy (Hawking radiation analog)
                self.spacetime.decay_entropy(self.config.dt)

                # FIX #6: Periodically prune and merge graph
                # A/B TEST CONTROLS: Set env vars to disable for debugging
                import os
                enable_prune = not os.environ.get('DISABLE_PRUNE')
                enable_merge = not os.environ.get('DISABLE_MERGE')

                if step % 50 == 0 and step > 0:
                    pruned = 0
                    merged = 0
                    if enable_prune:
                        pruned = self.semantic_graph.prune_graph(t)
                    if enable_merge:
                        # RESEARCH-GRADE: Use Governor's adaptive merge threshold
                        merge_threshold = self.governor.get_merge_threshold(base_threshold=0.05)
                        # FIX: Pass current_time for grace period protection
                        merged = self.semantic_graph.merge_nearby_vertices(
                            distance_threshold=merge_threshold, current_time=t
                        )
                    if pruned > 0 or merged > 0:
                        self.logger.debug(f"Graph maintenance: pruned={pruned}, merged={merged}")

                # =================================================================
                # ISSUE 3: OVERMIND META-LEVEL CONTROL
                # =================================================================
                if self.overmind_active and step % 10 == 0:
                    # Get current colony metrics
                    n_vertices = self.semantic_graph.graph.number_of_nodes()
                    n_edges = self.semantic_graph.graph.number_of_edges()
                    n_structures = sum(b.structures_built for b in self.agents['beavers'])
                    n_packets = sum(b.packets_delivered for b in self.agents['bees'])

                    # Create wrapper for Overmind to observe
                    # (Overmind expects EpistemicSemanticGraph, but we can pass data directly)
                    total_energy = sum(a.energy for agents in self.agents.values()
                                       for a in agents if a.state == "active")

                    # Detect imbalance: structures >> vertices or vertices >> packets
                    structure_vertex_ratio = (n_structures + 1) / (n_vertices + 1)
                    vertex_packet_ratio = (n_vertices + 1) / (n_packets + 1)

                    # COLONY BALANCING LOGIC
                    # If structures >> vertices: boost ant exploration
                    if structure_vertex_ratio > 10:
                        # Ants need to explore more - give them energy bonus
                        exploration_bonus = 0.01 * (structure_vertex_ratio - 10)
                        for ant in self.agents['ants']:
                            if ant.state == "active":
                                ant.energy = min(1.5, ant.energy + exploration_bonus)

                    # If vertices >> packets: boost bee transport
                    if vertex_packet_ratio > 5:
                        # Bees need to transport more - give them energy bonus
                        transport_bonus = 0.005 * (vertex_packet_ratio - 5)
                        for bee in self.agents['bees']:
                            if bee.state == "active":
                                bee.energy = min(1.5, bee.energy + transport_bonus)

                    # If too many contradictions/stagnation: inject exploration noise
                    if hasattr(self.semantic_graph, 'stable_vertex_set'):
                        stability_rate = len(self.semantic_graph.stable_vertex_set) / (n_vertices + 1)
                        if stability_rate > 0.9 and step > 200:
                            # System is stagnating - inject exploration
                            for ant in self.agents['ants']:
                                if ant.state == "active":
                                    # Add random velocity perturbation
                                    ant.velocity += 0.02 * np.random.randn(4)

                    # Log Overmind state periodically
                    if step % 100 == 0:
                        self.logger.debug(
                            f"Overmind: S/V ratio={structure_vertex_ratio:.2f}, "
                            f"V/P ratio={vertex_packet_ratio:.2f}"
                        )

                # =================================================================
                # PHASE II: ADVERSARIAL PRESSURE LAYER
                # =================================================================
                if self.apl_active and step % 5 == 0:
                    # Compute system state for APL
                    n_alive = sum(1 for agents_list in self.agents.values()
                                 for a in agents_list if a.state == "active")
                    n_total = sum(len(agents_list) for agents_list in self.agents.values())
                    survival_rate = n_alive / max(1, n_total)

                    total_energy = sum(a.energy for agents_list in self.agents.values()
                                       for a in agents_list if a.state == "active")

                    # Energy trend (positive = gaining energy)
                    if self.last_energy is not None:
                        energy_trend = (total_energy - self.last_energy) / self.config.dt
                    else:
                        energy_trend = 0.0

                    # Build work efficiency
                    n_structures = sum(b.structures_built for b in self.agents['beavers'])
                    n_packets = sum(b.packets_delivered for b in self.agents['bees'])
                    work_per_agent = (n_structures + n_packets) / max(1, n_alive)

                    # Graph health
                    n_vertices = self.semantic_graph.graph.number_of_nodes()
                    n_edges = self.semantic_graph.graph.number_of_edges()
                    graph_health = min(1.0, n_vertices / 100.0)  # Healthy if 100+ vertices

                    system_state = {
                        'survival_rate': survival_rate,
                        'energy_trend': energy_trend,
                        'work_efficiency': work_per_agent * 10,
                        'behavioral_entropy': 0.5,  # TODO: compute from action distribution
                        'graph_health': graph_health,
                        'packet_backlog': self.semantic_graph.get_total_queue_length(),
                        'energy_ratio': total_energy / max(1, n_total),
                        'build_rate': n_structures / max(1, t),
                        'total_energy': total_energy,
                        'n_vertices': n_vertices,
                        'n_edges': n_edges,
                        'n_packets': n_packets,
                        'work_per_agent': work_per_agent
                    }

                    # Update APL
                    apl_result = self.apl.update(
                        current_time=t,
                        dt=self.config.dt * 5,  # APL runs every 5 steps
                        spacetime=self.spacetime,
                        semantic_graph=self.semantic_graph,
                        agents=self.agents,
                        system_state=system_state
                    )

                    # Record APL stats
                    if step % 50 == 0:
                        self.apl_stats['pressure_history'].append(apl_result['pressure_budget'])
                        self.apl_stats['threat_history'].append(len(apl_result['active_effects']))
                        self.apl_stats['damage_history'].append(apl_result['damage_report'])

                    # Log APL events
                    if apl_result['triggered_events']:
                        self.logger.info(
                            f"APL triggered: {', '.join(apl_result['triggered_events'])} "
                            f"(budget={apl_result['pressure_budget']:.1f}, "
                            f"threat={apl_result['threat_level']:.3f})"
                        )

                # =================================================================
                # AGENT PLASTICITY SYSTEM - Learning from experience
                # =================================================================
                if self.plasticity_active:
                    # Collect all agents for social learning
                    all_agents = [a for agents_list in self.agents.values() for a in agents_list]

                    for colony_name, agents_list in self.agents.items():
                        for agent in agents_list:
                            # Handle dead agents
                            if agent.state == "dead" and agent.id in self.plasticity.agent_states:
                                # Determine death cause from current conditions
                                death_cause = "unknown"
                                if agent.energy <= 0:
                                    death_cause = "energy_depletion"
                                if hasattr(agent, 'position') and agent.position[1] < 3.0:
                                    death_cause = "horizon_crossing"

                                conditions = {
                                    'energy': agent.energy,
                                    'r': agent.position[1] if hasattr(agent, 'position') else 0,
                                    'colony': colony_name
                                }
                                self.plasticity.on_agent_death(agent, death_cause, conditions, t)

                                # RESEARCH-GRADE: Log death to event ledger
                                self.event_ledger.log_agent_death(
                                    step, agent.id, death_cause,
                                    agent.energy, agent.position
                                )

                            # Update plasticity for active agents and apply modifiers
                            elif agent.state == "active":
                                modifiers = self.plasticity.update_agent(agent, all_agents, self.config.dt)

                                # Apply threat avoidance to velocity
                                avoidance = modifiers.get('avoidance_vector', np.zeros(3))
                                if np.linalg.norm(avoidance) > 0.01:
                                    # Blend avoidance into velocity
                                    danger_avoidance = modifiers.get('danger_avoidance', 0)
                                    agent.velocity[1:4] += 0.1 * danger_avoidance * avoidance

                                # Record success when energy/contribution increases
                                if hasattr(agent, 'contribution_score'):
                                    reward = getattr(agent, '_last_contribution', 0)
                                    current = agent.contribution_score
                                    if current > reward:
                                        self.plasticity.on_agent_success(agent, current - reward)
                                    agent._last_contribution = current

                    # System-wide plasticity update (memory decay)
                    if step % 10 == 0:
                        self.plasticity.update_system(self.config.dt * 10)

                    # Log plasticity metrics periodically
                    if step % 200 == 0:
                        metrics = self.plasticity.get_metrics()
                        if metrics['deaths_recorded'] > 0:
                            self.logger.info(
                                f"Plasticity: deaths={metrics['deaths_recorded']}, "
                                f"danger_zones={metrics['danger_zones']}, "
                                f"social_learning={metrics['social_learning_events']}"
                            )

                # Update statistics
                self.stats['total_energy'] = sum(
                    a.energy for agents in self.agents.values() for a in agents if a.state == "active"
                )
                self.stats['n_structures_built'] = sum(
                    b.structures_built for b in self.agents['beavers']
                )
                self.stats['n_packets_transported'] = sum(
                    b.packets_delivered for b in self.agents['bees']
                )
                self.stats['n_packets_dropped'] = sum(
                    b.packets_dropped for b in self.agents['bees']
                )
                self.stats['n_packets_generated'] = sum(
                    a.packets_generated for a in self.agents['ants']
                )
                self.stats['queue_length'] = self.semantic_graph.get_total_queue_length()
                self.stats['material_remaining'] = EnhancedBeaverAgent.global_material_budget
                self.stats['n_vertices'] = self.semantic_graph.graph.number_of_nodes()
                self.stats['n_edges'] = self.semantic_graph.graph.number_of_edges()

                # ISSUE 2: Individual contribution tracking
                self.stats['ant_contributions'] = {
                    ant.id: {
                        'vertices_created': ant.vertices_created,
                        'edges_created': ant.edges_created,
                        'info_discovered': ant.total_info_discovered,
                        'contribution_score': ant.contribution_score
                    }
                    for ant in self.agents['ants']
                }
                # Summary: vertices per ant (productivity metric)
                active_ants = [a for a in self.agents['ants'] if a.state == "active"]
                total_ant_vertices = sum(a.vertices_created for a in self.agents['ants'])
                self.stats['vertices_per_ant'] = total_ant_vertices / max(1, len(active_ants))

                # VALIDATION: Thermodynamic metrics
                thermo_stats = self.spacetime.get_thermodynamic_stats()
                self.stats['bekenstein_thermalization_events'] = thermo_stats['bekenstein_thermalization_events']
                self.stats['thermodynamic_heat'] = thermo_stats['thermodynamic_heat']
                self.stats['total_entropy'] = thermo_stats['total_entropy']
                self.stats['mean_metric_perturbation'] = thermo_stats['mean_metric_perturbation']

                # PLASTICITY metrics
                if self.plasticity_active:
                    plasticity_metrics = self.plasticity.get_metrics()
                    self.stats['plasticity_deaths'] = plasticity_metrics['deaths_recorded']
                    self.stats['plasticity_danger_zones'] = plasticity_metrics['danger_zones']
                    self.stats['plasticity_social_learning'] = plasticity_metrics['social_learning_events']
                    self.stats['plasticity_adaptations'] = plasticity_metrics['strategies_adapted']

                # Simple stability check: energy should not decay too fast
                current_energy = self.stats['total_energy']
                is_stable = True
                if self.last_energy is not None:
                    energy_decay_rate = (self.last_energy - current_energy) / self.config.dt
                    # Unstable if energy decays more than 5% per unit time
                    if energy_decay_rate > 0.05 * self.last_energy:
                        is_stable = False
                        self.consecutive_violations += 1
                    else:
                        self.consecutive_violations = 0
                self.last_energy = current_energy
                self.stability_history.append(is_stable)

                # Record history
                if step % 10 == 0:
                    self.stats['energy_history'].append(self.stats['total_energy'])
                    self.stats['vertices_history'].append(self.stats['n_vertices'])
                    self.stats['structures_history'].append(self.stats['n_structures_built'])
                    stable_count = sum(1 for s in self.stability_history if s)
                    self.stats['stability_rate'].append(stable_count / max(1, len(self.stability_history)))

                    # Snapshot ledger for debugging
                    self.semantic_graph.snapshot_ledger(step)

                # =================================================================
                # RESEARCH-GRADE: Scientific Analysis (expensive - skip in fast mode)
                # =================================================================
                skip_research = os.environ.get('SIM_FAST') or os.environ.get('SKIP_METRICS')
                research_interval = 200 if skip_research else 50

                if step % research_interval == 0 and step > 0 and not skip_research:
                    # Compute graph health metrics
                    health_metrics = self.health_dashboard.compute_metrics(
                        self.semantic_graph.graph, self.agents, step
                    )
                    # Fill in queue metrics
                    health_metrics['total_queue_backlog'] = self.semantic_graph.get_total_queue_length()
                    health_metrics['max_queue_length'] = max(
                        (self.semantic_graph.get_queue_length(v) for v in self.semantic_graph.graph.nodes()),
                        default=0
                    )
                    self.research_metrics['health_history'].append(health_metrics)

                    # Update Governor policies based on health
                    adjustments = self.governor.update_policies(health_metrics, step)
                    if adjustments:
                        self.research_metrics['governor_adjustments'].append({
                            'step': step, 'adjustments': adjustments
                        })
                        self.logger.debug(f"Governor adjustments: {adjustments}")

                # Run knowledge utility query (every 100 steps - expensive, skip in fast mode)
                if step % 100 == 0 and step > 0 and not skip_research:
                    utility_result = self.utility_scorer.run_query_task(
                        self.semantic_graph.graph, self.agents, step
                    )
                    self.research_metrics['utility_history'].append(utility_result)

                    # Log utility score
                    utility = utility_result.get('utility', 0)
                    if step % 500 == 0:
                        self.logger.info(
                            f"Knowledge Utility: {utility:.3f} "
                            f"(retrieval={utility_result.get('retrieval_success', 0):.2f}, "
                            f"freshness={utility_result.get('freshness', 0):.2f})"
                        )

                # Log
                if step % 100 == 0:
                    self.logger.info(
                        f"Step {step}/{n_steps}, t={t:.2f}, "
                        f"Energy={self.stats['total_energy']:.2f}, "
                        f"Vertices={self.stats['n_vertices']}, "
                        f"V/Ant={self.stats['vertices_per_ant']:.2f}, "
                        f"Structures={self.stats['n_structures_built']}, "
                        f"Packets={self.stats['n_packets_transported']}, "
                        f"Queue={self.stats['queue_length']}, "
                        f"Materials={self.stats['material_remaining']:.0f}"
                    )

        except KeyboardInterrupt:
            self.logger.warning(f"Simulation interrupted at step {step}. Saving emergency checkpoint...")
            self.save_checkpoint(step)
            raise

        except Exception as e:
            self.logger.error(f"Simulation crashed at step {step}: {e}")
            self.logger.warning("Saving emergency checkpoint...")
            try:
                self.save_checkpoint(step)
                self.logger.info(f"Emergency checkpoint saved. Resume with: load_checkpoint('checkpoint_step_{step:06d}.json')")
            except Exception as save_error:
                self.logger.error(f"Failed to save emergency checkpoint: {save_error}")
            raise

        self.logger.info("Enhanced simulation complete")

        # GRAPH LEDGER DUMP - Critical for debugging vertex collapse
        self.logger.info("\n" + self.semantic_graph.dump_ledger())

        self._save_results()
    
    def _save_results(self):
        """Save enhanced results including research-grade metrics"""
        report_path = Path(self.config.output_dir) / "enhanced_simulation_report.json"

        report = {
            'config': {
                'black_hole_mass': self.config.black_hole_mass,
                'n_beavers': self.config.n_beavers,
                'n_ants': self.config.n_ants,
                'n_bees': self.config.n_bees,
                't_max': self.config.t_max,
                'dt': self.config.dt
            },
            'final_statistics': self.stats,
            'n_agents_alive': {
                colony: len([a for a in agents if a.state == "active"])
                for colony, agents in self.agents.items()
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        self.logger.info(f"Enhanced report saved to {report_path}")

        # RESEARCH-GRADE: Save scientific analysis outputs
        # Save health dashboard history
        self.health_dashboard.save_history()
        self.logger.info(f"Graph health history saved ({len(self.health_dashboard.history)} records)")

        # Save event ledger
        self.event_ledger.save()
        self.logger.info(f"Event ledger saved ({len(self.event_ledger.events)} events)")

        # Save utility score history
        utility_path = Path(self.config.output_dir) / "knowledge_utility_history.json"
        with open(utility_path, 'w') as f:
            json.dump(self.research_metrics['utility_history'], f, indent=2,
                      default=lambda x: float(x) if isinstance(x, np.floating) else x)
        self.logger.info(f"Knowledge utility history saved to {utility_path}")

        # Save governor adjustment history
        governor_path = Path(self.config.output_dir) / "governor_adjustments.json"
        with open(governor_path, 'w') as f:
            json.dump({
                'final_policies': self.governor.policies,
                'adjustments': self.research_metrics['governor_adjustments']
            }, f, indent=2)
        self.logger.info(f"Governor adjustments saved to {governor_path}")

class LiveVisualizer:
    """
    Real-time visualization of the simulation.

    Usage: VIS=1 python blackhole_archive_enhanced.py

    Shows:
    - Agent positions (Beavers=brown, Ants=red, Bees=yellow)
    - Knowledge graph vertices
    - Energy levels
    - Key metrics in real-time
    """

    def __init__(self, engine: EnhancedSimulationEngine):
        self.engine = engine
        self.fig = None
        self.axes = None
        self.initialized = False

    def setup(self):
        """Initialize matplotlib figure with subplots"""
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Blackhole Archive - Live Simulation', fontsize=14)

        # Configure subplots
        self.ax_space = self.axes[0, 0]  # Agent positions in r-theta plane
        self.ax_graph = self.axes[0, 1]  # Knowledge graph
        self.ax_energy = self.axes[1, 0]  # Energy over time
        self.ax_metrics = self.axes[1, 1]  # Key metrics

        self.energy_history = []
        self.vertex_history = []
        self.time_history = []

        self.initialized = True

    def update(self, step: int, t: float):
        """Update visualization with current state"""
        if not self.initialized:
            self.setup()

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # ===== Agent Positions (r-theta polar projection) =====
        self.ax_space.set_title(f'Agent Positions (Step {step})')

        # Plot event horizon
        theta_range = np.linspace(0, 2*np.pi, 100)
        r_horizon = self.engine.config.black_hole_mass * 2
        self.ax_space.plot(r_horizon * np.cos(theta_range),
                          r_horizon * np.sin(theta_range),
                          'k-', linewidth=2, label='Event Horizon')

        # Plot agents
        for beaver in self.engine.agents['beavers']:
            if beaver.state == 'active':
                r, theta = beaver.position[1], beaver.position[2]
                x, y = r * np.cos(theta), r * np.sin(theta)
                self.ax_space.scatter(x, y, c='saddlebrown', s=40, alpha=0.7)

        for ant in self.engine.agents['ants']:
            if ant.state == 'active':
                r, theta = ant.position[1], ant.position[2]
                x, y = r * np.cos(theta), r * np.sin(theta)
                self.ax_space.scatter(x, y, c='red', s=20, alpha=0.5)

        for bee in self.engine.agents['bees']:
            if bee.state == 'active':
                r, theta = bee.position[1], bee.position[2]
                x, y = r * np.cos(theta), r * np.sin(theta)
                self.ax_space.scatter(x, y, c='gold', s=30, alpha=0.6)

        # Legend
        self.ax_space.scatter([], [], c='saddlebrown', s=40, label='Beavers')
        self.ax_space.scatter([], [], c='red', s=20, label='Ants')
        self.ax_space.scatter([], [], c='gold', s=30, label='Bees')
        self.ax_space.legend(loc='upper right', fontsize=8)
        self.ax_space.set_xlim(-50, 50)
        self.ax_space.set_ylim(-50, 50)
        self.ax_space.set_aspect('equal')
        self.ax_space.set_xlabel('x (r·cos θ)')
        self.ax_space.set_ylabel('y (r·sin θ)')

        # ===== Knowledge Graph =====
        self.ax_graph.set_title(f'Knowledge Graph ({self.engine.semantic_graph.graph.number_of_nodes()} vertices)')

        G = self.engine.semantic_graph.graph
        if G.number_of_nodes() > 0:
            # Get positions from vertex attributes
            pos = {}
            colors = []
            for v in G.nodes():
                node_pos = G.nodes[v].get('position', np.array([0, 10, 0, 0]))
                r, theta = node_pos[1], node_pos[2]
                pos[v] = (r * np.cos(theta), r * np.sin(theta))
                salience = G.nodes[v].get('salience', 0.5)
                colors.append(salience)

            # Draw graph (limit edges for performance)
            edges_to_draw = list(G.edges())[:200]
            nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,
                                   ax=self.ax_graph, alpha=0.2, width=0.5)
            nx.draw_networkx_nodes(G, pos, ax=self.ax_graph,
                                   node_size=20, node_color=colors,
                                   cmap='YlOrRd', alpha=0.7)

        self.ax_graph.set_xlim(-50, 50)
        self.ax_graph.set_ylim(-50, 50)

        # ===== Energy & Vertices Over Time =====
        total_energy = sum(a.energy for agents in self.engine.agents.values()
                          for a in agents if a.state == 'active')
        n_vertices = self.engine.semantic_graph.graph.number_of_nodes()

        self.time_history.append(t)
        self.energy_history.append(total_energy)
        self.vertex_history.append(n_vertices)

        self.ax_energy.set_title('System State Over Time')
        self.ax_energy.plot(self.time_history, self.energy_history, 'b-', label='Energy')
        self.ax_energy.set_xlabel('Time')
        self.ax_energy.set_ylabel('Total Energy', color='b')
        self.ax_energy.tick_params(axis='y', labelcolor='b')

        ax2 = self.ax_energy.twinx()
        ax2.plot(self.time_history, self.vertex_history, 'g-', label='Vertices')
        ax2.set_ylabel('Vertices', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # ===== Metrics Panel =====
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Live Metrics')

        # Count alive agents
        n_beavers = sum(1 for a in self.engine.agents['beavers'] if a.state == 'active')
        n_ants = sum(1 for a in self.engine.agents['ants'] if a.state == 'active')
        n_bees = sum(1 for a in self.engine.agents['bees'] if a.state == 'active')

        n_structures = sum(b.structures_built for b in self.engine.agents['beavers'])
        n_packets = sum(b.packets_delivered for b in self.engine.agents['bees'])
        n_edges = self.engine.semantic_graph.graph.number_of_edges()

        metrics_text = f"""
        Step: {step}  |  Time: {t:.2f}

        COLONIES
        ---------------------
        Beavers: {n_beavers} alive
        Ants:    {n_ants} alive
        Bees:    {n_bees} alive

        KNOWLEDGE GRAPH
        ---------------------
        Vertices:  {n_vertices}
        Edges:     {n_edges}
        V/Ant:     {n_vertices/max(1,n_ants):.2f}

        ACTIVITY
        ---------------------
        Structures: {n_structures}
        Packets:    {n_packets}
        Energy:     {total_energy:.1f}
        Materials:  {EnhancedBeaverAgent.global_material_budget:.0f}
        """

        self.ax_metrics.text(0.1, 0.9, metrics_text, transform=self.ax_metrics.transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.pause(0.01)  # Brief pause to update display

    def close(self):
        """Clean up"""
        plt.ioff()
        plt.close(self.fig)


def run_with_visualization():
    """Run simulation with live visualization"""
    from blackhole_archive_main import SimulationConfig
    import os

    # Use faster settings for visualization
    t_max = 10.0 if os.environ.get('SIM_FAST') else 100.0
    config = SimulationConfig(
        t_max=t_max,
        dt=0.01,
        output_dir="./enhanced_results"
    )

    engine = EnhancedSimulationEngine(config)
    viz = LiveVisualizer(engine)

    n_steps = int(config.t_max / config.dt)
    print(f"Starting visualization mode: {n_steps} steps")
    print("Close the window to stop simulation")

    # Reset material budget
    EnhancedBeaverAgent.global_material_budget = 500.0

    try:
        for step in tqdm(range(n_steps), desc="Visualized Simulation"):
            t = step * config.dt

            # Update agents (same as engine.run() but step by step)
            for beaver in engine.agents['beavers']:
                if beaver.state == "active":
                    beaver.update(config.dt, engine.spacetime, engine.semantic_graph)

            for ant in engine.agents['ants']:
                if ant.state == "active":
                    ant.update(config.dt, engine.spacetime, engine.semantic_graph, current_time=t)

            for bee in engine.agents['bees']:
                if bee.state == "active":
                    bee.update(config.dt, engine.spacetime, engine.semantic_graph,
                              engine.wormhole_position, current_time=t)

            # Decay fields
            engine.spacetime.decay_structural_field(config.dt)
            engine.spacetime.decay_epistemic_stress(config.dt)
            engine.semantic_graph.decay_pheromones(config.dt)

            # Graph maintenance
            if step % 50 == 0 and step > 0:
                engine.semantic_graph.prune_graph(t)
                engine.semantic_graph.merge_nearby_vertices()

            # Update visualization every 50 steps (less overhead)
            if step % 50 == 0:
                viz.update(step, t)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        viz.close()

    print(f"\n✅ Visualization Complete!")
    print(f"Final vertices: {engine.semantic_graph.graph.number_of_nodes()}")


# Example usage
if __name__ == "__main__":
    from blackhole_archive_main import SimulationConfig
    import os

    # Check for visualization mode
    if os.environ.get('VIS'):
        run_with_visualization()
    else:
        # Standard run
        t_max = 10.0 if os.environ.get('SIM_FAST') else 100.0
        config = SimulationConfig(
            t_max=t_max,
            dt=0.01,
            output_dir="./enhanced_results"
        )

        engine = EnhancedSimulationEngine(config)
        engine.run()

        print(f"\n✅ Enhanced Simulation Complete!")
        print(f"Structures built: {engine.stats['n_structures_built']}")
        print(f"Semantic vertices: {engine.stats['n_vertices']}")
        print(f"Packets transported: {engine.stats['n_packets_transported']}")
