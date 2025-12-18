"""
MycoBeaver Semantic Construction System
=========================================
Ant-inspired knowledge graph with semantic entropy and coherence maintenance.

Based on Distributed Cognitive Architecture in Adversarial Information Environments:
- Knowledge graph with pheromone-weighted edges
- Semantic entropy for measuring information coherence
- Ant-based traversal for pattern discovery
- Contradiction detection and resolution
- Information thermodynamics (free energy principle)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict

from .config import SemanticConfig, InfoCostConfig

# Forward reference for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .environment import AgentState


class VertexType(Enum):
    """Types of vertices in the knowledge graph"""
    OBSERVATION = "observation"  # Sensory data
    BELIEF = "belief"  # Inferred beliefs
    GOAL = "goal"  # Agent goals
    ACTION = "action"  # Action nodes
    ENTITY = "entity"  # World entities (dams, lodges, agents)
    LOCATION = "location"  # Spatial locations
    STATE = "state"  # Environmental states
    TEMPORAL = "temporal"  # Time-related nodes


class EdgeType(Enum):
    """Types of edges in the knowledge graph"""
    CAUSAL = "causal"  # A causes B
    TEMPORAL = "temporal"  # A precedes B
    SPATIAL = "spatial"  # A near B
    SEMANTIC = "semantic"  # A related to B
    CONTRADICTS = "contradicts"  # A contradicts B
    SUPPORTS = "supports"  # A supports B
    INHIBITS = "inhibits"  # A inhibits B
    ENABLES = "enables"  # A enables B


@dataclass
class Vertex:
    """Vertex in the knowledge graph"""
    id: int
    vertex_type: VertexType
    content: Any  # Semantic content
    confidence: float = 1.0  # Belief confidence
    timestamp: int = 0  # When created
    activation: float = 0.0  # Current activation level

    # For location vertices
    position: Optional[Tuple[int, int]] = None

    # Metadata
    source_agent: int = -1
    decay_rate: float = 0.01


@dataclass
class Edge:
    """Edge in the knowledge graph"""
    id: int
    from_vertex: int
    to_vertex: int
    edge_type: EdgeType
    weight: float = 1.0  # Base weight
    pheromone: float = 0.1  # Ant pheromone level
    traversal_count: int = 0

    # For temporal edges
    time_delta: int = 0

    # Confidence in this relationship
    confidence: float = 1.0


class SemanticAnt:
    """
    Ant agent that traverses the knowledge graph.

    Ants explore paths based on pheromone and heuristic,
    depositing pheromone on useful paths.
    """

    def __init__(self, ant_id: int, start_vertex: int):
        self.id = ant_id
        self.current_vertex = start_vertex
        self.path: List[int] = [start_vertex]
        self.path_edges: List[int] = []
        self.energy: float = 1.0
        self.found_goal: bool = False
        self.contradiction_found: bool = False


class SemanticGraph:
    """
    Knowledge graph with ant-based exploration and semantic entropy.

    Key features:
    1. Dynamic graph construction from observations
    2. Pheromone-weighted edges for information routing
    3. Ant colony traversal for pattern discovery
    4. Semantic entropy calculation for coherence
    5. Contradiction detection and resolution

    PHASE 2: Information Thermodynamics
    ------------------------------------
    Adding vertices, edges, and querying have info costs.
    Agents must have sufficient info_energy to modify the graph.
    """

    def __init__(self, config: SemanticConfig,
                 info_costs: Optional[InfoCostConfig] = None):
        self.config = config

        # PHASE 2: Info cost configuration
        self.info_costs = info_costs or InfoCostConfig()

        # Graph storage
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[int, Edge] = {}

        # Adjacency lists
        self.out_edges: Dict[int, List[int]] = defaultdict(list)
        self.in_edges: Dict[int, List[int]] = defaultdict(list)

        # ID counters
        self.next_vertex_id = 0
        self.next_edge_id = 0

        # Ants for exploration
        self.ants: List[SemanticAnt] = []

        # Contradiction tracking
        self.contradictions: List[Tuple[int, int]] = []

        # Semantic entropy tracking
        self.entropy_history: List[float] = []
        self.temperature = config.semantic_temperature_init

        # Step counter
        self.step_count = 0

        # PHASE 2: Info dissipation tracking
        self.info_spent_this_step = 0.0
        self.operations_blocked_by_info = 0

    def add_vertex(self, vertex_type: VertexType, content: Any,
                   confidence: float = 1.0,
                   position: Optional[Tuple[int, int]] = None,
                   source_agent: int = -1,
                   agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """
        Add a vertex to the graph.

        PHASE 2: Adding vertices costs info_energy (cost_semantic_vertex).
        Silent failure if insufficient energy.

        Args:
            vertex_type: Type of vertex
            content: Semantic content
            confidence: Confidence level
            position: Spatial position (for location vertices)
            source_agent: ID of agent that created this
            agent_state: Optional agent state for info cost checking

        Returns:
            Vertex ID, or None if blocked by info cost
        """
        # PHASE 2: Check info cost
        if agent_state is not None:
            cost = self.info_costs.cost_semantic_vertex
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                self.operations_blocked_by_info += 1
                return None
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost

        if len(self.vertices) >= self.config.max_vertices:
            # Remove lowest activation vertex
            self._prune_lowest_activation()

        vertex = Vertex(
            id=self.next_vertex_id,
            vertex_type=vertex_type,
            content=content,
            confidence=confidence,
            timestamp=self.step_count,
            position=position,
            source_agent=source_agent,
        )

        self.vertices[vertex.id] = vertex
        self.next_vertex_id += 1

        return vertex.id

    def add_edge(self, from_vertex: int, to_vertex: int,
                 edge_type: EdgeType, weight: float = 1.0,
                 confidence: float = 1.0,
                 agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """
        Add an edge between vertices.

        PHASE 2: Adding edges costs info_energy (cost_semantic_edge).
        Silent failure if insufficient energy.

        Args:
            from_vertex: Source vertex ID
            to_vertex: Target vertex ID
            edge_type: Type of relationship
            weight: Edge weight
            confidence: Confidence in relationship
            agent_state: Optional agent state for info cost checking

        Returns:
            Edge ID or None if invalid or blocked by info cost
        """
        if from_vertex not in self.vertices or to_vertex not in self.vertices:
            return None

        # Check max edges per vertex
        if len(self.out_edges[from_vertex]) >= self.config.max_edges_per_vertex:
            return None

        # PHASE 2: Check info cost
        if agent_state is not None:
            cost = self.info_costs.cost_semantic_edge
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                self.operations_blocked_by_info += 1
                return None
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost

        edge = Edge(
            id=self.next_edge_id,
            from_vertex=from_vertex,
            to_vertex=to_vertex,
            edge_type=edge_type,
            weight=weight,
            pheromone=self.config.semantic_pheromone_deposit,
            confidence=confidence,
        )

        self.edges[edge.id] = edge
        self.out_edges[from_vertex].append(edge.id)
        self.in_edges[to_vertex].append(edge.id)
        self.next_edge_id += 1

        # Check for contradictions
        if edge_type == EdgeType.CONTRADICTS:
            self.contradictions.append((from_vertex, to_vertex))

        return edge.id

    def _prune_lowest_activation(self):
        """Remove vertex with lowest activation"""
        if not self.vertices:
            return

        min_vertex = min(self.vertices.values(), key=lambda v: v.activation)

        # Remove edges
        for edge_id in list(self.out_edges[min_vertex.id]):
            self._remove_edge(edge_id)
        for edge_id in list(self.in_edges[min_vertex.id]):
            self._remove_edge(edge_id)

        del self.vertices[min_vertex.id]
        del self.out_edges[min_vertex.id]
        del self.in_edges[min_vertex.id]

    def _remove_edge(self, edge_id: int):
        """Remove an edge"""
        if edge_id not in self.edges:
            return

        edge = self.edges[edge_id]
        self.out_edges[edge.from_vertex].remove(edge_id)
        self.in_edges[edge.to_vertex].remove(edge_id)
        del self.edges[edge_id]

    def get_neighbors(self, vertex_id: int) -> List[Tuple[int, int, float]]:
        """
        Get neighboring vertices with edge info.

        Returns:
            List of (neighbor_id, edge_id, combined_weight)
        """
        neighbors = []

        for edge_id in self.out_edges[vertex_id]:
            edge = self.edges[edge_id]
            # Combined weight includes pheromone
            weight = edge.weight * (1 + edge.pheromone)
            neighbors.append((edge.to_vertex, edge_id, weight))

        return neighbors

    def spawn_ant(self, start_vertex: int) -> int:
        """Spawn a new ant at a vertex"""
        ant = SemanticAnt(len(self.ants), start_vertex)
        self.ants.append(ant)

        if start_vertex in self.vertices:
            self.vertices[start_vertex].activation += 0.1

        return ant.id

    def step_ants(self, goal_vertices: Optional[Set[int]] = None):
        """
        Advance all ants one step.

        Ants move based on pheromone-weighted probabilistic routing.

        Args:
            goal_vertices: Set of goal vertex IDs
        """
        for ant in self.ants:
            if ant.energy <= 0:
                continue

            current = ant.current_vertex
            neighbors = self.get_neighbors(current)

            if not neighbors:
                ant.energy = 0  # Dead end
                continue

            # Compute transition probabilities
            probs = []
            for neighbor_id, edge_id, weight in neighbors:
                edge = self.edges[edge_id]

                # Avoid revisiting (unless no choice)
                if neighbor_id in ant.path and len(neighbors) > 1:
                    prob = 0.001
                else:
                    prob = np.power(edge.pheromone + 0.001, 1.0) * weight

                probs.append(prob)

            # Normalize
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(neighbors)] * len(neighbors)

            # Choose next vertex
            idx = np.random.choice(len(neighbors), p=probs)
            next_vertex, edge_id, _ = neighbors[idx]

            # Move ant
            ant.path.append(next_vertex)
            ant.path_edges.append(edge_id)
            ant.current_vertex = next_vertex
            ant.energy -= 0.1

            # Activate vertex
            if next_vertex in self.vertices:
                self.vertices[next_vertex].activation += 0.1

            # Check for goal
            if goal_vertices and next_vertex in goal_vertices:
                ant.found_goal = True

            # Check for contradiction
            vertex = self.vertices.get(next_vertex)
            if vertex and vertex.vertex_type == VertexType.BELIEF:
                # Check if contradicts any belief in path
                for prev_vid in ant.path[:-1]:
                    prev = self.vertices.get(prev_vid)
                    if prev and prev.vertex_type == VertexType.BELIEF:
                        if self._check_contradiction(prev, vertex):
                            ant.contradiction_found = True

            # Increment edge traversal
            self.edges[edge_id].traversal_count += 1

    def _check_contradiction(self, v1: Vertex, v2: Vertex) -> bool:
        """Check if two vertices contradict"""
        # Simple check - could be more sophisticated
        if v1.content == v2.content:
            return False

        # Check for explicit contradiction edge
        for edge_id in self.out_edges[v1.id]:
            edge = self.edges[edge_id]
            if edge.to_vertex == v2.id and edge.edge_type == EdgeType.CONTRADICTS:
                return True

        return False

    def deposit_pheromone(self, reinforce_successful: bool = True):
        """
        Deposit pheromone based on ant paths.

        Successful paths (found goal) get extra pheromone.
        """
        for ant in self.ants:
            if ant.energy <= 0 and not ant.found_goal:
                continue

            amount = self.config.semantic_pheromone_deposit
            if ant.found_goal and reinforce_successful:
                amount *= 2.0

            # Deposit on path edges
            for edge_id in ant.path_edges:
                if edge_id in self.edges:
                    self.edges[edge_id].pheromone += amount

    def evaporate_pheromone(self):
        """Apply pheromone evaporation to all edges"""
        for edge in self.edges.values():
            edge.pheromone *= (1 - self.config.semantic_pheromone_decay)
            edge.pheromone = max(0.01, edge.pheromone)

    def decay_activations(self):
        """Decay vertex activations over time"""
        for vertex in self.vertices.values():
            vertex.activation *= 0.99
            vertex.confidence *= (1 - vertex.decay_rate)

    def compute_semantic_entropy(self) -> float:
        """
        Compute semantic entropy of the graph.

        Based on information thermodynamics from the DCA framework.
        Higher entropy = less coherent beliefs.

        S = -Σ p_i * log(p_i)

        Where p_i is the normalized activation/confidence.
        """
        if not self.vertices:
            return 0.0

        # Get confidence distribution for belief vertices
        confidences = []
        for v in self.vertices.values():
            if v.vertex_type == VertexType.BELIEF:
                confidences.append(v.confidence)

        if not confidences:
            return 0.0

        # Normalize
        total = sum(confidences)
        if total <= 0:
            return 0.0

        probs = [c / total for c in confidences]

        # Compute entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p + 1e-10)

        return entropy

    def compute_contradiction_cost(self) -> float:
        """
        Compute cost of contradictions in the graph.

        Cost increases with confidence of contradicting beliefs.
        """
        cost = 0.0

        for v1_id, v2_id in self.contradictions:
            v1 = self.vertices.get(v1_id)
            v2 = self.vertices.get(v2_id)

            if v1 and v2:
                cost += self.config.contradiction_resolution_cost * v1.confidence * v2.confidence

        return cost

    def compute_coherence(self) -> float:
        """
        Compute overall coherence of the knowledge graph.

        Coherence = 1 - normalized_entropy - normalized_contradiction_cost
        """
        entropy = self.compute_semantic_entropy()
        max_entropy = np.log(max(1, len(self.vertices)))
        normalized_entropy = entropy / max(1.0, max_entropy)

        contradiction_cost = self.compute_contradiction_cost()
        max_cost = len(self.contradictions) * self.config.contradiction_resolution_cost
        normalized_cost = contradiction_cost / max(1.0, max_cost)

        coherence = 1.0 - 0.5 * normalized_entropy - 0.5 * normalized_cost
        return max(0.0, min(1.0, coherence))

    def resolve_contradiction(self, v1_id: int, v2_id: int) -> int:
        """
        Resolve contradiction between two vertices.

        Returns the ID of the surviving vertex (higher confidence wins).
        Lower confidence vertex has its confidence reduced.
        """
        v1 = self.vertices.get(v1_id)
        v2 = self.vertices.get(v2_id)

        if not v1 or not v2:
            return -1

        # Higher confidence wins
        if v1.confidence >= v2.confidence:
            winner, loser = v1, v2
        else:
            winner, loser = v2, v1

        # Reduce loser's confidence
        loser.confidence *= 0.5

        # Remove contradiction if loser is now very weak
        if loser.confidence < 0.1:
            if (v1_id, v2_id) in self.contradictions:
                self.contradictions.remove((v1_id, v2_id))
            if (v2_id, v1_id) in self.contradictions:
                self.contradictions.remove((v2_id, v1_id))

        return winner.id

    def spread_activation(self, source_vertices: List[int],
                          spread_rate: float = 0.5,
                          steps: int = 3):
        """
        Spread activation from source vertices through the graph.

        This simulates semantic priming and association.

        Args:
            source_vertices: Starting vertices
            spread_rate: Fraction of activation to spread
            steps: Number of spreading steps
        """
        for source in source_vertices:
            if source in self.vertices:
                self.vertices[source].activation = 1.0

        for _ in range(steps):
            new_activations = {}

            for vid, vertex in self.vertices.items():
                if vertex.activation <= 0.01:
                    continue

                spread_amount = vertex.activation * spread_rate

                for edge_id in self.out_edges[vid]:
                    edge = self.edges[edge_id]
                    target = edge.to_vertex

                    if target not in new_activations:
                        new_activations[target] = 0.0

                    # Weight by edge weight and pheromone
                    factor = edge.weight * (1 + edge.pheromone * 0.1)
                    new_activations[target] += spread_amount * factor

            # Apply new activations
            for vid, activation in new_activations.items():
                if vid in self.vertices:
                    self.vertices[vid].activation = min(
                        1.0,
                        self.vertices[vid].activation + activation
                    )

    def find_path(self, from_vertex: int, to_vertex: int,
                  max_depth: int = 10,
                  agent_state: Optional['AgentState'] = None) -> Optional[List[int]]:
        """
        Find path between vertices using A* with pheromone heuristic.

        PHASE 2: Querying the graph costs info_energy (cost_semantic_query).
        Silent failure if insufficient energy.

        Args:
            from_vertex: Start vertex
            to_vertex: End vertex
            max_depth: Maximum path length
            agent_state: Optional agent state for info cost checking

        Returns:
            List of vertex IDs forming path, or None
        """
        # PHASE 2: Check info cost for queries
        if agent_state is not None:
            cost = self.info_costs.cost_semantic_query
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                self.operations_blocked_by_info += 1
                return None
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost

        if from_vertex not in self.vertices or to_vertex not in self.vertices:
            return None

        # Priority queue: (cost, depth, vertex, path)
        heap = [(0, 0, from_vertex, [from_vertex])]
        visited = set()

        while heap:
            cost, depth, current, path = heapq.heappop(heap)

            if current == to_vertex:
                return path

            if depth >= max_depth or current in visited:
                continue

            visited.add(current)

            for neighbor_id, edge_id, weight in self.get_neighbors(current):
                if neighbor_id not in visited:
                    edge = self.edges[edge_id]
                    # Cost inversely proportional to pheromone
                    edge_cost = 1.0 / (weight + edge.pheromone + 0.1)
                    new_path = path + [neighbor_id]

                    heapq.heappush(heap, (
                        cost + edge_cost,
                        depth + 1,
                        neighbor_id,
                        new_path
                    ))

        return None

    def update(self, dt: float):
        """
        Full update step for the semantic graph.

        1. Step ants
        2. Deposit pheromone
        3. Evaporate pheromone
        4. Decay activations
        5. Compute entropy
        """
        # Run ant traversal
        goal_vertices = {v.id for v in self.vertices.values()
                        if v.vertex_type == VertexType.GOAL}

        for _ in range(self.config.traversal_steps_per_update):
            self.step_ants(goal_vertices)

        # Pheromone dynamics
        self.deposit_pheromone()
        self.evaporate_pheromone()

        # Decay
        self.decay_activations()

        # Track entropy
        entropy = self.compute_semantic_entropy()
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > 100:
            self.entropy_history.pop(0)

        # Update temperature based on entropy
        target_entropy = 0.5 * np.log(max(1, len(self.vertices)))
        if entropy > target_entropy:
            self.temperature *= 1.01  # Heat up to explore
        else:
            self.temperature *= 0.99  # Cool down to exploit

        self.temperature = np.clip(self.temperature, 0.1, 10.0)

        # Remove dead ants and spawn new ones
        self.ants = [a for a in self.ants if a.energy > 0]

        # Spawn new ants at high-activation vertices
        active_vertices = sorted(
            self.vertices.keys(),
            key=lambda v: self.vertices[v].activation,
            reverse=True
        )[:5]

        for vid in active_vertices:
            if len(self.ants) < 20:
                self.spawn_ant(vid)

        self.step_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "n_vertices": len(self.vertices),
            "n_edges": len(self.edges),
            "n_contradictions": len(self.contradictions),
            "semantic_entropy": self.compute_semantic_entropy(),
            "coherence": self.compute_coherence(),
            "temperature": self.temperature,
            "n_ants": len(self.ants),
            "avg_pheromone": np.mean([e.pheromone for e in self.edges.values()])
                            if self.edges else 0.0,
            # PHASE 2: Info dissipation metrics
            "info_spent_this_step": self.info_spent_this_step,
            "operations_blocked_by_info": self.operations_blocked_by_info,
        }

    def get_info_dissipation(self) -> float:
        """
        PHASE 2: Get info energy spent this step.

        Used by Overmind to observe global info dissipation rate.
        """
        return self.info_spent_this_step

    def reset_step_tracking(self):
        """
        PHASE 2: Reset per-step tracking variables.

        Call at the beginning of each step.
        """
        self.info_spent_this_step = 0.0
        self.operations_blocked_by_info = 0

    def consolidate(self, prune_low_confidence: bool = True,
                    remove_weak_edges: bool = True,
                    clear_ants: bool = True,
                    confidence_threshold: float = 0.2,
                    edge_pheromone_threshold: float = 0.05) -> Dict[str, int]:
        """
        PHASE 3: Consolidate the semantic graph at episode end.

        This is a SLOW time-scale operation that happens only at episode boundaries.
        It prevents graph bloat and improves coherence by:
        1. Pruning low-confidence vertices (noise)
        2. Removing weak edges (unused paths)
        3. Clearing ants (reset exploration)
        4. Boosting surviving vertices/edges (survival = importance)

        Args:
            prune_low_confidence: Remove vertices with low confidence
            remove_weak_edges: Remove edges with low pheromone
            clear_ants: Clear all ants for fresh exploration next episode
            confidence_threshold: Vertices below this confidence are pruned
            edge_pheromone_threshold: Edges below this pheromone level are pruned

        Returns:
            Dict with counts of pruned items
        """
        stats = {
            "vertices_pruned": 0,
            "edges_pruned": 0,
            "ants_cleared": len(self.ants) if clear_ants else 0,
            "vertices_boosted": 0,
            "edges_boosted": 0,
        }

        # === 1. Identify low-confidence vertices to prune ===
        if prune_low_confidence:
            vertices_to_remove = []
            for vid, vertex in self.vertices.items():
                if vertex.confidence < confidence_threshold:
                    vertices_to_remove.append(vid)

            # Remove identified vertices and their edges
            for vid in vertices_to_remove:
                self._remove_vertex(vid)
                stats["vertices_pruned"] += 1

        # === 2. Remove weak edges ===
        if remove_weak_edges:
            edges_to_remove = []
            for eid, edge in self.edges.items():
                if edge.pheromone < edge_pheromone_threshold:
                    # Also check traversal count - if heavily traversed, keep it
                    if edge.traversal_count < 3:
                        edges_to_remove.append(eid)

            for eid in edges_to_remove:
                self._remove_edge(eid)
                stats["edges_pruned"] += 1

        # === 3. Boost surviving vertices and edges ===
        # Survival through consolidation = importance signal
        for vertex in self.vertices.values():
            # Small confidence boost for survivors
            vertex.confidence = min(1.0, vertex.confidence + 0.05)
            stats["vertices_boosted"] += 1

        for edge in self.edges.values():
            # Edges that survived get a pheromone boost
            edge.pheromone = min(1.0, edge.pheromone * 1.1)
            # Reset traversal count for next episode
            edge.traversal_count = 0
            stats["edges_boosted"] += 1

        # === 4. Clear ants for fresh exploration ===
        if clear_ants:
            self.ants.clear()

        # === 5. Cool temperature toward equilibrium ===
        self.temperature = max(
            self.config.semantic_temperature_min,
            self.temperature * 0.9
        )

        return stats

    def _remove_vertex(self, vid: int):
        """Remove a vertex and all its connected edges"""
        if vid not in self.vertices:
            return

        # Get all connected edges
        out_edges = list(self.out_edges.get(vid, set()))
        in_edges = list(self.in_edges.get(vid, set()))

        # Remove edges
        for eid in out_edges + in_edges:
            self._remove_edge(eid)

        # Remove vertex
        del self.vertices[vid]

        # Clean up edge maps
        self.out_edges.pop(vid, None)
        self.in_edges.pop(vid, None)

    def _remove_edge(self, eid: int):
        """Remove an edge from the graph"""
        if eid not in self.edges:
            return

        edge = self.edges[eid]

        # Remove from adjacency (using correct attribute names)
        if edge.from_vertex in self.out_edges:
            self.out_edges[edge.from_vertex].discard(eid)
        if edge.to_vertex in self.in_edges:
            self.in_edges[edge.to_vertex].discard(eid)

        # Remove edge
        del self.edges[eid]

    def reset(self):
        """Reset graph to initial state"""
        self.vertices.clear()
        self.edges.clear()
        self.out_edges.clear()
        self.in_edges.clear()
        self.ants.clear()
        self.contradictions.clear()
        self.entropy_history.clear()
        self.next_vertex_id = 0
        self.next_edge_id = 0
        self.temperature = self.config.semantic_temperature_init
        self.step_count = 0

        # PHASE 2: Reset info tracking
        self.info_spent_this_step = 0.0
        self.operations_blocked_by_info = 0


class ColonySemanticSystem:
    """
    Colony-wide semantic construction system.

    Manages shared knowledge graph for the entire colony,
    integrating observations from all agents.

    PHASE 2: All graph modifications have info costs when agent_state is provided.
    """

    def __init__(self, config: SemanticConfig, n_agents: int,
                 info_costs: Optional[InfoCostConfig] = None):
        self.config = config
        self.n_agents = n_agents

        # Shared knowledge graph
        self.shared_graph = SemanticGraph(config, info_costs)

        # Per-agent local graphs (subset views)
        self.agent_views: Dict[int, Set[int]] = defaultdict(set)

        # Observation integration buffer
        # Now includes optional agent_state for info cost checking
        self.pending_observations: List[Tuple[int, VertexType, Any, Optional[Tuple[int, int]], Optional['AgentState']]] = []

    def add_observation(self, agent_id: int, obs_type: VertexType,
                        content: Any, position: Optional[Tuple[int, int]] = None,
                        agent_state: Optional['AgentState'] = None):
        """
        Add an observation from an agent.

        PHASE 2: Observations are buffered with agent_state for info cost checking
        during integration.

        Observations are buffered and integrated during update.
        """
        self.pending_observations.append((agent_id, obs_type, content, position, agent_state))

    def integrate_observations(self):
        """
        Integrate pending observations into the graph.

        PHASE 2: If agent_state was provided, checks and deducts info_energy.
        """
        for agent_id, obs_type, content, position, agent_state in self.pending_observations:
            # Check for duplicate content
            existing = None
            for v in self.shared_graph.vertices.values():
                if v.content == content:
                    existing = v
                    break

            if existing:
                # Reinforce existing vertex (no info cost for reinforcement)
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.activation += 0.2
                self.agent_views[agent_id].add(existing.id)
            else:
                # Create new vertex (info cost applies)
                vid = self.shared_graph.add_vertex(
                    vertex_type=obs_type,
                    content=content,
                    position=position,
                    source_agent=agent_id,
                    agent_state=agent_state,  # PHASE 2: Info cost check
                )

                if vid is not None:  # Only if not blocked by info cost
                    self.agent_views[agent_id].add(vid)

                    # Link to nearby vertices (edge costs apply)
                    if position:
                        for v in self.shared_graph.vertices.values():
                            if v.position and v.id != vid:
                                dist = abs(v.position[0] - position[0]) + abs(v.position[1] - position[1])
                                if dist <= 3:
                                    self.shared_graph.add_edge(
                                        vid, v.id,
                                        EdgeType.SPATIAL,
                                        weight=1.0 / (dist + 1),
                                        agent_state=agent_state,  # PHASE 2: Info cost check
                                    )

        self.pending_observations.clear()

    def update(self, dt: float):
        """Update the colony semantic system"""
        self.integrate_observations()
        self.shared_graph.update(dt)

    def get_agent_beliefs(self, agent_id: int) -> List[Vertex]:
        """Get beliefs visible to a specific agent"""
        beliefs = []
        for vid in self.agent_views[agent_id]:
            if vid in self.shared_graph.vertices:
                v = self.shared_graph.vertices[vid]
                if v.vertex_type == VertexType.BELIEF:
                    beliefs.append(v)
        return beliefs

    def get_coherence(self) -> float:
        """Get overall colony belief coherence"""
        return self.shared_graph.compute_coherence()

    def get_info_dissipation(self) -> float:
        """PHASE 2: Get info energy spent this step."""
        return self.shared_graph.get_info_dissipation()

    def reset_step_tracking(self):
        """PHASE 2: Reset per-step tracking."""
        self.shared_graph.reset_step_tracking()

    def consolidate(self, clear_ants: bool = True) -> Dict[str, int]:
        """
        PHASE 3: Consolidate semantic graph at episode end.

        This is a SLOW time-scale operation that happens at episode boundaries.
        It cleans up the shared knowledge graph to prevent bloat and
        improve coherence for the next episode.

        Args:
            clear_ants: Whether to clear all semantic ants

        Returns:
            Statistics about consolidation
        """
        stats = self.shared_graph.consolidate(
            prune_low_confidence=True,
            remove_weak_edges=True,
            clear_ants=clear_ants,
            confidence_threshold=0.2,
            edge_pheromone_threshold=0.05,
        )

        # Also clean up agent views - remove references to pruned vertices
        for agent_id in self.agent_views:
            valid_vertices = {vid for vid in self.agent_views[agent_id]
                            if vid in self.shared_graph.vertices}
            self.agent_views[agent_id] = valid_vertices

        return stats

    def reset(self):
        """Reset semantic system"""
        self.shared_graph.reset()
        self.agent_views.clear()
        self.pending_observations.clear()
