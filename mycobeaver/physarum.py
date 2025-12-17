"""
MycoBeaver Physarum Network
============================
Physarum polycephalum-inspired adaptive transport network.

Based on MycoBeaver Simulator Design Plan Section 2.4:
- Edge conductivity D_ij adapts based on flow
- Flow Q_ij computed by solving pressure equations
- dD_ij/dt = α_D * g(|Q_ij|) - β_D * D_ij
- Network self-organizes to connect sources and sinks efficiently
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from .config import PhysarumConfig


@dataclass
class PhysarumEdge:
    """Edge in the Physarum network"""
    from_node: int  # Linearized index
    to_node: int
    conductivity: float
    length: float
    flow: float = 0.0


class PhysarumNetwork:
    """
    Physarum polycephalum-inspired adaptive transport network.

    The network overlays the grid and adapts its conductivity
    based on flow patterns between sources (resources) and sinks (lodge, projects).

    Key equations:
    1. Flow: Q_ij = D_ij * (p_i - p_j) / L_ij
    2. Conservation: Σ Q_ij = S_i (source/sink term)
    3. Adaptation: dD_ij/dt = α * g(|Q|) - β * D_ij

    The network naturally finds efficient paths connecting sources to sinks.
    """

    def __init__(self, config: PhysarumConfig, grid_size: int):
        self.config = config
        self.grid_size = grid_size
        self.n_nodes = grid_size * grid_size

        # Edge storage (adjacency structure)
        # For efficiency, we use sparse representation
        # Edges connect 4-neighbors (can extend to 8)
        self.neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.n_neighbors = 4

        # Conductivity matrix D_ij (stored as 3D: y, x, direction)
        self.conductivity = np.ones(
            (grid_size, grid_size, self.n_neighbors)
        ) * config.initial_conductivity

        # Edge lengths L_ij
        self.edge_lengths = np.ones(
            (grid_size, grid_size, self.n_neighbors)
        ) * config.base_length

        # Flow matrix Q_ij (updated each step)
        self.flow = np.zeros((grid_size, grid_size, self.n_neighbors))

        # Pressure at each node
        self.pressure = np.zeros((grid_size, grid_size))

        # Sources and sinks
        self.sources: Set[Tuple[int, int]] = set()
        self.sinks: Set[Tuple[int, int]] = set()

        # Track step count
        self.step_count = 0

    def _to_linear(self, y: int, x: int) -> int:
        """Convert (y, x) to linear index"""
        return y * self.grid_size + x

    def _to_2d(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (y, x)"""
        return idx // self.grid_size, idx % self.grid_size

    def set_edge_costs(self, grid_state):
        """
        Update edge lengths based on terrain.

        L_ij = L_0 * (1 + λ_z * |Δz| + λ_h * h_avg)

        Args:
            grid_state: Current grid state with elevation and water
        """
        for idx, (dy, dx) in enumerate(self.neighbor_offsets):
            # Shifted arrays for neighbor values
            elevation_neighbor = np.roll(np.roll(grid_state.elevation, dy, axis=0), dx, axis=1)
            water_neighbor = np.roll(np.roll(grid_state.water_depth, dy, axis=0), dx, axis=1)

            # Elevation difference
            delta_z = np.abs(grid_state.elevation - elevation_neighbor)

            # Average water depth
            avg_water = 0.5 * (grid_state.water_depth + water_neighbor)

            # Edge length
            self.edge_lengths[:, :, idx] = self.config.base_length * (
                1.0 +
                self.config.elevation_cost_factor * delta_z +
                self.config.water_cost_factor * avg_water
            )

    def solve_pressure(self, source_strengths: np.ndarray) -> np.ndarray:
        """
        Solve for pressure at each node given sources/sinks.

        Uses Kirchhoff's current law: Σ Q_ij = S_i
        Where Q_ij = D_ij * (p_i - p_j) / L_ij

        This is a linear system: A * p = s

        Args:
            source_strengths: Source/sink term at each node (positive = source)

        Returns:
            Pressure at each node
        """
        n = self.n_nodes

        # Build sparse matrix
        row_indices = []
        col_indices = []
        data = []

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                i = self._to_linear(y, x)
                diagonal = 0.0

                for idx, (dy, dx) in enumerate(self.neighbor_offsets):
                    ny, nx = y + dy, x + dx

                    if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                        j = self._to_linear(ny, nx)

                        # Conductance for this edge
                        D = self.conductivity[y, x, idx]
                        L = self.edge_lengths[y, x, idx]
                        G = D / L  # Conductance

                        # Off-diagonal entry
                        row_indices.append(i)
                        col_indices.append(j)
                        data.append(-G)

                        # Accumulate diagonal
                        diagonal += G

                # Diagonal entry
                row_indices.append(i)
                col_indices.append(i)
                data.append(diagonal + 1e-10)  # Small regularization

        # Create sparse matrix
        A = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n, n)
        )

        # Right-hand side (sources/sinks)
        rhs = source_strengths.flatten()

        # Ground one node (reference pressure)
        # Set pressure at corner to zero
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1.0
        rhs[0] = 0.0
        A = A.tocsr()

        # Solve
        try:
            pressure_flat = spsolve(A, rhs)
        except Exception:
            pressure_flat = np.zeros(n)

        return pressure_flat.reshape((self.grid_size, self.grid_size))

    def compute_flow(self):
        """
        Compute flow on all edges from pressure differences.

        Q_ij = D_ij * (p_i - p_j) / L_ij
        """
        for idx, (dy, dx) in enumerate(self.neighbor_offsets):
            # Neighbor pressure
            p_neighbor = np.roll(np.roll(self.pressure, dy, axis=0), dx, axis=1)

            # Flow = D * (p_i - p_j) / L
            self.flow[:, :, idx] = (
                self.conductivity[:, :, idx] *
                (self.pressure - p_neighbor) /
                self.edge_lengths[:, :, idx]
            )

    def adapt_conductivity(self, dt: float):
        """
        Adapt conductivity based on flow.

        dD_ij/dt = α_D * g(|Q_ij|) - β_D * D_ij

        Where g(|Q|) = |Q|^γ (flow reinforcement function)
        """
        # Flow magnitude
        flow_magnitude = np.abs(self.flow)

        # Reinforcement function g(|Q|) = |Q|^γ
        g_flow = np.power(flow_magnitude + 1e-8, self.config.flow_exponent)

        # Adaptation
        dD = dt * (
            self.config.reinforcement_rate * g_flow -
            self.config.decay_rate * self.conductivity
        )

        self.conductivity += dD

        # Clip to bounds
        self.conductivity = np.clip(
            self.conductivity,
            self.config.min_conductivity,
            self.config.max_conductivity
        )

    def update(self, sources: List[Tuple[int, int]],
               sinks: List[Tuple[int, int]],
               grid_state,
               dt: float):
        """
        Full update step for Physarum network.

        1. Set edge costs based on terrain
        2. Build source/sink strength array
        3. Solve for pressure
        4. Compute flow
        5. Adapt conductivity

        Args:
            sources: List of source positions (high vegetation)
            sinks: List of sink positions (lodge, projects)
            grid_state: Current grid state
            dt: Time step
        """
        # Store sources/sinks
        self.sources = set(sources)
        self.sinks = set(sinks)

        # 1. Update edge costs
        self.set_edge_costs(grid_state)

        # 2. Build source/sink array
        source_strengths = np.zeros((self.grid_size, self.grid_size))

        # Sources inject flow
        for (y, x) in sources:
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                source_strengths[y, x] += 1.0

        # Sinks absorb flow
        for (y, x) in sinks:
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                source_strengths[y, x] -= 1.0

        # Balance sources and sinks
        total_source = np.sum(source_strengths[source_strengths > 0])
        total_sink = -np.sum(source_strengths[source_strengths < 0])

        if total_source > 0 and total_sink > 0:
            # Scale to balance
            scale = min(total_source, total_sink)
            source_strengths[source_strengths > 0] *= scale / total_source
            source_strengths[source_strengths < 0] *= scale / total_sink

        # 3. Solve pressure
        self.pressure = self.solve_pressure(source_strengths)

        # 4. Compute flow
        self.compute_flow()

        # 5. Adapt conductivity
        self.adapt_conductivity(dt)

        self.step_count += 1

    def get_average_conductivity(self, y: int, x: int) -> float:
        """
        Get average conductivity for edges from a cell.

        Used for observation features.

        Args:
            y: Row index
            x: Column index

        Returns:
            Normalized average conductivity
        """
        if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
            avg = np.mean(self.conductivity[y, x, :])
            # Normalize to [0, 1]
            return min(1.0, avg / self.config.max_conductivity)
        return 0.0

    def get_flow_direction(self, y: int, x: int) -> Tuple[float, float]:
        """
        Get net flow direction at a cell.

        Returns vector pointing in direction of net outflow.

        Args:
            y: Row index
            x: Column index

        Returns:
            (fy, fx) normalized flow direction
        """
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return (0.0, 0.0)

        fy, fx = 0.0, 0.0

        for idx, (dy, dx) in enumerate(self.neighbor_offsets):
            flow = self.flow[y, x, idx]
            fy += flow * dy
            fx += flow * dx

        # Normalize
        mag = np.sqrt(fy**2 + fx**2)
        if mag > 1e-6:
            fy /= mag
            fx /= mag

        return (fy, fx)

    def get_path_to_sink(self, start: Tuple[int, int],
                         max_steps: int = 100) -> List[Tuple[int, int]]:
        """
        Follow high-flow edges from start toward sink.

        Useful for agents to find efficient paths.

        Args:
            start: Starting position (y, x)
            max_steps: Maximum path length

        Returns:
            List of positions forming path
        """
        path = [start]
        current = start

        for _ in range(max_steps):
            y, x = current

            if current in self.sinks:
                break

            # Find neighbor with highest outgoing flow
            best_flow = -float('inf')
            best_next = None

            for idx, (dy, dx) in enumerate(self.neighbor_offsets):
                ny, nx = y + dy, x + dx

                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    # Flow going toward neighbor
                    flow = self.flow[y, x, idx]

                    if flow > best_flow:
                        best_flow = flow
                        best_next = (ny, nx)

            if best_next is None or best_next in path:
                break

            current = best_next
            path.append(current)

        return path

    def get_conductivity_heatmap(self) -> np.ndarray:
        """
        Get 2D heatmap of total conductivity per cell.

        Returns:
            (grid_size, grid_size) array
        """
        return np.sum(self.conductivity, axis=2)

    def get_flow_magnitude_heatmap(self) -> np.ndarray:
        """
        Get 2D heatmap of total flow magnitude per cell.

        Returns:
            (grid_size, grid_size) array
        """
        return np.sum(np.abs(self.flow), axis=2)

    def get_statistics(self) -> Dict:
        """Get network statistics"""
        return {
            "mean_conductivity": float(np.mean(self.conductivity)),
            "max_conductivity": float(np.max(self.conductivity)),
            "mean_flow": float(np.mean(np.abs(self.flow))),
            "max_flow": float(np.max(np.abs(self.flow))),
            "n_sources": len(self.sources),
            "n_sinks": len(self.sinks),
            "steps": self.step_count,
        }

    def prune_weak_edges(self, threshold: float = 0.1):
        """
        Set very weak edges to minimum conductivity.

        Helps network converge to sparse structure.

        Args:
            threshold: Conductivity threshold (fraction of max)
        """
        cutoff = threshold * self.config.max_conductivity
        weak_mask = self.conductivity < cutoff
        self.conductivity[weak_mask] = self.config.min_conductivity

    def boost_edge(self, from_cell: Tuple[int, int], to_cell: Tuple[int, int],
                   amount: float = 0.1):
        """
        Manually boost conductivity on an edge.

        Used when agents successfully traverse a path.

        Args:
            from_cell: Start cell (y, x)
            to_cell: End cell (y, x)
            amount: Amount to boost
        """
        y1, x1 = from_cell
        y2, x2 = to_cell

        dy = np.clip(y2 - y1, -1, 1)
        dx = np.clip(x2 - x1, -1, 1)

        # Find direction index
        for idx, (ody, odx) in enumerate(self.neighbor_offsets):
            if ody == dy and odx == dx:
                if 0 <= y1 < self.grid_size and 0 <= x1 < self.grid_size:
                    self.conductivity[y1, x1, idx] = min(
                        self.config.max_conductivity,
                        self.conductivity[y1, x1, idx] + amount
                    )
                break

    def reset(self):
        """Reset network to initial state"""
        self.conductivity = np.ones(
            (self.grid_size, self.grid_size, self.n_neighbors)
        ) * self.config.initial_conductivity

        self.flow = np.zeros((self.grid_size, self.grid_size, self.n_neighbors))
        self.pressure = np.zeros((self.grid_size, self.grid_size))
        self.sources.clear()
        self.sinks.clear()
        self.step_count = 0
