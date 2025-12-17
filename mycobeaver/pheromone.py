"""
MycoBeaver Pheromone Field System
==================================
Ant Colony Optimization-inspired pheromone routing.

Based on MycoBeaver Simulator Design Plan Section 2.2:
- Edge-based pheromone storage τ_ij
- Evaporation: τ_{ij}(t+1) = (1 - ρ) * τ_{ij}(t)
- Deposition: τ_{ij} += δ_k for traversals
- Routing probability: p_ij ∝ τ_ij^α * η_ij^β
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from .config import PheromoneConfig, InfoCostConfig

# Forward reference for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .environment import AgentState


@dataclass
class PheromoneEdge:
    """Represents pheromone on an edge between two cells"""
    from_cell: Tuple[int, int]
    to_cell: Tuple[int, int]
    pheromone: float = 0.0

    # Track traversal counts for statistics
    traversal_count: int = 0
    success_count: int = 0


class PheromoneField:
    """
    Edge-based pheromone field for ant-style routing.

    Pheromones are stored on edges between adjacent cells.
    Each cell has edges to its 8 neighbors (or fewer at boundaries).

    Key operations:
    1. Evaporation: Global decay of all pheromone values
    2. Deposition: Agents deposit pheromone when moving
    3. Routing: Probability distribution for movement decisions

    PHASE 2: Information Thermodynamics
    ------------------------------------
    Depositing pheromone costs info_energy.
    Agents must have sufficient info_energy to deposit.
    """

    def __init__(self, config: PheromoneConfig, grid_size: int,
                 info_costs: Optional[InfoCostConfig] = None):
        self.config = config
        self.grid_size = grid_size

        # PHASE 2: Info cost configuration
        self.info_costs = info_costs or InfoCostConfig()

        # Store pheromones as a 4D array: [y, x, dy_offset, dx_offset]
        # Offsets are encoded as: 0=(-1,-1), 1=(-1,0), 2=(-1,1), 3=(0,-1),
        #                         4=(0,1), 5=(1,-1), 6=(1,0), 7=(1,1)
        # This represents 8 directional edges from each cell
        self.pheromone_grid = np.ones((grid_size, grid_size, 8)) * config.min_pheromone

        # Direction offset mapping (index -> (dy, dx))
        self.direction_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Reverse mapping: (dy, dx) -> index
        self.offset_to_index = {offset: idx for idx, offset in enumerate(self.direction_offsets)}

        # Track total depositions for statistics
        self.total_depositions = 0
        self.step_count = 0

        # PHASE 2: Info dissipation tracking
        self.info_spent_this_step = 0.0
        self.deposits_blocked_by_info = 0

    def evaporate(self, dt: float):
        """
        Apply global evaporation to all pheromone values.

        τ_{ij}(t+1) = (1 - ρ) * τ_{ij}(t)

        Args:
            dt: Time step (used for continuous-time adjustment)
        """
        decay_factor = (1 - self.config.evaporation_rate) ** dt
        self.pheromone_grid *= decay_factor

        # Ensure minimum pheromone level (for exploration)
        self.pheromone_grid = np.maximum(self.pheromone_grid, self.config.min_pheromone)

        self.step_count += 1

    def deposit(self, from_cell: Tuple[int, int], to_cell: Tuple[int, int],
                amount: Optional[float] = None, success: bool = False,
                agent_state: Optional['AgentState'] = None) -> bool:
        """
        Deposit pheromone on edge from one cell to another.

        PHASE 2: Depositing costs info_energy (cost_pheromone_deposit).
        Silent failure if insufficient energy.

        τ_{ij} += δ_k
        If success (e.g., found food), multiply by success multiplier.

        Args:
            from_cell: (y, x) of source cell
            to_cell: (y, x) of destination cell
            amount: Pheromone amount (default: base_deposit)
            success: Whether this was a successful traversal
            agent_state: Optional agent state for info cost checking

        Returns:
            True if deposit succeeded, False if blocked by info cost
        """
        y1, x1 = from_cell
        y2, x2 = to_cell

        # Compute direction
        dy = y2 - y1
        dx = x2 - x1

        # Clamp to valid directions
        dy = np.clip(dy, -1, 1)
        dx = np.clip(dx, -1, 1)

        if (dy, dx) == (0, 0):
            return False  # No movement

        direction_idx = self.offset_to_index.get((dy, dx))
        if direction_idx is None:
            return False  # Invalid direction

        # PHASE 2: Check info cost
        if agent_state is not None:
            cost = self.info_costs.cost_pheromone_deposit
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                self.deposits_blocked_by_info += 1
                return False
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost

        # Compute deposit amount
        if amount is None:
            amount = self.config.base_deposit

        if success:
            amount *= self.config.success_deposit_multiplier

        # Apply deposit
        if 0 <= y1 < self.grid_size and 0 <= x1 < self.grid_size:
            self.pheromone_grid[y1, x1, direction_idx] += amount

            # Clip to maximum
            self.pheromone_grid[y1, x1, direction_idx] = min(
                self.pheromone_grid[y1, x1, direction_idx],
                self.config.max_pheromone
            )

            self.total_depositions += 1
            return True

        return False

    def update(self, movements: List[Tuple[Tuple[int, int], Tuple[int, int]]],
               successful_paths: Optional[List[bool]] = None):
        """
        Update pheromones based on agent movements.

        Args:
            movements: List of (from_cell, to_cell) tuples
            successful_paths: Optional list indicating which movements were successful
        """
        if successful_paths is None:
            successful_paths = [False] * len(movements)

        for i, (from_cell, to_cell) in enumerate(movements):
            success = successful_paths[i] if i < len(successful_paths) else False
            self.deposit(from_cell, to_cell, success=success)

    def get_pheromone(self, from_cell: Tuple[int, int], to_cell: Tuple[int, int]) -> float:
        """Get pheromone level on specific edge"""
        y1, x1 = from_cell
        y2, x2 = to_cell

        dy = np.clip(y2 - y1, -1, 1)
        dx = np.clip(x2 - x1, -1, 1)

        if (dy, dx) == (0, 0):
            return 0.0

        direction_idx = self.offset_to_index.get((dy, dx))
        if direction_idx is None:
            return 0.0

        if 0 <= y1 < self.grid_size and 0 <= x1 < self.grid_size:
            return self.pheromone_grid[y1, x1, direction_idx]
        return 0.0

    def get_average_level(self, y: int, x: int) -> float:
        """
        Get average pheromone level for all edges from a cell.
        Used for observation feature.

        Args:
            y: Row index
            x: Column index

        Returns:
            Average pheromone level (normalized)
        """
        if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
            avg = np.mean(self.pheromone_grid[y, x, :])
            # Normalize to [0, 1]
            return min(1.0, avg / (self.config.max_pheromone * 0.1))
        return 0.0

    def get_routing_probabilities(self, y: int, x: int,
                                   heuristic: Optional[np.ndarray] = None,
                                   blocked: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute routing probabilities for movement from a cell.

        p_ij ∝ τ_ij^α * η_ij^β + ε

        Args:
            y: Current row
            x: Current column
            heuristic: Optional array of heuristic values for each direction (shape: (8,))
                      (e.g., distance to goal, resource density)
            blocked: Optional boolean array indicating blocked directions

        Returns:
            Probability distribution over 8 directions
        """
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return np.ones(8) / 8  # Uniform if out of bounds

        # Get pheromone values
        pheromone = self.pheromone_grid[y, x, :].copy()

        # Apply alpha exponent
        pheromone_factor = np.power(pheromone, self.config.pheromone_alpha)

        # Apply heuristic if provided
        if heuristic is not None:
            heuristic_factor = np.power(np.maximum(heuristic, 0.001), self.config.heuristic_beta)
        else:
            heuristic_factor = np.ones(8)

        # Combine factors
        scores = pheromone_factor * heuristic_factor

        # Add exploration baseline
        scores += self.config.exploration_epsilon

        # Block invalid directions
        if blocked is not None:
            scores = np.where(blocked, 0.0, scores)

        # Check boundary conditions
        for i, (dy, dx) in enumerate(self.direction_offsets):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
                scores[i] = 0.0

        # Normalize to probabilities
        total = np.sum(scores)
        if total > 0:
            return scores / total
        else:
            # All blocked - return uniform over valid
            valid = np.ones(8)
            for i, (dy, dx) in enumerate(self.direction_offsets):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
                    valid[i] = 0.0
            if np.sum(valid) > 0:
                return valid / np.sum(valid)
            return np.ones(8) / 8  # Fallback

    def sample_direction(self, y: int, x: int,
                         heuristic: Optional[np.ndarray] = None,
                         blocked: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Sample a movement direction based on pheromone routing.

        Args:
            y: Current row
            x: Current column
            heuristic: Optional heuristic values
            blocked: Optional blocked directions

        Returns:
            (dy, dx) direction offset
        """
        probs = self.get_routing_probabilities(y, x, heuristic, blocked)

        # Sample direction
        direction_idx = np.random.choice(8, p=probs)
        return self.direction_offsets[direction_idx]

    def get_gradient(self, y: int, x: int) -> Tuple[float, float]:
        """
        Compute pheromone gradient at a cell.

        Returns the average direction of increasing pheromone.
        Useful for gradient-following behavior.

        Returns:
            (gy, gx) normalized gradient vector
        """
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return (0.0, 0.0)

        gy, gx = 0.0, 0.0

        for i, (dy, dx) in enumerate(self.direction_offsets):
            weight = self.pheromone_grid[y, x, i]
            gy += weight * dy
            gx += weight * dx

        # Normalize
        magnitude = np.sqrt(gy**2 + gx**2)
        if magnitude > 0.001:
            gy /= magnitude
            gx /= magnitude

        return (gy, gx)

    def reinforce_path(self, path: List[Tuple[int, int]],
                       quality: float = 1.0,
                       decay_along_path: bool = True,
                       agent_state: Optional['AgentState'] = None) -> int:
        """
        Reinforce pheromone along an entire path.

        PHASE 2: Path reinforcement costs info_energy (cost_pheromone_reinforce)
        for the whole path, not per-segment.

        Used when an agent successfully reaches a goal and wants
        to strengthen the path for others.

        Args:
            path: List of (y, x) cells in order
            quality: Quality multiplier (e.g., path efficiency)
            decay_along_path: Whether to decay deposit along path length
            agent_state: Optional agent state for info cost checking

        Returns:
            Number of path segments reinforced
        """
        if len(path) < 2:
            return 0

        # PHASE 2: Check info cost for path reinforcement (higher cost)
        if agent_state is not None:
            cost = self.info_costs.cost_pheromone_reinforce
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                self.deposits_blocked_by_info += 1
                return 0
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost

        n = len(path)
        reinforced = 0

        for i in range(len(path) - 1):
            from_cell = path[i]
            to_cell = path[i + 1]

            # Optionally decay deposit along path
            if decay_along_path:
                position_factor = 1.0 - (i / n) * 0.5  # 50% decay from start to end
            else:
                position_factor = 1.0

            amount = self.config.base_deposit * quality * position_factor
            # Don't pass agent_state here - already paid the cost
            if self.deposit(from_cell, to_cell, amount=amount, success=True, agent_state=None):
                reinforced += 1

        return reinforced

    def diffuse(self, rate: float = 0.1):
        """
        Diffuse pheromones to neighboring cells.

        This creates a more gradual gradient rather than sharp edges.

        Args:
            rate: Diffusion rate (fraction to spread to neighbors)
        """
        # Create padded array for convolution
        padded = np.pad(self.pheromone_grid, ((1, 1), (1, 1), (0, 0)), mode='edge')

        # Simple 3x3 averaging kernel for spatial diffusion
        new_grid = self.pheromone_grid.copy()

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                shifted = padded[1+dy:1+dy+self.grid_size, 1+dx:1+dx+self.grid_size, :]
                new_grid += rate * (shifted - self.pheromone_grid) / 8

        # Apply with clipping
        self.pheromone_grid = np.clip(
            new_grid,
            self.config.min_pheromone,
            self.config.max_pheromone
        )

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about the pheromone field"""
        return {
            "mean_pheromone": float(np.mean(self.pheromone_grid)),
            "max_pheromone": float(np.max(self.pheromone_grid)),
            "min_pheromone": float(np.min(self.pheromone_grid)),
            "std_pheromone": float(np.std(self.pheromone_grid)),
            "total_depositions": self.total_depositions,
            "steps": self.step_count,
            # PHASE 2: Info dissipation metrics
            "info_spent_this_step": self.info_spent_this_step,
            "deposits_blocked_by_info": self.deposits_blocked_by_info,
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
        self.deposits_blocked_by_info = 0

    def get_heatmap(self) -> np.ndarray:
        """
        Get a 2D heatmap of total pheromone per cell.

        Useful for visualization.

        Returns:
            (grid_size, grid_size) array of total pheromone levels
        """
        return np.sum(self.pheromone_grid, axis=2)

    def reset(self):
        """Reset pheromone field to initial state"""
        self.pheromone_grid = np.ones(
            (self.grid_size, self.grid_size, 8)
        ) * self.config.min_pheromone
        self.total_depositions = 0
        self.step_count = 0

        # PHASE 2: Reset info tracking
        self.info_spent_this_step = 0.0
        self.deposits_blocked_by_info = 0


class MultiPheromoneField:
    """
    Multiple pheromone types for different purposes.

    Implements the multi-pheromone system from the design:
    - Resource pheromone: Guides to food/wood sources
    - Danger pheromone: Warns of hazards
    - Project pheromone: Guides to construction sites
    - Home pheromone: Guides back to lodge

    PHASE 2: All deposit operations have info costs when agent_state is provided.
    """

    def __init__(self, config: PheromoneConfig, grid_size: int, n_types: int = 4,
                 info_costs: Optional[InfoCostConfig] = None):
        self.config = config
        self.grid_size = grid_size
        self.n_types = n_types

        # Pheromone type names
        self.type_names = ["resource", "danger", "project", "home"][:n_types]

        # Create separate fields for each type
        self.fields = {
            name: PheromoneField(config, grid_size, info_costs)
            for name in self.type_names
        }

    def evaporate(self, dt: float):
        """Evaporate all pheromone types"""
        for field in self.fields.values():
            field.evaporate(dt)

    def deposit(self, pheromone_type: str, from_cell: Tuple[int, int],
                to_cell: Tuple[int, int], amount: Optional[float] = None,
                success: bool = False,
                agent_state: Optional['AgentState'] = None) -> bool:
        """
        Deposit specific pheromone type.

        PHASE 2: Costs info_energy when agent_state is provided.
        """
        if pheromone_type in self.fields:
            return self.fields[pheromone_type].deposit(
                from_cell, to_cell, amount, success, agent_state
            )
        return False

    def get_combined_routing(self, y: int, x: int,
                             weights: Optional[Dict[str, float]] = None,
                             heuristic: Optional[np.ndarray] = None,
                             blocked: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get combined routing probabilities from all pheromone types.

        Args:
            y, x: Current position
            weights: Weights for each pheromone type
            heuristic: Heuristic values
            blocked: Blocked directions

        Returns:
            Combined probability distribution
        """
        if weights is None:
            weights = {name: 1.0 for name in self.type_names}

        combined_scores = np.zeros(8)

        for name, field in self.fields.items():
            w = weights.get(name, 1.0)
            probs = field.get_routing_probabilities(y, x, heuristic, blocked)
            combined_scores += w * probs

        # Normalize
        total = np.sum(combined_scores)
        if total > 0:
            return combined_scores / total
        return np.ones(8) / 8

    def get_average_level(self, y: int, x: int) -> float:
        """Get average pheromone level across all types"""
        total = 0.0
        for field in self.fields.values():
            total += field.get_average_level(y, x)
        return total / len(self.fields)

    def get_field(self, pheromone_type: str) -> Optional[PheromoneField]:
        """Get specific pheromone field"""
        return self.fields.get(pheromone_type)

    def get_info_dissipation(self) -> float:
        """PHASE 2: Get total info energy spent this step across all fields."""
        return sum(field.get_info_dissipation() for field in self.fields.values())

    def reset_step_tracking(self):
        """PHASE 2: Reset per-step tracking for all fields."""
        for field in self.fields.values():
            field.reset_step_tracking()

    def reset(self):
        """Reset all pheromone fields"""
        for field in self.fields.values():
            field.reset()
