"""
MycoBeaver Grid Environment
============================
Complete implementation of the 2D grid environment with hydrology dynamics.

Based on the MycoBeaver Simulator Design Plan Section 1:
- Grid state arrays (elevation, water, vegetation, soil, dams, lodges)
- Hydrological flow computation
- Vegetation and soil dynamics
- Gym-compatible interface
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces

from .config import (
    SimulationConfig, GridConfig, AgentConfig, ActionType, AgentRole,
    ProjectType, ProjectStatus
)


@dataclass
class GridState:
    """
    Complete state of the grid environment.
    All arrays are shape (grid_size, grid_size).
    """
    # Terrain (static)
    elevation: np.ndarray  # Height map

    # Hydrology (dynamic)
    water_depth: np.ndarray  # h_i - meters of water

    # Biology (dynamic)
    vegetation: np.ndarray  # v_i - biomass [0, max_vegetation]

    # Soil (dynamic)
    soil_moisture: np.ndarray  # m_i - [0, 1]

    # Structures (modified by agents)
    dam_permeability: np.ndarray  # d_i - [0, 1], 0=blocked, 1=no dam
    dam_integrity: np.ndarray  # Structure health [0, 1], 0=broken, 1=perfect
    lodge_map: np.ndarray  # Boolean indicating lodge locations

    # Agent presence (updated each step)
    agent_positions: np.ndarray  # Count of agents per cell


class HydrologyEngine:
    """
    Realistic water flow dynamics with sources, gradient flow, and pooling.

    Features:
    - Water sources (springs) at fixed locations that emit water
    - Gradient-based flow (water flows downhill based on terrain slope)
    - Pooling/accumulation in basins (low areas surrounded by higher terrain)
    - Rain events (periodic bursts of water)
    - Dam effects (reduce flow through, increase upstream retention)
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.size = config.grid_size

        # Precompute neighbor offsets (4-connected)
        self.neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Water source locations (will be set during reset)
        self.source_locations: List[Tuple[int, int]] = []

        # Flow momentum field (tracks water flow direction for inertia)
        self.flow_momentum_y = np.zeros((self.size, self.size))
        self.flow_momentum_x = np.zeros((self.size, self.size))

        # Rain event state
        self.rain_active = False
        self.rain_steps_remaining = 0

    def initialize_sources(self, elevation: np.ndarray) -> None:
        """
        Place water sources at elevated positions that can flow downhill.
        Sources are placed in the upper portion of the terrain.
        """
        self.source_locations = []
        n_sources = self.config.n_water_sources

        # Find good source locations (elevated areas in the upper third)
        upper_region = elevation[:self.size // 3, :]

        for _ in range(n_sources):
            # Find a high point in the upper region
            # Add some randomness to avoid clustering
            attempts = 0
            while attempts < 50:
                y = np.random.randint(0, self.size // 4)
                x = np.random.randint(self.size // 4, 3 * self.size // 4)

                # Check elevation is reasonable
                if elevation[y, x] > np.percentile(elevation, 60):
                    # Check not too close to existing sources
                    too_close = False
                    for sy, sx in self.source_locations:
                        if abs(y - sy) + abs(x - sx) < 10:
                            too_close = True
                            break
                    if not too_close:
                        self.source_locations.append((y, x))
                        break
                attempts += 1

            # Fallback: random high point
            if attempts >= 50:
                high_points = np.argwhere(elevation > np.percentile(elevation, 70))
                if len(high_points) > 0:
                    idx = np.random.randint(len(high_points))
                    self.source_locations.append(tuple(high_points[idx]))

    def compute_terrain_gradient(self, state: GridState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute terrain gradient (slope) in y and x directions.
        Returns (grad_y, grad_x) where positive values indicate uphill.
        """
        # Use central differences for smoother gradient
        grad_y = np.zeros_like(state.elevation)
        grad_x = np.zeros_like(state.elevation)

        # Y gradient (positive = terrain rises going down/south)
        grad_y[1:-1, :] = (state.elevation[2:, :] - state.elevation[:-2, :]) / 2.0
        grad_y[0, :] = state.elevation[1, :] - state.elevation[0, :]
        grad_y[-1, :] = state.elevation[-1, :] - state.elevation[-2, :]

        # X gradient (positive = terrain rises going right/east)
        grad_x[:, 1:-1] = (state.elevation[:, 2:] - state.elevation[:, :-2]) / 2.0
        grad_x[:, 0] = state.elevation[:, 1] - state.elevation[:, 0]
        grad_x[:, -1] = state.elevation[:, -1] - state.elevation[:, -2]

        return grad_y, grad_x

    def compute_water_surface_height(self, state: GridState) -> np.ndarray:
        """Compute H_i = elevation_i + h_i"""
        return state.elevation + state.water_depth

    def compute_conductance(self, state: GridState) -> np.ndarray:
        """
        Compute conductance matrix g_ij for each cell and its neighbors.
        Returns shape (grid_size, grid_size, 4) for 4 neighbors.
        Dam permeability reduces conductance.
        """
        g = np.zeros((self.size, self.size, 4))
        g0 = self.config.base_conductance

        for idx, (dy, dx) in enumerate(self.neighbor_offsets):
            # Shifted permeability arrays
            d_j = np.roll(np.roll(state.dam_permeability, dy, axis=0), dx, axis=1)

            # f(d_i, d_j) = 0.5 * (d_i + d_j) - dams reduce flow
            f = self.config.dam_permeability_effect * (state.dam_permeability + d_j)

            g[:, :, idx] = g0 * f

        return g

    def apply_water_sources(self, state: GridState) -> np.ndarray:
        """
        Generate water from source locations (springs).
        Water spreads in a small radius around each source.
        """
        source_water = np.zeros((self.size, self.size))

        for sy, sx in self.source_locations:
            # Add water in a circular area around the source
            radius = int(self.config.source_radius)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < self.size and 0 <= nx < self.size:
                        dist = np.sqrt(dy**2 + dx**2)
                        if dist <= self.config.source_radius:
                            # Water amount decreases with distance from center
                            amount = self.config.source_flow_rate * (1 - dist / (self.config.source_radius + 1))
                            source_water[ny, nx] += amount

        return source_water

    def detect_pools(self, state: GridState) -> np.ndarray:
        """
        Detect pooling areas - cells that are lower than all neighbors.
        Returns a mask where True indicates a pool/basin cell.
        """
        elevation = state.elevation
        is_pool = np.ones((self.size, self.size), dtype=bool)

        for dy, dx in self.neighbor_offsets:
            neighbor_elev = np.roll(np.roll(elevation, dy, axis=0), dx, axis=1)
            # Cell is a pool if it's lower than or equal to all neighbors
            is_pool &= (elevation <= neighbor_elev)

        return is_pool

    def update_rain_event(self) -> np.ndarray:
        """
        Handle rain events - periodic bursts of water over the map.
        """
        rain = np.zeros((self.size, self.size))

        # Check if new rain should start
        if not self.rain_active:
            if np.random.random() < self.config.rain_event_probability:
                self.rain_active = True
                self.rain_steps_remaining = self.config.rain_event_duration

        # If rain is active, add water
        if self.rain_active:
            # Rain with some spatial variation (not perfectly uniform)
            rain = np.random.uniform(
                0.5 * self.config.rain_event_intensity,
                1.5 * self.config.rain_event_intensity,
                (self.size, self.size)
            )
            self.rain_steps_remaining -= 1
            if self.rain_steps_remaining <= 0:
                self.rain_active = False

        return rain

    def update_water(self, state: GridState, rainfall: np.ndarray,
                     boundary_inflow: np.ndarray, dt: float) -> np.ndarray:
        """
        Update water depth using realistic flow dynamics.

        Flow rules:
        1. Water from sources (springs)
        2. Gradient-based flow (downhill bias)
        3. Pressure-based flow (high water to low)
        4. Dam effects (reduce flow, pool upstream)
        5. Evaporation and seepage losses
        """
        # Get terrain gradient
        grad_y, grad_x = self.compute_terrain_gradient(state)

        # Water surface height for pressure-based flow
        H = self.compute_water_surface_height(state)

        # Conductance (affected by dams)
        g = self.compute_conductance(state)

        # Initialize flux accumulator
        net_flux = np.zeros((self.size, self.size))

        # Detect pools for special handling
        pools = self.detect_pools(state)

        # Compute flow for each neighbor direction
        for idx, (dy, dx) in enumerate(self.neighbor_offsets):
            # Neighbor's water surface height
            H_j = np.roll(np.roll(H, dy, axis=0), dx, axis=1)

            # Neighbor's elevation
            elev_j = np.roll(np.roll(state.elevation, dy, axis=0), dx, axis=1)

            # 1. Pressure-based flux (water flows from high H to low H)
            pressure_flux = g[:, :, idx] * (H - H_j)

            # 2. Gravity-based flux (water flows downhill)
            # Terrain slope in this direction
            slope = elev_j - state.elevation  # Positive = neighbor is higher
            gravity_flux = self.config.flow_gravity * state.water_depth * np.maximum(-slope, 0)

            # 3. Momentum-based flux (water tends to keep flowing)
            # Match direction with momentum field
            if dy != 0:
                momentum_flux = self.config.flow_momentum * self.flow_momentum_y * (dy > 0).astype(float)
            else:
                momentum_flux = self.config.flow_momentum * self.flow_momentum_x * (dx > 0).astype(float)

            # Total flux to this neighbor
            total_flux = pressure_flux + gravity_flux + momentum_flux

            # Only allow outflow if we have water
            total_flux = np.minimum(total_flux, state.water_depth * 0.25)  # Max 25% per direction
            total_flux = np.maximum(total_flux, 0)  # No negative flux (handled by reverse direction)

            # Pool cells retain more water
            total_flux = np.where(pools, total_flux * 0.5, total_flux)

            # Net flux: outflow from this cell
            net_flux -= total_flux

            # Update momentum field based on flow
            if dy != 0:
                self.flow_momentum_y += 0.1 * total_flux * dy
            else:
                self.flow_momentum_x += 0.1 * total_flux * dx

        # Apply inflows from neighbors (reverse flux computation)
        for idx, (dy, dx) in enumerate(self.neighbor_offsets):
            # What neighbor cell (i-dy, i-dx) sends to us
            neighbor_water = np.roll(np.roll(state.water_depth, -dy, axis=0), -dx, axis=1)
            neighbor_H = np.roll(np.roll(H, -dy, axis=0), -dx, axis=1)
            neighbor_g = np.roll(np.roll(g[:, :, idx], -dy, axis=0), -dx, axis=1)

            # Flux from neighbor to us (their outflow is our inflow)
            inflow = neighbor_g * np.maximum(neighbor_H - H, 0)
            inflow = np.minimum(inflow, neighbor_water * 0.25)
            net_flux += inflow

        # Water sources
        source_water = self.apply_water_sources(state)

        # Rain events
        rain_water = self.update_rain_event()

        # Loss terms: evaporation and seepage
        loss = (self.config.evaporation_rate + self.config.seepage_rate) * state.water_depth

        # Update water depth
        new_water = state.water_depth + dt * (
            rainfall + boundary_inflow + source_water + rain_water + net_flux - loss
        )

        # Decay momentum over time
        self.flow_momentum_y *= 0.9
        self.flow_momentum_x *= 0.9

        # Apply smoothing to prevent numerical artifacts
        from scipy.ndimage import uniform_filter
        diffusion_strength = 0.1
        smoothed = uniform_filter(new_water, size=3, mode='reflect')
        new_water = (1 - diffusion_strength) * new_water + diffusion_strength * smoothed

        # Clamp to valid range
        new_water = np.clip(new_water, 0.0, self.config.max_water_depth)

        return new_water

    def get_upstream_downstream_delta(self, state: GridState, dam_y: int, dam_x: int) -> float:
        """
        Compute the water level difference between upstream and downstream of a dam.
        Useful for measuring dam effectiveness.
        """
        # Determine flow direction at this point (based on terrain gradient)
        grad_y, grad_x = self.compute_terrain_gradient(state)

        # Upstream is opposite to gradient direction
        upstream_dy = 1 if grad_y[dam_y, dam_x] > 0 else -1
        upstream_dx = 1 if grad_x[dam_y, dam_x] > 0 else -1

        # Get upstream water (3x3 area)
        upstream_water = 0.0
        upstream_count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = dam_y + upstream_dy + dy, dam_x + upstream_dx + dx
                if 0 <= ny < self.size and 0 <= nx < self.size:
                    upstream_water += state.water_depth[ny, nx]
                    upstream_count += 1

        # Get downstream water (3x3 area)
        downstream_water = 0.0
        downstream_count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = dam_y - upstream_dy + dy, dam_x - upstream_dx + dx
                if 0 <= ny < self.size and 0 <= nx < self.size:
                    downstream_water += state.water_depth[ny, nx]
                    downstream_count += 1

        if upstream_count > 0 and downstream_count > 0:
            return (upstream_water / upstream_count) - (downstream_water / downstream_count)
        return 0.0

    def compute_spatial_autocorrelation(self, state: GridState) -> float:
        """
        Compute spatial autocorrelation of water field.
        Higher values indicate more coherent patterns (rivers/ponds).
        """
        water = state.water_depth
        mean_water = np.mean(water)

        if mean_water < 0.01:
            return 0.0

        # Compute neighbor similarity (Moran's I simplified)
        total_similarity = 0.0
        count = 0

        for dy, dx in self.neighbor_offsets:
            neighbor = np.roll(np.roll(water, dy, axis=0), dx, axis=1)
            # Correlation between cell and neighbor
            diff_self = water - mean_water
            diff_neighbor = neighbor - mean_water
            total_similarity += np.sum(diff_self * diff_neighbor)
            count += water.size

        variance = np.var(water)
        if variance > 0:
            return total_similarity / (count * variance)
        return 0.0


class VegetationEngine:
    """
    Computes vegetation and soil dynamics.

    Vegetation regrows based on soil moisture.
    Soil moisture depends on water presence.
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.size = config.grid_size

    def update_vegetation(self, state: GridState, harvested: np.ndarray, dt: float) -> np.ndarray:
        """
        Update vegetation biomass.

        FIX: Vegetation now depends on:
        - Soil moisture (water availability)
        - Elevation (lower = better drainage/water access)
        - Proximity to water (riparian bonus)
        - Harvesting pressure (creates scarcity)
        """
        # === FIX: Carrying capacity varies by terrain ===
        # Lower elevation = closer to water = higher carrying capacity
        # Normalize elevation to [0, 1]
        elev_norm = (state.elevation - state.elevation.min()) / (
            state.elevation.max() - state.elevation.min() + 1e-8
        )
        # Carrying capacity: 0.4 to 1.0 based on elevation (lower = higher capacity)
        carrying_capacity = 0.4 + 0.6 * (1 - elev_norm)

        # === FIX: Water proximity bonus ===
        # Areas near water grow faster
        water_nearby = state.water_depth > 0.1
        from scipy.ndimage import maximum_filter
        riparian_zone = maximum_filter(water_nearby.astype(float), size=5)
        riparian_bonus = 0.5 * riparian_zone  # Up to 50% growth bonus near water

        # === Regrowth rate depends on moisture and terrain ===
        base_rate = self.config.vegetation_regrowth_rate
        moisture_bonus = self.config.moisture_effect_on_growth * state.soil_moisture
        regrowth_rate = base_rate * (1 + moisture_bonus + riparian_bonus)

        # Growth limited by distance from carrying capacity (not fixed max)
        max_veg = self.config.max_vegetation * carrying_capacity
        growth = regrowth_rate * (max_veg - state.vegetation)

        # === FIX: Water stress - vegetation dies if too much water ===
        # Flooded areas lose vegetation
        flood_damage = np.where(state.water_depth > 2.0, 0.1 * state.vegetation, 0.0)

        # === FIX: Drought stress - vegetation dies without moisture ===
        drought_damage = np.where(state.soil_moisture < 0.1, 0.05 * state.vegetation, 0.0)

        # Update vegetation
        new_vegetation = state.vegetation + dt * growth - harvested - flood_damage - drought_damage

        # Clamp to valid range
        new_vegetation = np.clip(new_vegetation, 0.0, self.config.max_vegetation)

        return new_vegetation

    def update_soil_moisture(self, state: GridState, dt: float) -> np.ndarray:
        """
        Update soil moisture based on water presence.

        When water is present, moisture trends toward 1.
        When dry, moisture decays toward 0.
        Moisture diffuses between cells.
        """
        # Target moisture based on water presence
        has_water = state.water_depth > 0.01
        target = np.where(has_water, 1.0, 0.0)

        # Also consider neighboring water
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_water = np.roll(np.roll(state.water_depth, dy, axis=0), dx, axis=1)
            target = np.maximum(target, 0.5 * (neighbor_water > 0.01).astype(float))

        # Relaxation toward target
        diff = target - state.soil_moisture
        new_moisture = state.soil_moisture + dt * self.config.moisture_diffusion_rate * diff

        # Decay when dry
        dry_mask = target < 0.5
        new_moisture = np.where(
            dry_mask,
            new_moisture * (1 - dt * self.config.moisture_decay_rate),
            new_moisture
        )

        # Clamp
        return np.clip(new_moisture, 0.0, 1.0)


class StructurePhysicsEngine:
    """
    Handles dam integrity, decay, overflow damage, and failure mechanics.

    Features:
    - Dams decay over time (need maintenance)
    - Overflow causes damage (water pressure)
    - Dams can break when integrity < threshold
    - Broken dams cause flood events
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.size = config.grid_size

        # Track dam lifetimes for metrics
        self.dam_creation_step: Dict[Tuple[int, int], int] = {}
        self.dam_failure_events: List[Tuple[int, int, int]] = []  # (y, x, step)
        self.flood_events_prevented: int = 0
        self.flood_events_caused: int = 0

    def update_dam_integrity(self, state: GridState, current_step: int, dt: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Update dam integrity based on:
        - Natural decay over time
        - Water pressure (overflow) damage
        - Failure when integrity drops below threshold

        Returns:
            (new_integrity, list of broken dam locations)
        """
        # Identify where dams exist (permeability < 1.0)
        dam_mask = state.dam_permeability < 1.0

        new_integrity = state.dam_integrity.copy()
        broken_dams = []

        # Only update where dams exist
        if not np.any(dam_mask):
            return new_integrity, broken_dams

        # 1. Natural decay
        decay = self.config.dam_decay_rate * dt
        new_integrity = np.where(dam_mask, new_integrity - decay, new_integrity)

        # 2. Overflow damage (water pressure)
        overflow_mask = (state.water_depth > self.config.dam_overflow_threshold) & dam_mask
        overflow_damage = self.config.dam_overflow_damage * dt
        new_integrity = np.where(overflow_mask, new_integrity - overflow_damage, new_integrity)

        # 3. Check for failures
        failure_mask = (new_integrity < self.config.dam_failure_threshold) & dam_mask

        if np.any(failure_mask):
            # Record failure locations
            failure_coords = np.argwhere(failure_mask)
            for coord in failure_coords:
                y, x = coord[0], coord[1]
                broken_dams.append((y, x))
                self.dam_failure_events.append((y, x, current_step))
                self.flood_events_caused += 1

                # Calculate dam lifetime
                if (y, x) in self.dam_creation_step:
                    lifetime = current_step - self.dam_creation_step[y, x]
                    del self.dam_creation_step[(y, x)]

        # Reset failed dams (permeability returns to 1.0)
        # This is handled separately in step() to trigger flood

        # Clamp integrity
        new_integrity = np.clip(new_integrity, 0.0, 1.0)

        # Reset integrity to 1.0 where there's no dam (permeability == 1.0)
        new_integrity = np.where(~dam_mask, 1.0, new_integrity)

        return new_integrity, broken_dams

    def handle_dam_breaks(self, state: GridState, broken_dams: List[Tuple[int, int]]) -> np.ndarray:
        """
        Handle dam failures - reset permeability and cause flood surge.

        Returns:
            Updated water_depth after flood surge
        """
        new_water = state.water_depth.copy()

        for y, x in broken_dams:
            # Reset dam permeability (dam is gone)
            state.dam_permeability[y, x] = 1.0
            state.dam_integrity[y, x] = 1.0

            # Flood surge - water rushes through
            # Add extra water downstream (in the direction of flow)
            surge = state.water_depth[y, x] * self.config.dam_break_flood_multiplier

            # Spread surge to neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.size and 0 <= nx < self.size:
                    new_water[ny, nx] += surge * 0.25

        return new_water

    def repair_dam(self, state: GridState, y: int, x: int) -> float:
        """
        Repair a dam at the given location.

        Returns:
            Amount of integrity restored
        """
        if state.dam_permeability[y, x] >= 1.0:
            # No dam here
            return 0.0

        old_integrity = state.dam_integrity[y, x]
        new_integrity = min(1.0, old_integrity + self.config.dam_repair_amount)
        state.dam_integrity[y, x] = new_integrity

        return new_integrity - old_integrity

    def record_dam_creation(self, y: int, x: int, current_step: int):
        """Record when a dam is created for lifetime tracking"""
        if (y, x) not in self.dam_creation_step:
            self.dam_creation_step[(y, x)] = current_step

    def get_dam_lifetime_stats(self) -> Dict[str, float]:
        """Get statistics about dam lifetimes"""
        if not self.dam_failure_events:
            return {"avg_lifetime": 0.0, "n_failures": 0}

        lifetimes = []
        for y, x, failure_step in self.dam_failure_events:
            # Approximate - we don't track creation time perfectly
            lifetimes.append(failure_step)  # Upper bound

        return {
            "avg_lifetime": np.mean(lifetimes) if lifetimes else 0.0,
            "n_failures": len(self.dam_failure_events),
            "flood_events_caused": self.flood_events_caused,
        }


@dataclass
class AgentState:
    """Internal state of a single beaver agent"""
    id: int
    position: Tuple[int, int]  # (y, x) grid coordinates
    energy: float
    satiety: float
    wetness: float
    role: AgentRole
    thresholds: np.ndarray  # Task response thresholds θ_{k,m}
    current_project: Optional[int] = None  # Project ID or None
    carrying_wood: int = 0
    alive: bool = True

    # Movement tracking
    previous_position: Optional[Tuple[int, int]] = None

    # === REWARD FIX: Personal tracking (fix free-rider problem) ===
    structures_built_by_me: int = 0  # Count of structures this agent built
    cells_visited: Optional[set] = None  # Set of (y, x) cells visited for exploration reward

    def get_visit_count(self, position: Tuple[int, int]) -> int:
        """Get how many times this cell was visited (for diminishing exploration reward)"""
        if self.cells_visited is None:
            self.cells_visited = set()
        return 1 if position in self.cells_visited else 0

    def record_visit(self, position: Tuple[int, int]):
        """Record visiting a cell"""
        if self.cells_visited is None:
            self.cells_visited = set()
        self.cells_visited.add(position)

    # === PHASE 2: Information Energy ===
    # Info energy is a first-class resource that constrains information actions
    info_energy: float = 100.0  # Current info energy level

    def can_afford_info(self, cost: float, min_threshold: float = 1.0) -> bool:
        """Check if agent can afford an info action"""
        return self.info_energy >= max(cost, min_threshold)

    def spend_info(self, cost: float) -> bool:
        """
        Spend info energy. Returns True if successful, False if insufficient.

        Silent failure pattern: if can't afford, action simply doesn't happen.
        """
        if self.info_energy >= cost:
            self.info_energy -= cost
            return True
        return False

    def recover_info(self, amount: float, max_energy: float = 150.0):
        """Recover info energy (capped at max)"""
        self.info_energy = min(max_energy, self.info_energy + amount)


class MycoBeaverEnv(gym.Env):
    """
    MycoBeaver Simulation Environment

    Gym-compatible environment for multi-agent beaver simulation
    with hydrology, vegetation, pheromones, projects, and physarum networks.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: SimulationConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # Initialize engines
        self.hydrology_engine = HydrologyEngine(config.grid)
        self.vegetation_engine = VegetationEngine(config.grid)
        self.structure_physics = StructurePhysicsEngine(config.grid)

        # Placeholders for subsystems (initialized in reset)
        self.pheromone_field = None
        self.project_manager = None
        self.physarum_network = None
        self.overmind = None
        self.semantic_system = None  # PHASE 3: Colony semantic system

        # Grid state
        self.grid_state: Optional[GridState] = None

        # Agents
        self.agents: List[AgentState] = []

        # Simulation state
        self.current_step = 0
        self.episode_reward = 0.0

        # === PHASE 3: Time-scale separation counters ===
        # These track when each subsystem was last updated
        self._last_physarum_update = 0
        self._last_overmind_update = 0
        self._last_semantic_consolidation = 0
        self._last_project_recruitment_update = 0

        # Update counts for debugging/logging
        self._physarum_update_count = 0
        self._overmind_update_count = 0
        self._semantic_consolidation_count = 0

        # Random state
        self.np_random = None

        # Define spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """Define observation and action spaces"""
        grid_size = self.config.grid.grid_size
        n_agents = self.config.n_beavers
        view_radius = self.config.policy.local_view_radius
        view_size = 2 * view_radius + 1

        # Single agent observation space
        # Local grid view: (n_channels, view_size, view_size)
        n_channels = self.config.policy.n_local_channels
        local_obs_shape = (n_channels, view_size, view_size)

        # Global features: recruitment signals, overmind wisdom, etc.
        n_global = self.config.policy.n_global_features

        # Internal state: energy, satiety, wetness, role, thresholds
        n_internal = self.config.policy.n_internal_features

        # Multi-agent observation space
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Dict({
                "local_grid": spaces.Box(low=0, high=1, shape=local_obs_shape, dtype=np.float32),
                "global_features": spaces.Box(low=-np.inf, high=np.inf, shape=(n_global,), dtype=np.float32),
                "internal_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n_internal,), dtype=np.float32),
            }) for i in range(n_agents)
        })

        # Action space: discrete actions per agent
        n_actions = self.config.policy.n_actions
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(n_actions) for i in range(n_agents)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        # Initialize grid state
        self._initialize_grid()

        # Initialize agents
        self._initialize_agents()

        # Initialize subsystems
        self._initialize_subsystems()

        # Reset counters
        self.current_step = 0
        self.episode_reward = 0.0

        # === PHASE 3: Reset time-scale separation counters ===
        self._last_physarum_update = 0
        self._last_overmind_update = 0
        self._last_semantic_consolidation = 0
        self._last_project_recruitment_update = 0
        self._physarum_update_count = 0
        self._overmind_update_count = 0
        self._semantic_consolidation_count = 0

        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def _initialize_grid(self):
        """Initialize grid state arrays with improved procedural generation"""
        size = self.config.grid.grid_size

        # === IMPROVED TERRAIN GENERATION ===
        # Use multi-octave noise for natural-looking terrain
        elevation = self._generate_terrain(size)

        # If stream location specified, create lower channel
        if self.config.stream_location:
            x1, y1, x2, y2 = self.config.stream_location
            for i in range(min(y1, y2), max(y1, y2) + 1):
                for j in range(min(x1, x2), max(x1, x2) + 1):
                    elevation[i, j] -= 0.2

        # === IMPROVED WATER GENERATION ===
        # Water collects in valleys (low elevation areas)
        water_depth = self._generate_water(size, elevation)

        # === IMPROVED VEGETATION GENERATION ===
        # Vegetation in clusters (creates exploration pressure)
        vegetation = self._generate_vegetation_clusters(size, water_depth)

        # Initialize soil moisture
        soil_moisture = 0.5 * np.ones((size, size))
        soil_moisture = np.where(water_depth > 0, 0.9, soil_moisture)

        # Initialize dams (no dams initially)
        dam_permeability = np.ones((size, size))
        dam_integrity = np.ones((size, size))  # Perfect integrity where there's no dam

        # Initialize lodges
        lodge_map = np.zeros((size, size), dtype=bool)
        if self.config.lodge_location:
            ly, lx = self.config.lodge_location
            lodge_map[ly, lx] = True
        else:
            # Default lodge in center
            center = size // 2
            lodge_map[center, center] = True

        # Agent positions
        agent_positions = np.zeros((size, size), dtype=np.int32)

        self.grid_state = GridState(
            elevation=elevation,
            water_depth=water_depth,
            vegetation=vegetation,
            soil_moisture=soil_moisture,
            dam_permeability=dam_permeability,
            dam_integrity=dam_integrity,
            lodge_map=lodge_map,
            agent_positions=agent_positions
        )

        # Initialize water sources based on terrain
        self.hydrology_engine.initialize_sources(elevation)

    def _generate_terrain(self, size: int) -> np.ndarray:
        """Generate natural-looking terrain using multi-octave noise"""
        # Create base gradient (hills on one side, valley on other)
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)

        # Multi-octave noise approximation (without external library)
        elevation = np.zeros((size, size))

        # Octave 1: Large features
        scale1 = 4
        noise1 = self._smooth_noise(size, scale1)
        elevation += 0.5 * noise1

        # Octave 2: Medium features
        scale2 = 8
        noise2 = self._smooth_noise(size, scale2)
        elevation += 0.25 * noise2

        # Octave 3: Small features
        scale3 = 16
        noise3 = self._smooth_noise(size, scale3)
        elevation += 0.125 * noise3

        # Add gradient to create consistent slope (water flows downhill)
        gradient = 0.3 * (1.0 - xx) + 0.1 * yy
        elevation += gradient

        # Normalize to [0, 1]
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)

        return elevation

    def _smooth_noise(self, size: int, scale: int) -> np.ndarray:
        """Generate smooth noise at given scale"""
        # Generate random values at grid points
        grid_size = max(2, size // scale + 1)
        grid = self.np_random.random((grid_size, grid_size))

        # Interpolate to full size using bilinear interpolation
        from scipy.ndimage import zoom
        smooth = zoom(grid, size / grid_size, order=1)

        # Ensure correct size
        if smooth.shape[0] != size or smooth.shape[1] != size:
            smooth = smooth[:size, :size]
            if smooth.shape[0] < size:
                smooth = np.pad(smooth, ((0, size - smooth.shape[0]), (0, 0)), mode='edge')
            if smooth.shape[1] < size:
                smooth = np.pad(smooth, ((0, 0), (0, size - smooth.shape[1])), mode='edge')

        return smooth

    def _generate_water(self, size: int, elevation: np.ndarray) -> np.ndarray:
        """Generate water that collects in valleys"""
        # Water accumulates in low elevation areas
        water_threshold = np.percentile(elevation, 30)  # Bottom 30% elevation gets water

        # Base water depth inversely proportional to elevation
        water_depth = np.zeros((size, size))

        # Add water in valleys
        valley_mask = elevation < water_threshold
        water_depth[valley_mask] = (water_threshold - elevation[valley_mask]) * 10.0

        # Add some water sources (springs) at random high points
        n_springs = 3
        for _ in range(n_springs):
            sy, sx = self.np_random.integers(0, size, 2)
            # Create small pond around spring
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < size and 0 <= nx < size:
                        dist = np.sqrt(dy**2 + dx**2)
                        water_depth[ny, nx] += max(0, 2.0 - dist)

        # Smooth water slightly for natural look
        from scipy.ndimage import gaussian_filter
        water_depth = gaussian_filter(water_depth, sigma=1.0)

        # Clamp water depth
        water_depth = np.clip(water_depth, 0.0, self.config.initial_water_level * 2)

        return water_depth

    def _generate_vegetation_clusters(self, size: int, water_depth: np.ndarray) -> np.ndarray:
        """Generate vegetation in clusters (creates exploration pressure)"""
        vegetation = np.zeros((size, size))

        # Create 5-8 vegetation clusters
        n_clusters = self.np_random.integers(5, 9)

        for _ in range(n_clusters):
            # Random cluster center
            cy, cx = self.np_random.integers(5, size - 5, 2)
            cluster_radius = self.np_random.integers(5, 12)
            cluster_density = self.np_random.uniform(0.6, 1.0)

            # Add vegetation in cluster (Gaussian falloff)
            for dy in range(-cluster_radius, cluster_radius + 1):
                for dx in range(-cluster_radius, cluster_radius + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < size and 0 <= nx < size:
                        dist = np.sqrt(dy**2 + dx**2)
                        if dist <= cluster_radius:
                            # Gaussian falloff
                            amount = cluster_density * np.exp(-dist**2 / (2 * (cluster_radius/2)**2))
                            vegetation[ny, nx] = max(vegetation[ny, nx], amount)

        # Add base sparse vegetation everywhere
        base_veg = self.np_random.uniform(0.05, 0.15, (size, size))
        vegetation = np.maximum(vegetation, base_veg)

        # Reduce vegetation in water
        vegetation = np.where(water_depth > 0.5, 0.1 * vegetation, vegetation)

        # Normalize to [0, 1]
        vegetation = np.clip(vegetation, 0.0, 1.0)

        return vegetation

    def _initialize_agents(self):
        """Initialize beaver agents"""
        self.agents = []
        size = self.config.grid.grid_size
        config = self.config.agent

        # Find lodge position for initial spawning
        lodge_positions = np.argwhere(self.grid_state.lodge_map)
        if len(lodge_positions) > 0:
            spawn_center = lodge_positions[0]
        else:
            spawn_center = np.array([size // 2, size // 2])

        for i in range(self.config.n_beavers):
            # Spawn near lodge with some spread
            offset = self.np_random.integers(-3, 4, size=2)
            pos = np.clip(spawn_center + offset, 0, size - 1)

            # Initialize task thresholds (personality)
            thresholds = self.np_random.normal(
                config.threshold_mean,
                config.threshold_std,
                size=config.n_task_types
            )
            thresholds = np.clip(thresholds, 0.1, 0.9)

            # Assign role with balanced distribution
            # Distribution: 15% scout, 25% worker, 15% guardian, 20% builder, 15% hauler, 10% maintainer
            role_idx = i % 6
            if role_idx == 0:
                role = AgentRole.SCOUT
            elif role_idx == 1:
                role = AgentRole.WORKER
            elif role_idx == 2:
                role = AgentRole.BUILDER
            elif role_idx == 3:
                role = AgentRole.HAULER
            elif role_idx == 4:
                role = AgentRole.MAINTAINER
            else:
                role = AgentRole.GUARDIAN

            agent = AgentState(
                id=i,
                position=(int(pos[0]), int(pos[1])),
                energy=config.initial_energy,
                satiety=config.initial_satiety,
                wetness=config.initial_wetness,
                role=role,
                thresholds=thresholds,
            )
            self.agents.append(agent)

            # Update agent positions grid
            self.grid_state.agent_positions[agent.position] += 1

    def _initialize_subsystems(self):
        """Initialize pheromone field, project manager, physarum network, overmind, semantic system"""
        from .pheromone import PheromoneField
        from .projects import ProjectManager
        from .physarum import PhysarumNetwork
        from .overmind import Overmind
        from .semantic import ColonySemanticSystem

        self.pheromone_field = PheromoneField(self.config.pheromone, self.config.grid.grid_size)
        self.project_manager = ProjectManager(self.config.project, self.config.grid.grid_size)
        self.physarum_network = PhysarumNetwork(self.config.physarum, self.config.grid.grid_size)
        self.overmind = Overmind(self.config.overmind, self.config)

        # PHASE 3: Initialize colony semantic system (slow time-scale)
        self.semantic_system = ColonySemanticSystem(
            self.config.semantic,
            self.config.n_beavers,
            self.config.info_costs,
        )

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Execute one environment step.

        Order of operations:
        1. Overmind phase - adjust meta-parameters
        2. Agent decision phase - already done (actions provided)
        3. Apply actions
        4. Environment dynamics update
        5. Subsystem updates (pheromone, project, physarum)
        6. Compute rewards
        7. Check termination
        """
        self.current_step += 1
        dt = self.config.grid.dt
        ts_config = self.config.time_scales

        # 1. Overmind phase (PHASE 3: respects time-scale interval)
        # Overmind operates on a slow time-scale to provide stable meta-parameters
        if self.config.training.use_overmind:
            if self.current_step - self._last_overmind_update >= ts_config.overmind_update_interval:
                overmind_obs = self._get_overmind_observation()
                self.overmind.update(overmind_obs, self)
                self._last_overmind_update = self.current_step
                self._overmind_update_count += 1

        # 2-3. Apply agent actions
        agent_rewards = self._apply_actions(actions)

        # 4. Environment dynamics
        self._update_environment_dynamics(dt)

        # 5. Subsystem updates
        self._update_subsystems(dt)

        # 6. Compute rewards
        global_reward = self._compute_global_reward()
        rewards = self._combine_rewards(agent_rewards, global_reward)
        self.episode_reward += sum(rewards.values())

        # 7. Check termination
        terminated = self._check_terminated()
        truncated = self.current_step >= self.config.training.max_steps_per_episode

        # === PHASE 3: Semantic consolidation at episode end (SLOW time-scale) ===
        # This is the slowest time-scale operation - only happens when episode ends
        if (terminated or truncated) and self.semantic_system is not None:
            ts_config = self.config.time_scales
            if ts_config.semantic_consolidate_on_episode_end:
                consolidation_stats = self.semantic_system.consolidate(
                    clear_ants=ts_config.clear_semantic_ants_on_episode_end
                )
                self._semantic_consolidation_count += 1
                self._last_semantic_consolidation = self.current_step

        # Get observations and info
        observations = self._get_observations()
        info = self._get_info()
        info["rewards"] = rewards

        return observations, rewards, terminated, truncated, info

    def _apply_actions(self, actions: Dict[str, int]) -> Dict[str, float]:
        """Apply agent actions and return individual rewards"""
        rewards = {}
        movements = []  # For pheromone updates
        harvested = np.zeros((self.config.grid.grid_size, self.config.grid.grid_size))

        for agent in self.agents:
            if not agent.alive:
                rewards[f"agent_{agent.id}"] = 0.0
                continue

            action_key = f"agent_{agent.id}"
            action_idx = actions.get(action_key, ActionType.STAY.value)

            # Store previous position for pheromone tracking
            agent.previous_position = agent.position

            # Execute action
            reward = self._execute_agent_action(agent, ActionType(action_idx), harvested)
            rewards[action_key] = reward

            # Track movement for pheromones
            if agent.previous_position != agent.position:
                movements.append((agent.previous_position, agent.position))

            # Update agent internal state
            self._update_agent_state(agent)

        # Update pheromones based on movements
        if self.config.training.use_pheromones:
            self.pheromone_field.update(movements)

        # Apply harvesting to vegetation
        new_vegetation = self.vegetation_engine.update_vegetation(
            self.grid_state, harvested, self.config.grid.dt
        )
        self.grid_state.vegetation = new_vegetation

        return rewards

    def _execute_agent_action(self, agent: AgentState, action: ActionType,
                               harvested: np.ndarray) -> float:
        """Execute a single agent's action"""
        reward = 0.0
        y, x = agent.position
        size = self.config.grid.grid_size
        config = self.config.agent

        # Movement actions
        move_map = {
            ActionType.MOVE_NORTH: (-1, 0),
            ActionType.MOVE_SOUTH: (1, 0),
            ActionType.MOVE_EAST: (0, 1),
            ActionType.MOVE_WEST: (0, -1),
            ActionType.MOVE_NE: (-1, 1),
            ActionType.MOVE_NW: (-1, -1),
            ActionType.MOVE_SE: (1, 1),
            ActionType.MOVE_SW: (1, -1),
        }

        if action in move_map:
            dy, dx = move_map[action]
            new_y = np.clip(y + dy, 0, size - 1)
            new_x = np.clip(x + dx, 0, size - 1)

            # Update position
            self.grid_state.agent_positions[y, x] -= 1
            self.grid_state.agent_positions[new_y, new_x] += 1
            agent.position = (new_y, new_x)

            # REWARD FIX: Exploration reward with diminishing returns
            new_pos = (new_y, new_x)
            visit_count = agent.get_visit_count(new_pos)
            base_explore_reward = self.config.reward.exploration_reward
            if visit_count == 0:
                # First visit - full exploration reward
                explore_reward = base_explore_reward
            else:
                # Revisit - diminished reward
                explore_reward = base_explore_reward * (
                    self.config.reward.exploration_decay ** visit_count
                )

            # ROLE BONUS: Scout gets exploration multiplier
            if agent.role == AgentRole.SCOUT:
                explore_reward *= self.config.reward.scout_exploration_multiplier
                # Extra coverage bonus for scouts visiting new cells
                if visit_count == 0:
                    explore_reward += self.config.reward.scout_coverage_bonus

            reward += explore_reward
            agent.record_visit(new_pos)

            # REWARD FIX: Carrying wood proximity bonus
            if agent.carrying_wood > 0:
                # Check if near structures (build sites)
                structure_mask = self.grid_state.dam_permeability < 0.9
                if np.any(structure_mask):
                    # Find nearest structure
                    struct_coords = np.argwhere(structure_mask)
                    distances = np.sqrt(
                        (struct_coords[:, 0] - new_y)**2 +
                        (struct_coords[:, 1] - new_x)**2
                    )
                    min_dist = distances.min()
                    if min_dist < self.config.reward.carry_wood_distance_threshold:
                        reward += self.config.reward.carry_wood_proximity_bonus

            # Energy cost
            agent.energy -= config.move_energy_cost

        elif action == ActionType.STAY:
            # Minimal energy cost
            # ROLE BONUS: Guardian gets bonus for staying near lodge
            if agent.role == AgentRole.GUARDIAN:
                # Find nearest lodge
                lodge_coords = np.argwhere(self.grid_state.lodge_map)
                if len(lodge_coords) > 0:
                    distances = np.sqrt((lodge_coords[:, 0] - y)**2 + (lodge_coords[:, 1] - x)**2)
                    min_dist = distances.min()
                    if min_dist < self.config.reward.guardian_protection_radius:
                        reward += self.config.reward.guardian_stay_multiplier

        elif action == ActionType.REST:
            # Recover energy, especially in lodge
            if self.grid_state.lodge_map[y, x]:
                agent.energy += config.rest_energy_gain * 2
                agent.wetness = max(0, agent.wetness - config.lodge_drying_rate)
            else:
                agent.energy += config.rest_energy_gain

        elif action == ActionType.FORAGE:
            # Harvest vegetation
            veg_available = self.grid_state.vegetation[y, x]
            min_veg = config.forage_min_vegetation
            if veg_available > min_veg:
                harvest_amount = min(0.2, veg_available)
                harvested[y, x] += harvest_amount
                agent.satiety = min(1.0, agent.satiety + config.forage_satiety_gain)
                agent.carrying_wood = min(config.max_carry_wood, agent.carrying_wood + 1)
                reward += self.config.reward.forage_reward

                # SHAPING: Extra reward for acquiring wood when near water (intent to build)
                if self.grid_state.water_depth[y, x] > 0.1:
                    reward += self.config.reward.forage_reward * 0.5  # Water proximity bonus
            agent.energy -= config.forage_energy_cost

        elif action == ActionType.BUILD_DAM:
            # Build/reinforce dam at current location
            # REWARD FIX: Relaxed project requirement - agents can build freely
            if agent.carrying_wood > 0:
                old_permeability = self.grid_state.dam_permeability[y, x]
                is_new_dam = old_permeability >= 1.0  # No dam existed before
                new_permeability = max(0.0, old_permeability - self.config.grid.dam_build_amount)
                self.grid_state.dam_permeability[y, x] = new_permeability
                agent.carrying_wood -= 1

                # Track dam creation for lifetime statistics
                if is_new_dam:
                    self.structure_physics.record_dam_creation(y, x, self.current_step)
                    # Initialize dam integrity
                    self.grid_state.dam_integrity[y, x] = self.config.grid.dam_initial_integrity

                # REWARD FIX: Track personal contribution (fixes free-rider)
                agent.structures_built_by_me += 1

                # Update project progress if project system enabled
                if self.config.training.use_projects and agent.current_project is not None:
                    self.project_manager.add_progress(agent.current_project, 1)

                # Base build reward
                build_reward = self.config.reward.build_action_reward

                # ROLE BONUS: Builder gets multiplier
                if agent.role == AgentRole.BUILDER:
                    build_reward *= self.config.reward.builder_build_multiplier

                reward += build_reward

                # REWARD FIX: Personal structure bonus
                reward += self.config.reward.personal_structure_bonus

                # SHAPING: Water proximity bonus (dams near water are more useful)
                water_nearby = self.grid_state.water_depth[max(0,y-2):min(self.config.grid.grid_size,y+3),
                                                           max(0,x-2):min(self.config.grid.grid_size,x+3)]
                strategic_placement = np.any(water_nearby > 0.1)
                if strategic_placement:
                    reward += self.config.reward.build_action_reward * 0.5  # Near-water bonus
                    # ROLE BONUS: Builder gets extra for strategic placement
                    if agent.role == AgentRole.BUILDER:
                        reward += self.config.reward.builder_placement_bonus

                # SHAPING: Early infrastructure bonus (encourage faster emergence)
                total_structures = np.sum(self.grid_state.dam_permeability < 0.9)
                if total_structures < 10:  # Extra bonus for first 10 structures
                    early_bonus = self.config.reward.build_action_reward * (1.0 - total_structures / 10)
                    reward += early_bonus
            agent.energy -= config.build_energy_cost

        elif action == ActionType.BUILD_LODGE:
            # Build lodge (similar to dam but marks lodge location)
            if agent.carrying_wood > 0 and not self.grid_state.lodge_map[y, x]:
                # Need enough wood and low water
                if self.grid_state.water_depth[y, x] < 0.5:
                    agent.carrying_wood -= 1
                    # Could implement progressive lodge building
                    reward += self.config.reward.build_action_reward * 0.5
            agent.energy -= config.build_energy_cost

        elif action == ActionType.CARRY_RESOURCE:
            # Pick up resource (if available and not already carrying)
            if agent.carrying_wood < config.max_carry_wood:
                veg = self.grid_state.vegetation[y, x]
                min_veg = config.carry_min_vegetation
                if veg > min_veg:
                    harvested[y, x] += 0.1
                    agent.carrying_wood += 1
                    # REWARD FIX: Reward for picking up wood
                    carry_reward = self.config.reward.carry_wood_reward

                    # ROLE BONUS: Hauler gets multiplier for picking up resources
                    if agent.role == AgentRole.HAULER:
                        carry_reward *= self.config.reward.hauler_carry_multiplier

                    reward += carry_reward

                    # SHAPING: Extra reward for picking up wood when structures exist (intent to contribute)
                    if np.any(self.grid_state.dam_permeability < 0.9):
                        reward += self.config.reward.carry_wood_reward * 0.5  # Structure existence bonus
                        # ROLE BONUS: Hauler gets delivery bonus when structures exist
                        if agent.role == AgentRole.HAULER:
                            reward += self.config.reward.hauler_delivery_bonus

        elif action == ActionType.DROP_RESOURCE:
            # Drop carried resource
            if agent.carrying_wood > 0:
                agent.carrying_wood -= 1
                self.grid_state.vegetation[y, x] = min(
                    self.config.grid.max_vegetation,
                    self.grid_state.vegetation[y, x] + 0.1
                )

        elif action == ActionType.ADVERTISE_PROJECT:
            # Scout advertising a project (waggle dance)
            if self.config.training.use_projects and agent.role == AgentRole.SCOUT:
                if agent.current_project is not None:
                    self.project_manager.advertise(agent.current_project, agent.id)

        elif action == ActionType.REPAIR_DAM:
            # Repair an existing dam at current location
            if self.grid_state.dam_permeability[y, x] < 1.0:  # Dam exists here
                if agent.carrying_wood > 0:
                    agent.carrying_wood -= 1
                    old_integrity = self.grid_state.dam_integrity[y, x]

                    # Repair the dam
                    integrity_restored = self.structure_physics.repair_dam(self.grid_state, y, x)

                    if integrity_restored > 0:
                        # Base repair reward
                        repair_reward = self.config.reward.repair_action_reward

                        # ROLE BONUS: Maintainer gets multiplier
                        if agent.role == AgentRole.MAINTAINER:
                            repair_reward *= self.config.reward.maintainer_repair_multiplier

                        reward += repair_reward

                        # Critical repair bonus (more reward for saving dying dams)
                        if old_integrity < self.config.reward.repair_critical_threshold:
                            reward += self.config.reward.repair_critical_bonus
                            # ROLE BONUS: Maintainer gets prevention bonus for critical repairs
                            if agent.role == AgentRole.MAINTAINER:
                                reward += self.config.reward.maintainer_prevention_bonus

                        # Track repair for agent
                        agent.structures_built_by_me += 1  # Repairs count as contribution
            agent.energy -= config.build_energy_cost  # Same energy cost as building

        return reward

    def _update_agent_state(self, agent: AgentState):
        """Update agent internal state (energy, satiety, wetness, info_energy, death)"""
        config = self.config.agent
        y, x = agent.position
        dt = self.config.grid.dt

        # Satiety decay
        agent.satiety -= config.satiety_decay_rate * dt
        agent.satiety = max(0.0, agent.satiety)

        # Energy decay (higher when hungry)
        hunger_multiplier = 1.0
        if agent.satiety < config.min_satiety_before_damage:
            hunger_multiplier = config.starvation_energy_multiplier

        # Wetness effect
        wetness_cost = config.wetness_energy_penalty * agent.wetness

        # Base energy decay
        agent.energy -= (config.base_energy_decay * hunger_multiplier + wetness_cost) * dt

        # Update wetness based on water presence
        water_here = self.grid_state.water_depth[y, x]
        if water_here > 0.1:
            agent.wetness = min(1.0, agent.wetness + config.water_wetness_rate * dt)
        else:
            agent.wetness = max(0.0, agent.wetness - config.ambient_drying_rate * dt)

        # === PHASE 2: Info Energy Recovery ===
        # Base passive recovery
        agent.recover_info(config.info_recovery_base * dt, config.max_info_energy)

        # Bonus recovery if agent is at a lodge (resting = clearer thinking)
        if self.grid_state.lodge_map[y, x]:
            agent.recover_info(config.info_recovery_base * 2 * dt, config.max_info_energy)

        # Bonus recovery if agent is in coordination with others (same cell)
        if self.grid_state.agent_positions[y, x] > 1:
            agent.recover_info(config.info_recovery_coordination * 0.1 * dt, config.max_info_energy)

        # Check death conditions
        if agent.energy <= config.min_energy:
            agent.alive = False
            self.grid_state.agent_positions[y, x] -= 1

    def _update_environment_dynamics(self, dt: float):
        """Update water and soil"""
        # Generate rainfall
        rainfall = self.np_random.normal(
            self.config.grid.rainfall_rate,
            self.config.grid.rainfall_variance,
            (self.config.grid.grid_size, self.config.grid.grid_size)
        )
        rainfall = np.maximum(rainfall, 0)

        # Boundary inflow (from edges representing external water sources)
        boundary_inflow = np.zeros_like(rainfall)
        boundary_inflow[0, :] = 0.05  # Inflow from top

        # Update water
        new_water = self.hydrology_engine.update_water(
            self.grid_state, rainfall, boundary_inflow, dt
        )
        # Clamp water depth to prevent numerical overflow (max 10 meters)
        self.grid_state.water_depth = np.clip(new_water, 0.0, 10.0)

        # === STRUCTURE PHYSICS: Dam integrity, decay, and failure ===
        new_integrity, broken_dams = self.structure_physics.update_dam_integrity(
            self.grid_state, self.current_step, dt
        )
        self.grid_state.dam_integrity = new_integrity

        # Handle dam failures (flood surge)
        if broken_dams:
            flood_water = self.structure_physics.handle_dam_breaks(self.grid_state, broken_dams)
            self.grid_state.water_depth = np.clip(flood_water, 0.0, 10.0)

        # Update soil moisture
        new_moisture = self.vegetation_engine.update_soil_moisture(self.grid_state, dt)
        self.grid_state.soil_moisture = new_moisture

    def _update_subsystems(self, dt: float):
        """
        Update pheromones, projects, physarum with PHASE 3 time-scale separation.

        Update hierarchy:
        - Pheromone evaporation: Every step (fast, natural decay)
        - Project recruitment: Every N steps (medium)
        - Physarum network: Every M steps (medium, 10-20)
        - Semantic consolidation: Handled separately at episode end
        """
        ts_config = self.config.time_scales

        # === FAST: Pheromone evaporation (every step) ===
        # Natural decay happens every step - this is physically realistic
        if self.config.training.use_pheromones:
            self.pheromone_field.evaporate(dt)

        # === MEDIUM: Project recruitment signals ===
        # Recruitment signals propagate at medium frequency
        if self.config.training.use_projects:
            if self.current_step - self._last_project_recruitment_update >= ts_config.project_recruitment_interval:
                advertising_scouts = [a.id for a in self.agents if a.role == AgentRole.SCOUT
                                     and a.current_project is not None]
                self.project_manager.update_recruitment(advertising_scouts, dt)
                self._last_project_recruitment_update = self.current_step

        # === MEDIUM: Physarum network (every 10-20 steps) ===
        # Physarum operates at a slower timescale than individual agent actions
        # This prevents the network topology from thrashing and allows
        # stable path formation before agents respond
        if self.config.training.use_physarum:
            if self.current_step - self._last_physarum_update >= ts_config.physarum_update_interval:
                # Determine sources and sinks
                sources = self._find_resource_sources()
                sinks = self._find_sinks()
                self.physarum_network.update(sources, sinks, self.grid_state, dt)
                self._last_physarum_update = self.current_step
                self._physarum_update_count += 1

    def _find_resource_sources(self) -> List[Tuple[int, int]]:
        """Find high-vegetation cells as resource sources for physarum"""
        sources = []
        threshold = 0.6 * self.config.grid.max_vegetation
        high_veg = np.argwhere(self.grid_state.vegetation > threshold)
        for pos in high_veg[:10]:  # Limit number of sources
            sources.append((int(pos[0]), int(pos[1])))
        return sources

    def _find_sinks(self) -> List[Tuple[int, int]]:
        """Find lodge positions and project sites as sinks"""
        sinks = []
        lodge_positions = np.argwhere(self.grid_state.lodge_map)
        for pos in lodge_positions:
            sinks.append((int(pos[0]), int(pos[1])))

        # Add active project sites
        if self.config.training.use_projects:
            for project in self.project_manager.active_projects.values():
                for cell in project.region_cells[:3]:  # Limit
                    sinks.append(cell)

        return sinks

    def _compute_global_reward(self) -> float:
        """Compute global reward (wisdom signal components)"""
        reward = 0.0

        # Hydrological stability (clip variance to prevent numerical overflow)
        water_variance = np.var(self.grid_state.water_depth)
        water_variance = np.clip(water_variance, 0.0, 100.0)  # Prevent overflow
        reward -= self.config.overmind.water_variance_weight * water_variance

        # Flood penalty
        flood_threshold = 1.0  # Water depth considered flooding
        flood_cells = np.sum(self.grid_state.water_depth > flood_threshold)
        reward -= self.config.reward.flood_penalty_per_cell * flood_cells

        # Drought penalty
        low_water = np.sum(self.grid_state.water_depth < 0.01)
        reward -= self.config.reward.drought_penalty_per_cell * low_water * 0.1

        # Habitat richness
        wetland_cells = np.sum(
            (self.grid_state.water_depth > 0.1) &
            (self.grid_state.vegetation > 0.3)
        )
        reward += self.config.reward.wetland_cell_bonus * wetland_cells

        # Vegetation total
        total_vegetation = np.sum(self.grid_state.vegetation)
        reward += self.config.reward.vegetation_reward_multiplier * total_vegetation / (
            self.config.grid.grid_size ** 2
        )

        # REWARD FIX: Global structure density bonus
        # Reward the colony for total infrastructure (encourages coordination)
        n_structures = np.sum(self.grid_state.dam_permeability < 0.9)
        reward += n_structures * 0.5  # Small bonus per structure cell

        # Project completions
        if self.config.training.use_projects:
            completions = self.project_manager.check_completions(self.grid_state)
            for project_type in completions:
                if project_type == ProjectType.DAM:
                    reward += self.config.reward.dam_completion_bonus
                elif project_type == ProjectType.LODGE:
                    reward += self.config.reward.lodge_completion_bonus

                # PHASE 2: Info recovery for project success
                # All agents get an info boost when projects complete
                for agent in self.agents:
                    if agent.alive:
                        agent.recover_info(
                            self.config.agent.info_recovery_project_success,
                            self.config.agent.max_info_energy
                        )

        # Clip reward to prevent numerical instability
        return float(np.clip(reward, -1000.0, 1000.0))

    def _combine_rewards(self, agent_rewards: Dict[str, float],
                         global_reward: float) -> Dict[str, float]:
        """Combine individual and global rewards"""
        alpha = self.config.reward.individual_weight
        beta = self.config.reward.global_weight

        # REWARD FIX: Find all structure locations for proximity bonus
        structure_mask = self.grid_state.dam_permeability < 0.9
        structure_coords = np.argwhere(structure_mask) if np.any(structure_mask) else None

        # Collect alive agent positions for dispersion bonus
        alive_agents = [a for a in self.agents if a.alive]
        agent_positions = [(a.id, a.position) for a in alive_agents]

        # Per-agent survival rewards and proximity bonuses
        for agent in self.agents:
            key = f"agent_{agent.id}"
            if agent.alive:
                agent_rewards[key] += self.config.reward.alive_reward_per_step

                # REWARD FIX: Structure proximity bonus (rewards being near ANY structure)
                if structure_coords is not None and len(structure_coords) > 0:
                    y, x = agent.position
                    distances = np.sqrt(
                        (structure_coords[:, 0] - y)**2 +
                        (structure_coords[:, 1] - x)**2
                    )
                    # Count structures within range
                    nearby_count = np.sum(distances < self.config.reward.structure_proximity_range)
                    if nearby_count > 0:
                        # Diminishing returns for many nearby structures
                        proximity_bonus = self.config.reward.structure_proximity_bonus * np.sqrt(nearby_count)
                        agent_rewards[key] += proximity_bonus

                # DISPERSION BONUS: Reward for spreading out, not clustering
                if len(agent_positions) > 1:
                    y, x = agent.position
                    min_other_dist = float('inf')
                    for other_id, other_pos in agent_positions:
                        if other_id != agent.id:
                            dist = np.sqrt((other_pos[0] - y)**2 + (other_pos[1] - x)**2)
                            min_other_dist = min(min_other_dist, dist)

                    # Reward if maintaining minimum distance from nearest other agent
                    min_desired = self.config.reward.min_dispersion_distance
                    if min_other_dist >= min_desired:
                        # Full bonus for being well-dispersed
                        dispersion_bonus = self.config.reward.dispersion_bonus
                    else:
                        # Partial bonus scaled by distance ratio
                        dispersion_bonus = self.config.reward.dispersion_bonus * (min_other_dist / min_desired)
                    agent_rewards[key] += dispersion_bonus

            else:
                agent_rewards[key] += self.config.reward.death_penalty

        # COVERAGE BONUS: Reward team for exploring more unique cells collectively
        if len(alive_agents) > 0:
            total_visited = set()
            for a in alive_agents:
                if a.cells_visited is not None:
                    total_visited.update(a.cells_visited)
            grid_size = self.config.grid.grid_size
            coverage_fraction = len(total_visited) / (grid_size * grid_size)
            # Bonus increases with coverage, shared among alive agents
            if coverage_fraction > 0.1:  # Only give bonus after exploring 10% of map
                coverage_reward = self.config.reward.coverage_bonus * coverage_fraction
                for agent in alive_agents:
                    key = f"agent_{agent.id}"
                    agent_rewards[key] += coverage_reward / len(alive_agents)

        # Combine
        combined = {}
        n_alive = sum(1 for a in self.agents if a.alive)
        global_per_agent = global_reward / max(1, n_alive)

        for key, ind_reward in agent_rewards.items():
            # Clip individual rewards for stability
            ind_reward = float(np.clip(ind_reward, -100.0, 100.0))
            final_reward = alpha * ind_reward + beta * global_per_agent
            # Clip combined reward to prevent numerical issues
            combined[key] = float(np.clip(final_reward, -1000.0, 1000.0))

        return combined

    def _check_terminated(self) -> bool:
        """Check if episode should terminate"""
        # All agents dead
        if not any(a.alive for a in self.agents):
            return True

        # Could add other termination conditions
        return False

    def _get_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get observations for all agents"""
        observations = {}

        for agent in self.agents:
            if not agent.alive:
                # Dead agent gets zero observation
                obs = self._get_zero_observation()
            else:
                obs = self._get_agent_observation(agent)
            observations[f"agent_{agent.id}"] = obs

        return observations

    def _get_agent_observation(self, agent: AgentState) -> Dict[str, np.ndarray]:
        """Get observation for a single agent"""
        y, x = agent.position
        r = self.config.policy.local_view_radius
        size = self.config.grid.grid_size

        # Local grid view with padding
        local_grid = np.zeros((self.config.policy.n_local_channels, 2*r+1, 2*r+1), dtype=np.float32)

        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                ny, nx = y + dy, x + dx
                oy, ox = dy + r, dx + r

                if 0 <= ny < size and 0 <= nx < size:
                    local_grid[0, oy, ox] = self.grid_state.elevation[ny, nx]
                    local_grid[1, oy, ox] = self.grid_state.water_depth[ny, nx]
                    local_grid[2, oy, ox] = self.grid_state.vegetation[ny, nx]
                    local_grid[3, oy, ox] = self.grid_state.soil_moisture[ny, nx]
                    local_grid[4, oy, ox] = 1.0 - self.grid_state.dam_permeability[ny, nx]
                    local_grid[5, oy, ox] = float(self.grid_state.lodge_map[ny, nx])

                    # Pheromone (average of edges from this cell)
                    if self.pheromone_field is not None:
                        local_grid[6, oy, ox] = self.pheromone_field.get_average_level(ny, nx)

                    # Physarum conductivity
                    if self.physarum_network is not None:
                        local_grid[7, oy, ox] = self.physarum_network.get_average_conductivity(ny, nx)

        # Normalize local grid
        local_grid = np.clip(local_grid, 0, 1)

        # Global features
        global_features = np.zeros(self.config.policy.n_global_features, dtype=np.float32)

        # Project recruitment signals
        if self.project_manager is not None:
            project_signals = self.project_manager.get_recruitment_signals()
            global_features[:len(project_signals)] = project_signals[:self.config.policy.n_global_features // 2]

        # Overmind wisdom (if available)
        if self.overmind is not None:
            wisdom = self.overmind.get_wisdom_signal()
            global_features[self.config.policy.n_global_features // 2] = wisdom

        # Colony statistics
        n_alive = sum(1 for a in self.agents if a.alive)
        global_features[-2] = n_alive / self.config.n_beavers
        global_features[-1] = np.mean([a.energy for a in self.agents if a.alive]) / self.config.agent.initial_energy

        # Internal state with all 6 role flags
        internal_state = np.array([
            agent.energy / self.config.agent.initial_energy,
            agent.satiety,
            agent.wetness,
            float(agent.role == AgentRole.SCOUT),
            float(agent.role == AgentRole.WORKER),
            float(agent.role == AgentRole.GUARDIAN),
            float(agent.role == AgentRole.BUILDER),
            float(agent.role == AgentRole.HAULER),
            float(agent.role == AgentRole.MAINTAINER),
            agent.carrying_wood / self.config.agent.max_carry_wood,
            float(agent.current_project is not None),
        ], dtype=np.float32)

        return {
            "local_grid": local_grid,
            "global_features": global_features,
            "internal_state": internal_state,
        }

    def _get_zero_observation(self) -> Dict[str, np.ndarray]:
        """Get zero observation for dead agent"""
        r = self.config.policy.local_view_radius
        return {
            "local_grid": np.zeros((self.config.policy.n_local_channels, 2*r+1, 2*r+1), dtype=np.float32),
            "global_features": np.zeros(self.config.policy.n_global_features, dtype=np.float32),
            "internal_state": np.zeros(self.config.policy.n_internal_features, dtype=np.float32),
        }

    def _get_overmind_observation(self) -> np.ndarray:
        """
        Get observation for overmind meta-controller.

        PHASE 2: Extended to include info dissipation metrics.
        """
        features = []

        # Water statistics
        features.append(np.mean(self.grid_state.water_depth))
        features.append(np.var(self.grid_state.water_depth))
        features.append(np.max(self.grid_state.water_depth))

        # Vegetation
        features.append(np.mean(self.grid_state.vegetation))

        # Flood/drought indicators
        flood_cells = np.sum(self.grid_state.water_depth > 1.0)
        drought_cells = np.sum(self.grid_state.water_depth < 0.01)
        features.append(flood_cells / (self.config.grid.grid_size ** 2))
        features.append(drought_cells / (self.config.grid.grid_size ** 2))

        # Colony health
        n_alive = sum(1 for a in self.agents if a.alive)
        avg_energy = np.mean([a.energy for a in self.agents if a.alive]) if n_alive > 0 else 0
        features.append(n_alive / self.config.n_beavers)
        features.append(avg_energy / self.config.agent.initial_energy)

        # Project statistics
        if self.project_manager is not None:
            n_active = len(self.project_manager.active_projects)
            features.append(n_active / 10.0)  # Normalized
        else:
            features.append(0.0)

        # === PHASE 2: Info dissipation metrics ===
        # Total info dissipation this step (from all subsystems)
        total_info_dissipation = 0.0
        info_blocked = 0

        # Collect from pheromone field
        if self.pheromone_field is not None:
            total_info_dissipation += self.pheromone_field.get_info_dissipation()
            stats = self.pheromone_field.get_statistics()
            info_blocked += stats.get("deposits_blocked_by_info", 0)

        # Average agent info energy
        avg_info_energy = 100.0
        if n_alive > 0:
            alive_agents = [a for a in self.agents if a.alive]
            avg_info_energy = np.mean([a.info_energy for a in alive_agents])

        features.append(total_info_dissipation / 100.0)  # Normalized
        features.append(avg_info_energy / self.config.agent.max_info_energy)  # Normalized
        features.append(float(info_blocked) / 10.0)  # Normalized

        # Pad to expected size
        while len(features) < self.config.overmind.n_observation_features:
            features.append(0.0)

        return np.array(features[:self.config.overmind.n_observation_features], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        n_alive = sum(1 for a in self.agents if a.alive)
        n_structures = np.sum(self.grid_state.dam_permeability < 0.9)

        # Hydrology metrics
        water_autocorr = self.hydrology_engine.compute_spatial_autocorrelation(self.grid_state)
        pools = self.hydrology_engine.detect_pools(self.grid_state)
        n_pools = np.sum(pools & (self.grid_state.water_depth > self.config.grid.pool_threshold))

        # Dam effectiveness (upstream-downstream delta for each dam)
        dam_mask = self.grid_state.dam_permeability < 0.9
        dam_effectiveness = 0.0
        if np.any(dam_mask):
            dam_coords = np.argwhere(dam_mask)
            for y, x in dam_coords[:5]:  # Sample up to 5 dams
                delta = self.hydrology_engine.get_upstream_downstream_delta(self.grid_state, y, x)
                dam_effectiveness += max(0, delta)  # Positive delta = water pooling upstream
            dam_effectiveness /= len(dam_coords[:5])

        # Vegetation metrics
        veg = self.grid_state.vegetation
        veg_normalized = veg / (np.max(veg) + 1e-8)
        # Vegetation entropy (higher = more variation, not flat)
        veg_nonzero = veg_normalized[veg_normalized > 0.01]
        if len(veg_nonzero) > 0:
            veg_probs = veg_nonzero / np.sum(veg_nonzero)
            veg_entropy = -np.sum(veg_probs * np.log(veg_probs + 1e-8))
        else:
            veg_entropy = 0.0

        # Structure physics metrics
        dam_lifetime_stats = self.structure_physics.get_dam_lifetime_stats()
        avg_dam_integrity = 0.0
        low_integrity_count = 0
        if np.any(dam_mask):
            dam_integrities = self.grid_state.dam_integrity[dam_mask]
            avg_dam_integrity = float(np.mean(dam_integrities))
            low_integrity_count = int(np.sum(dam_integrities < self.config.grid.dam_failure_threshold * 2))

        info = {
            "step": self.current_step,
            "n_alive_agents": n_alive,
            "n_structures": n_structures,
            "total_vegetation": np.sum(self.grid_state.vegetation),
            "avg_water_level": np.mean(self.grid_state.water_depth),
            "episode_reward": self.episode_reward,
            # Hydrology metrics
            "water_spatial_autocorr": water_autocorr,
            "n_water_pools": n_pools,
            "dam_effectiveness": dam_effectiveness,
            "rain_active": self.hydrology_engine.rain_active,
            # Vegetation metrics
            "vegetation_entropy": veg_entropy,
            "vegetation_variance": float(np.var(veg)),
            # Structure physics metrics
            "avg_dam_integrity": avg_dam_integrity,
            "low_integrity_dams": low_integrity_count,
            "dam_failures": dam_lifetime_stats.get("n_failures", 0),
            "flood_events_caused": dam_lifetime_stats.get("flood_events_caused", 0),
        }

        # === PHASE 3: Time-scale separation tracking ===
        # Only include if tracking is enabled (useful for debugging/logging)
        if self.config.time_scales.track_update_counts:
            info["time_scale_stats"] = {
                "physarum_updates": self._physarum_update_count,
                "overmind_updates": self._overmind_update_count,
                "semantic_consolidations": self._semantic_consolidation_count,
                "last_physarum_update": self._last_physarum_update,
                "last_overmind_update": self._last_overmind_update,
                "last_semantic_consolidation": self._last_semantic_consolidation,
            }

        return info

    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        elif self.render_mode == "human":
            # Would use matplotlib or pygame
            pass

    def _render_rgb(self) -> np.ndarray:
        """Render environment as RGB image"""
        size = self.config.grid.grid_size
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Water (blue)
        water_normalized = np.clip(self.grid_state.water_depth / 0.5, 0, 1)
        img[:, :, 2] = (water_normalized * 255).astype(np.uint8)

        # Vegetation (green)
        veg_normalized = self.grid_state.vegetation / self.config.grid.max_vegetation
        img[:, :, 1] = np.clip(
            (veg_normalized * 200).astype(np.uint8) + img[:, :, 1],
            0, 255
        )

        # Dams (brown/red)
        dam_mask = self.grid_state.dam_permeability < 0.9
        img[dam_mask, 0] = 139  # Brown-ish
        img[dam_mask, 1] = 69
        img[dam_mask, 2] = 19

        # Lodges (white)
        img[self.grid_state.lodge_map, :] = 255

        # Agents (yellow dots)
        for agent in self.agents:
            if agent.alive:
                y, x = agent.position
                img[y, x, 0] = 255
                img[y, x, 1] = 255
                img[y, x, 2] = 0

        return img

    def close(self):
        """Clean up resources"""
        pass
