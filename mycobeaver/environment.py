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
    lodge_map: np.ndarray  # Boolean indicating lodge locations

    # Agent presence (updated each step)
    agent_positions: np.ndarray  # Count of agents per cell


class HydrologyEngine:
    """
    Computes water flow dynamics between cells.

    Water flows from high to low water surface height:
        H_i = elevation_i + h_i
        q_ij = g_ij * (H_i - H_j)
        g_ij = g0 * f(d_i, d_j)
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.size = config.grid_size

        # Precompute neighbor offsets (4-connected)
        self.neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def compute_water_surface_height(self, state: GridState) -> np.ndarray:
        """Compute H_i = elevation_i + h_i"""
        return state.elevation + state.water_depth

    def compute_conductance(self, state: GridState) -> np.ndarray:
        """
        Compute conductance matrix g_ij for each cell and its neighbors.
        Returns shape (grid_size, grid_size, 4) for 4 neighbors.
        """
        g = np.zeros((self.size, self.size, 4))
        g0 = self.config.base_conductance

        for idx, (dy, dx) in enumerate(self.neighbor_offsets):
            # Shifted permeability arrays
            d_j = np.roll(np.roll(state.dam_permeability, dy, axis=0), dx, axis=1)

            # f(d_i, d_j) = 0.5 * (d_i + d_j)
            f = self.config.dam_permeability_effect * (state.dam_permeability + d_j)

            g[:, :, idx] = g0 * f

        return g

    def update_water(self, state: GridState, rainfall: np.ndarray,
                     boundary_inflow: np.ndarray, dt: float) -> np.ndarray:
        """
        Update water depth for one time step.

        h_i(t+1) = h_i(t) + dt * (rainfall + boundary_inflow + inflows - outflows - losses)
        """
        H = self.compute_water_surface_height(state)
        g = self.compute_conductance(state)

        # Compute net flux for each cell
        net_flux = np.zeros((self.size, self.size))

        for idx, (dy, dx) in enumerate(self.neighbor_offsets):
            # Neighbor's water surface height
            H_j = np.roll(np.roll(H, dy, axis=0), dx, axis=1)

            # Flux from i to j: positive means outflow from i
            # q_ij = g_ij * (H_i - H_j)
            flux_to_neighbor = g[:, :, idx] * (H - H_j)

            # Net flux: subtract outflows, add inflows
            # This is outflow from cell i
            net_flux -= flux_to_neighbor

        # Loss terms: evaporation and seepage
        loss = (self.config.evaporation_rate + self.config.seepage_rate) * state.water_depth

        # Update water depth
        new_water = state.water_depth + dt * (rainfall + boundary_inflow + net_flux - loss)

        # Ensure non-negative
        new_water = np.maximum(new_water, 0.0)

        return new_water


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

        Regrowth rate depends on soil moisture.
        Harvesting by agents reduces vegetation.
        """
        # Regrowth rate: higher moisture = faster growth
        base_rate = self.config.vegetation_regrowth_rate
        moisture_bonus = self.config.moisture_effect_on_growth * state.soil_moisture
        regrowth_rate = base_rate * (1 + moisture_bonus)

        # Growth limited by distance from maximum
        growth = regrowth_rate * (self.config.max_vegetation - state.vegetation)

        # Update vegetation
        new_vegetation = state.vegetation + dt * growth - harvested

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

        # Placeholders for subsystems (initialized in reset)
        self.pheromone_field = None
        self.project_manager = None
        self.physarum_network = None
        self.overmind = None

        # Grid state
        self.grid_state: Optional[GridState] = None

        # Agents
        self.agents: List[AgentState] = []

        # Simulation state
        self.current_step = 0
        self.episode_reward = 0.0

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

        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def _initialize_grid(self):
        """Initialize grid state arrays"""
        size = self.config.grid.grid_size

        # Generate elevation (example: gradient with noise)
        base_elevation = np.linspace(0.5, 0.0, size).reshape(-1, 1)
        base_elevation = np.tile(base_elevation, (1, size))
        elevation_noise = self.np_random.uniform(-0.1, 0.1, (size, size))
        elevation = base_elevation + elevation_noise

        # If stream location specified, create lower channel
        if self.config.stream_location:
            x1, y1, x2, y2 = self.config.stream_location
            for i in range(min(y1, y2), max(y1, y2) + 1):
                for j in range(min(x1, x2), max(x1, x2) + 1):
                    elevation[i, j] -= 0.2

        # Initialize water (some initial level, higher in low elevation)
        water_depth = np.where(
            elevation < 0.3,
            self.config.initial_water_level,
            0.0
        )

        # Initialize vegetation (denser away from water)
        vegetation = self.np_random.uniform(
            0.3 * self.config.initial_vegetation_density,
            self.config.initial_vegetation_density,
            (size, size)
        )
        vegetation = np.where(water_depth > 0.1, 0.1 * vegetation, vegetation)

        # Initialize soil moisture
        soil_moisture = 0.5 * np.ones((size, size))
        soil_moisture = np.where(water_depth > 0, 0.9, soil_moisture)

        # Initialize dams (no dams initially)
        dam_permeability = np.ones((size, size))

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
            lodge_map=lodge_map,
            agent_positions=agent_positions
        )

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

            # Assign role (initial distribution)
            if i < self.config.n_beavers * 0.2:
                role = AgentRole.SCOUT
            elif i < self.config.n_beavers * 0.6:
                role = AgentRole.WORKER
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
        """Initialize pheromone field, project manager, physarum network, overmind"""
        from .pheromone import PheromoneField
        from .projects import ProjectManager
        from .physarum import PhysarumNetwork
        from .overmind import Overmind

        self.pheromone_field = PheromoneField(self.config.pheromone, self.config.grid.grid_size)
        self.project_manager = ProjectManager(self.config.project, self.config.grid.grid_size)
        self.physarum_network = PhysarumNetwork(self.config.physarum, self.config.grid.grid_size)
        self.overmind = Overmind(self.config.overmind, self.config)

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

        # 1. Overmind phase
        if self.config.training.use_overmind:
            overmind_obs = self._get_overmind_observation()
            self.overmind.update(overmind_obs, self)

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

            # Energy cost
            agent.energy -= config.move_energy_cost

        elif action == ActionType.STAY:
            # Minimal energy cost
            pass

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
            if veg_available > 0.1:
                harvest_amount = min(0.2, veg_available)
                harvested[y, x] += harvest_amount
                agent.satiety = min(1.0, agent.satiety + config.forage_satiety_gain)
                agent.carrying_wood = min(config.max_carry_wood, agent.carrying_wood + 1)
                reward += self.config.reward.forage_reward
            agent.energy -= config.forage_energy_cost

        elif action == ActionType.BUILD_DAM:
            # Build/reinforce dam at current location
            if agent.carrying_wood > 0 and agent.current_project is not None:
                old_permeability = self.grid_state.dam_permeability[y, x]
                self.grid_state.dam_permeability[y, x] = max(
                    0.0, old_permeability - self.config.grid.dam_build_amount
                )
                agent.carrying_wood -= 1

                # Update project progress
                if self.config.training.use_projects:
                    self.project_manager.add_progress(agent.current_project, 1)

                reward += self.config.reward.build_action_reward
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
                if veg > 0.2:
                    harvested[y, x] += 0.1
                    agent.carrying_wood += 1

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
        self.grid_state.water_depth = new_water

        # Update soil moisture
        new_moisture = self.vegetation_engine.update_soil_moisture(self.grid_state, dt)
        self.grid_state.soil_moisture = new_moisture

    def _update_subsystems(self, dt: float):
        """Update pheromones, projects, physarum"""
        # Pheromone evaporation
        if self.config.training.use_pheromones:
            self.pheromone_field.evaporate(dt)

        # Project recruitment signals
        if self.config.training.use_projects:
            advertising_scouts = [a.id for a in self.agents if a.role == AgentRole.SCOUT
                                 and a.current_project is not None]
            self.project_manager.update_recruitment(advertising_scouts, dt)

        # Physarum network
        if self.config.training.use_physarum:
            if self.current_step % self.config.physarum.update_every_n_steps == 0:
                # Determine sources and sinks
                sources = self._find_resource_sources()
                sinks = self._find_sinks()
                self.physarum_network.update(sources, sinks, self.grid_state, dt)

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

        # Hydrological stability
        water_variance = np.var(self.grid_state.water_depth)
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

        return reward

    def _combine_rewards(self, agent_rewards: Dict[str, float],
                         global_reward: float) -> Dict[str, float]:
        """Combine individual and global rewards"""
        alpha = self.config.reward.individual_weight
        beta = self.config.reward.global_weight

        # Per-agent survival rewards
        for agent in self.agents:
            key = f"agent_{agent.id}"
            if agent.alive:
                agent_rewards[key] += self.config.reward.alive_reward_per_step
            else:
                agent_rewards[key] += self.config.reward.death_penalty

        # Combine
        combined = {}
        n_alive = sum(1 for a in self.agents if a.alive)
        global_per_agent = global_reward / max(1, n_alive)

        for key, ind_reward in agent_rewards.items():
            combined[key] = alpha * ind_reward + beta * global_per_agent

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

        # Internal state
        internal_state = np.array([
            agent.energy / self.config.agent.initial_energy,
            agent.satiety,
            agent.wetness,
            float(agent.role.value == 'scout'),
            float(agent.role.value == 'worker'),
            float(agent.role.value == 'guardian'),
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

        return {
            "step": self.current_step,
            "n_alive_agents": n_alive,
            "n_structures": n_structures,
            "total_vegetation": np.sum(self.grid_state.vegetation),
            "avg_water_level": np.mean(self.grid_state.water_depth),
            "episode_reward": self.episode_reward,
        }

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
