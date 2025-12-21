"""
MycoBeaver Simulator Configuration
===================================
Complete configuration system based on the MycoBeaver Design Plan.
All hyperparameters for environment, agents, and training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class AgentRole(Enum):
    """Beaver agent behavioral roles"""
    SCOUT = "scout"
    WORKER = "worker"
    GUARDIAN = "guardian"
    BUILDER = "builder"


class ActionType(Enum):
    """Agent action types"""
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    MOVE_NE = 4
    MOVE_NW = 5
    MOVE_SE = 6
    MOVE_SW = 7
    STAY = 8
    FORAGE = 9
    BUILD_DAM = 10
    BUILD_LODGE = 11
    CARRY_RESOURCE = 12
    DROP_RESOURCE = 13
    REST = 14
    ADVERTISE_PROJECT = 15


class ProjectType(Enum):
    """Types of construction projects"""
    DAM = "dam"
    LODGE = "lodge"
    CANAL = "canal"


class ProjectStatus(Enum):
    """Project lifecycle states"""
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class GridConfig:
    """Grid environment configuration"""
    # Grid dimensions
    grid_size: int = 64  # NxN grid

    # Hydrology parameters
    base_conductance: float = 0.5  # g0 base water conductance
    evaporation_rate: float = 0.01  # α_evap
    seepage_rate: float = 0.005  # α_seep
    rainfall_rate: float = 0.02  # Base rainfall per step
    rainfall_variance: float = 0.01  # Variance in rainfall

    # Vegetation parameters
    vegetation_regrowth_rate: float = 0.01  # Base regrowth rate
    max_vegetation: float = 1.0  # Maximum biomass per cell
    moisture_effect_on_growth: float = 0.5  # Higher moisture = faster growth

    # Soil parameters
    moisture_diffusion_rate: float = 0.1  # How fast moisture spreads
    moisture_decay_rate: float = 0.05  # How fast dry soil loses moisture

    # Dam parameters
    dam_permeability_effect: float = 0.5  # f(d_i, d_j) = 0.5 * (d_i + d_j)
    dam_build_amount: float = 0.1  # Δd per build action

    # Time step
    dt: float = 1.0  # Simulation time step


@dataclass
class AgentConfig:
    """Beaver agent configuration"""
    # Initial state
    initial_energy: float = 100.0
    initial_satiety: float = 0.8  # 80% full
    initial_wetness: float = 0.0

    # Energy dynamics
    move_energy_cost: float = 1.0
    build_energy_cost: float = 5.0
    forage_energy_cost: float = 2.0
    rest_energy_gain: float = 2.0
    base_energy_decay: float = 0.5  # Per time step

    # Satiety dynamics
    satiety_decay_rate: float = 0.02  # Per time step
    forage_satiety_gain: float = 0.2  # Per forage action
    starvation_energy_multiplier: float = 2.0  # Energy cost multiplier when hungry

    # Wetness dynamics
    water_wetness_rate: float = 0.1  # Wetness gain in water
    lodge_drying_rate: float = 0.2  # Wetness loss in lodge
    ambient_drying_rate: float = 0.02  # Wetness loss elsewhere
    wetness_energy_penalty: float = 0.5  # Extra energy cost per wetness

    # Task response thresholds (personality parameters)
    n_task_types: int = 5  # M task types
    threshold_mean: float = 0.5
    threshold_std: float = 0.15

    # Carrying capacity
    max_carry_wood: int = 2  # Wood units agent can carry (increased for faster building)

    # Foraging thresholds (lowered for easier wood acquisition)
    forage_min_vegetation: float = 0.05  # Min vegetation to forage
    carry_min_vegetation: float = 0.1  # Min vegetation to pick up wood

    # Role parameters
    scout_explore_bonus: float = 1.5  # Scouts get bonus for exploration
    worker_build_bonus: float = 1.5  # Workers get bonus for building
    guardian_stay_bonus: float = 1.5  # Guardians get bonus for staying near lodge

    # Death conditions
    min_energy: float = 0.0
    min_satiety_before_damage: float = 0.1

    # === PHASE 2: Information Energy ===
    # Information is a first-class resource with explicit costs
    initial_info_energy: float = 100.0  # Starting info energy
    max_info_energy: float = 150.0  # Maximum info energy capacity

    # Info energy recovery rates
    info_recovery_base: float = 0.5  # Passive recovery per step
    info_recovery_coordination: float = 2.0  # Bonus for successful coordination
    info_recovery_project_success: float = 10.0  # Bonus when project completes
    info_recovery_uncertainty_reduction: float = 1.0  # Bonus for reducing entropy


@dataclass
class InfoCostConfig:
    """
    PHASE 2: Information Thermodynamics - Explicit info costs.

    Every information-producing action has a cost.
    This prevents unbounded communication and incentivizes
    efficient information flow.
    """
    # Communication costs
    cost_send_message: float = 1.0  # c_msg - sending any message
    cost_broadcast: float = 2.0  # Higher cost for broadcasts
    cost_consensus_vote: float = 0.5  # Voting in consensus

    # Pheromone costs
    cost_pheromone_deposit: float = 0.5  # c_pher - depositing pheromone
    cost_pheromone_reinforce: float = 1.0  # Extra cost for path reinforcement

    # Semantic costs
    cost_semantic_vertex: float = 1.5  # c_sem - adding knowledge vertex
    cost_semantic_edge: float = 0.5  # Adding edge to knowledge graph
    cost_semantic_query: float = 0.2  # Querying the semantic graph

    # Project costs
    cost_project_proposal: float = 5.0  # c_proj - proposing new project
    cost_project_advertise: float = 1.0  # Waggle dance advertising
    cost_project_vote: float = 0.5  # Voting on project

    # Minimum info energy to perform actions
    # Below this, actions fail silently (agents become "quiet")
    min_info_for_action: float = 1.0


@dataclass
class PheromoneConfig:
    """Ant-style pheromone routing configuration"""
    # Evaporation
    evaporation_rate: float = 0.1  # ρ - 10% decay per step

    # Deposition
    base_deposit: float = 1.0  # δ_k base deposit amount
    success_deposit_multiplier: float = 2.0  # Extra deposit on successful path

    # Routing influence
    pheromone_alpha: float = 1.0  # α - pheromone influence exponent
    heuristic_beta: float = 2.0  # β - heuristic influence exponent
    exploration_epsilon: float = 0.01  # ε - minimum baseline for exploration

    # Clipping
    max_pheromone: float = 100.0
    min_pheromone: float = 0.0


@dataclass
class ProjectConfig:
    """Bee-style project recruitment configuration"""
    # Quality estimation weights
    resource_weight: float = 0.3  # w1
    hydro_impact_weight: float = 0.3  # w2
    safety_weight: float = 0.2  # w3
    distance_cost_weight: float = 0.2  # w4

    # Recruitment dynamics
    recruitment_decay: float = 0.1  # κ - dance enthusiasm decay
    dance_gain: float = 1.0  # γ_dance
    quality_sensitivity: float = 1.0  # λ in f(Q) = exp(λQ)

    # Cross-inhibition
    cross_inhibition: float = 0.1  # χ - suppression of competing projects

    # Thresholds
    recruitment_threshold: float = 0.5  # R_p threshold for agent recruitment
    quorum_threshold: float = 0.8  # Fraction needed for consensus

    # Project requirements
    dam_wood_required: int = 100  # Wood units for dam
    lodge_wood_required: int = 50  # Wood units for lodge


@dataclass
class PhysarumConfig:
    """Physarum-inspired transport network configuration"""
    # Initial conductivity
    initial_conductivity: float = 0.1  # D_ij(0)

    # Adaptation parameters
    reinforcement_rate: float = 0.5  # α_D
    decay_rate: float = 0.1  # β_D
    flow_exponent: float = 1.0  # γ in g(|Q|) = |Q|^γ

    # Edge costs
    base_length: float = 1.0  # L_ij(0) for adjacent cells
    elevation_cost_factor: float = 0.5  # λ_z for elevation difference
    water_cost_factor: float = 0.3  # λ_h for water depth

    # Bounds
    min_conductivity: float = 0.001
    max_conductivity: float = 10.0

    # Update frequency
    update_every_n_steps: int = 1


@dataclass
class OvermindConfig:
    """Overmind meta-controller configuration"""
    # Observation features
    n_observation_features: int = 20

    # Action space (meta-parameters to adjust)
    n_actions: int = 8  # ρ, κ, γ_dance, thresholds, etc.

    # Wisdom signal weights
    water_variance_weight: float = 1.0  # λ_σ
    flood_weight: float = 2.0  # λ_F
    drought_weight: float = 2.0  # λ_D
    failure_weight: float = 5.0  # λ_B (dam breaches, agent deaths)
    habitat_weight: float = 1.0  # λ_H

    # Action bounds
    pheromone_evap_bounds: Tuple[float, float] = (0.01, 0.5)
    recruitment_decay_bounds: Tuple[float, float] = (0.01, 0.5)
    dance_gain_bounds: Tuple[float, float] = (0.1, 5.0)
    threshold_adjustment_bounds: Tuple[float, float] = (0.5, 2.0)

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])


@dataclass
class RewardConfig:
    """Reward structure configuration"""
    # === REBALANCED REWARDS (Fix survival >> building imbalance) ===

    # Survival rewards (REDUCED to not dominate)
    alive_reward_per_step: float = 0.01  # Was 0.1 - reduced 10x
    death_penalty: float = -50.0  # Was -100.0 - reduced to not discourage risk

    # Hydrological stability
    flood_penalty_per_cell: float = -0.1  # Was -0.5 - reduced
    drought_penalty_per_cell: float = -0.1  # Was -0.5 - reduced
    stability_bonus: float = 1.0

    # Habitat rewards
    vegetation_reward_multiplier: float = 0.1
    wetland_cell_bonus: float = 0.05  # Water + vegetation

    # Project rewards (INCREASED significantly)
    dam_completion_bonus: float = 100.0  # Was 50.0 - doubled
    lodge_completion_bonus: float = 60.0  # Was 30.0 - doubled

    # Resource rewards
    forage_reward: float = 0.5
    build_action_reward: float = 10.0  # Was 1.0 - increased 10x

    # === NEW REWARDS (Fix missing incentives) ===

    # Carrying wood reward (with proximity bonus)
    carry_wood_reward: float = 0.5  # Reward for picking up wood
    carry_wood_proximity_bonus: float = 1.0  # Extra reward when near build site
    carry_wood_distance_threshold: float = 5.0  # Distance for proximity bonus

    # Exploration reward (diminishing returns)
    exploration_reward: float = 2.0  # Base reward for visiting new cells (increased)
    exploration_decay: float = 0.5  # Multiplier for revisiting cells

    # Dispersion bonus (encourage spreading out, not clustering)
    dispersion_bonus: float = 0.5  # Reward per unit distance from other agents
    min_dispersion_distance: float = 3.0  # Minimum desired distance between agents
    coverage_bonus: float = 5.0  # Bonus for team covering more unique cells

    # Personal structure credit (fix free-rider problem)
    personal_structure_bonus: float = 15.0  # Reward for structures YOU built
    structure_proximity_bonus: float = 0.5  # Reward for being near ANY structure
    structure_proximity_range: float = 10.0  # Range for proximity bonus

    # Efficiency penalty
    wasted_material_penalty: float = -1.0

    # Global vs individual balance (REBALANCED)
    individual_weight: float = 0.5  # α - Was 0.3, increased for personal credit
    global_weight: float = 0.5  # β - Was 0.7, reduced to fix free-rider


@dataclass
class PolicyNetworkConfig:
    """Neural network policy configuration"""
    # Observation space
    local_view_radius: int = 5  # r - agent sees (2r+1) x (2r+1) grid
    n_local_channels: int = 8  # elevation, water, vegetation, soil, dam, lodge, pheromone, physarum
    n_global_features: int = 16  # Colony signals, project recruitment, etc.
    n_internal_features: int = 8  # Energy, satiety, wetness, role, thresholds...

    # Network architecture
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    fc_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    # Action space
    n_actions: int = 16  # Number of discrete actions

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clip
    entropy_coef: float = 0.05  # Starting entropy (increased for exploration)
    entropy_coef_final: float = 0.005  # Final entropy (lower for exploitation)
    entropy_decay_episodes: int = 500  # Episodes over which to decay entropy
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Shared policy
    share_policy: bool = True  # All agents share one policy


@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    # Episodes
    n_episodes: int = 10000
    max_steps_per_episode: int = 1000

    # Batch
    batch_size: int = 64
    n_epochs_per_update: int = 4

    # Parallelization
    n_parallel_envs: int = 4

    # Checkpointing
    checkpoint_every: int = 100
    checkpoint_dir: str = "./checkpoints"

    # Logging
    log_every: int = 10
    tensorboard_dir: str = "./tensorboard"

    # Seeds
    seed: int = 42

    # Component toggles for ablation
    use_pheromones: bool = True
    use_physarum: bool = True
    use_overmind: bool = True
    use_projects: bool = True

    # === PHASE 6: PPO Stability Enhancements ===

    # KL-target early stopping
    # If KL divergence exceeds target, stop epoch early to prevent catastrophic updates
    kl_target: float = 0.015  # Target KL divergence
    kl_early_stop: bool = True  # Enable early stopping on KL

    # Value loss normalization
    # Normalize value targets to improve training stability
    normalize_value_targets: bool = True
    value_clip_range: float = 0.2  # Clip value function updates (PPO-style)

    # Advantage normalization (per-batch vs global)
    normalize_advantages_per_batch: bool = True

    # === PHASE 6: Reproducibility ===

    # Full determinism (may slow down training)
    deterministic: bool = False

    # Seed control
    torch_seed: Optional[int] = None  # If None, uses main seed
    numpy_seed: Optional[int] = None  # If None, uses main seed
    env_seed: Optional[int] = None  # If None, uses main seed

    # Config dumping
    dump_config_on_start: bool = True
    config_dump_dir: str = "./configs"


@dataclass
class TimeScaleConfig:
    """
    PHASE 3: Time-Scale Separation Configuration

    Critical for RL stability - prevents non-stationarity collapse by
    ensuring different adaptive systems update at different frequencies.

    Update Hierarchy:
    - PPO (agent learning): Every step (fastest)
    - Pheromone dynamics: Every step (but decay is natural)
    - Physarum network: Every N steps (medium, 10-20)
    - Overmind modulation: Every M steps (slow, 50-100)
    - Semantic consolidation: Episode end only (slowest)
    """
    # Fast systems (every step)
    # PPO updates happen during training, not environment step
    # Pheromone evaporation happens every step naturally

    # Medium systems
    physarum_update_interval: int = 15  # Update physarum every N steps
    project_recruitment_interval: int = 5  # Update recruitment signals

    # Slow systems
    overmind_update_interval: int = 50  # Overmind modulation every M steps
    semantic_consolidation_interval: int = 100  # Consolidate semantic graph

    # Episode-level systems
    semantic_consolidate_on_episode_end: bool = True
    clear_semantic_ants_on_episode_end: bool = True

    # Tracking (for debugging/logging)
    track_update_counts: bool = True


@dataclass
class SemanticConfig:
    """Distributed Cognitive Architecture - Semantic subsystem config"""
    # Knowledge graph
    max_vertices: int = 10000
    max_edges_per_vertex: int = 20

    # Semantic entropy
    contradiction_resolution_cost: float = 1.0

    # Coherence thresholds
    coherence_threshold: float = 0.7
    semantic_temperature_init: float = 1.0
    semantic_temperature_min: float = 0.1  # Minimum temperature (exploration floor)

    # Pheromone for semantic graph
    semantic_pheromone_decay: float = 0.1
    semantic_pheromone_deposit: float = 1.0

    # Ant traversal
    traversal_steps_per_update: int = 10


@dataclass
class CommunicationConfig:
    """Distributed Cognitive Architecture - Communication subsystem config"""
    # Network topology
    n_message_types: int = 10
    max_message_queue: int = 100

    # Consensus
    quorum_fraction: float = 0.6
    consensus_timeout: int = 100  # Steps

    # Clock synchronization
    sync_interval: int = 10
    max_clock_drift: float = 0.1

    # Bandwidth
    max_bandwidth_per_step: int = 50  # Messages


@dataclass
class SimulationConfig:
    """Master configuration combining all subsystems"""
    # Environment
    grid: GridConfig = field(default_factory=GridConfig)

    # Agents
    n_beavers: int = 20
    agent: AgentConfig = field(default_factory=AgentConfig)

    # PHASE 2: Information costs
    info_costs: InfoCostConfig = field(default_factory=InfoCostConfig)

    # Colony systems
    pheromone: PheromoneConfig = field(default_factory=PheromoneConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    physarum: PhysarumConfig = field(default_factory=PhysarumConfig)
    overmind: OvermindConfig = field(default_factory=OvermindConfig)

    # Cognitive architecture
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)

    # PHASE 3: Time-scale separation
    time_scales: TimeScaleConfig = field(default_factory=TimeScaleConfig)

    # Learning
    reward: RewardConfig = field(default_factory=RewardConfig)
    policy: PolicyNetworkConfig = field(default_factory=PolicyNetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Scenario
    scenario_name: str = "default"
    initial_water_level: float = 0.1
    initial_vegetation_density: float = 0.5
    stream_location: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    lodge_location: Optional[Tuple[int, int]] = None

    def validate(self):
        """Validate configuration consistency"""
        assert self.grid.grid_size > 0, "Grid size must be positive"
        assert self.n_beavers > 0, "Must have at least one beaver"
        assert 0 <= self.pheromone.evaporation_rate <= 1, "Evaporation rate must be in [0, 1]"
        assert self.policy.local_view_radius * 2 + 1 <= self.grid.grid_size, "View radius too large for grid"

        # Validate bounds
        for bounds in [self.overmind.pheromone_evap_bounds,
                       self.overmind.recruitment_decay_bounds,
                       self.overmind.dance_gain_bounds]:
            assert bounds[0] <= bounds[1], f"Invalid bounds: {bounds}"

        return True


def create_default_config() -> SimulationConfig:
    """Create default configuration"""
    return SimulationConfig()


def create_small_test_config() -> SimulationConfig:
    """Create small configuration for testing"""
    config = SimulationConfig()
    config.grid.grid_size = 16
    config.n_beavers = 5
    config.training.n_episodes = 100
    config.training.max_steps_per_episode = 100
    return config


def create_large_scale_config() -> SimulationConfig:
    """Create large-scale configuration for full experiments"""
    config = SimulationConfig()
    config.grid.grid_size = 128
    config.n_beavers = 50
    config.training.n_episodes = 50000
    config.training.n_parallel_envs = 8
    return config
