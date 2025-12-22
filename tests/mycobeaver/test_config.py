"""
Unit tests for mycobeaver/config.py

Tests configuration validation and default values.
"""

import pytest
from dataclasses import fields
from mycobeaver.config import (
    GridConfig, AgentConfig, InfoCostConfig, PheromoneConfig,
    ProjectConfig, PhysarumConfig, OvermindConfig, SemanticConfig,
    CommunicationConfig, MemoryConfig, TimeScaleConfig, RewardConfig,
    PolicyNetworkConfig, TrainingConfig, SimulationConfig,
    ProjectType, ProjectStatus
)


class TestEnums:
    """Tests for configuration enums"""

    def test_project_type_values(self):
        """Test ProjectType enum values"""
        assert ProjectType.DAM.value == "dam"
        assert ProjectType.LODGE.value == "lodge"
        assert ProjectType.CANAL.value == "canal"

    def test_project_status_values(self):
        """Test ProjectStatus enum values"""
        assert ProjectStatus.PROPOSED.value == "proposed"
        assert ProjectStatus.ACTIVE.value == "active"
        assert ProjectStatus.COMPLETED.value == "completed"
        assert ProjectStatus.ABANDONED.value == "abandoned"


class TestGridConfig:
    """Tests for GridConfig"""

    def test_default_values(self):
        """Test default grid configuration"""
        config = GridConfig()
        assert config.grid_size == 64
        assert config.n_water_sources >= 0

    def test_custom_values(self):
        """Test custom grid configuration"""
        config = GridConfig(grid_size=32, n_water_sources=5)
        assert config.grid_size == 32
        assert config.n_water_sources == 5


class TestAgentConfig:
    """Tests for AgentConfig"""

    def test_default_values(self):
        """Test default agent configuration"""
        config = AgentConfig()
        assert config.n_agents > 0
        assert config.max_energy > 0
        assert config.max_satiety > 0

    def test_energy_bounds(self):
        """Test energy configuration bounds"""
        config = AgentConfig()
        assert config.max_energy >= config.low_energy_threshold


class TestInfoCostConfig:
    """Tests for InfoCostConfig"""

    def test_default_values(self):
        """Test default info cost configuration"""
        config = InfoCostConfig()
        assert config.cost_send_message >= 0
        assert config.cost_broadcast >= 0
        assert config.cost_pheromone_deposit >= 0

    def test_broadcast_higher_than_send(self):
        """Test broadcast cost is higher than single send"""
        config = InfoCostConfig()
        assert config.cost_broadcast >= config.cost_send_message


class TestPheromoneConfig:
    """Tests for PheromoneConfig"""

    def test_default_values(self):
        """Test default pheromone configuration"""
        config = PheromoneConfig()
        assert 0 < config.evaporation_rate < 1
        assert config.min_pheromone >= 0
        assert config.max_pheromone > config.min_pheromone

    def test_alpha_beta_positive(self):
        """Test alpha and beta exponents are positive"""
        config = PheromoneConfig()
        assert config.pheromone_alpha >= 0
        assert config.heuristic_beta >= 0


class TestProjectConfig:
    """Tests for ProjectConfig"""

    def test_default_values(self):
        """Test default project configuration"""
        config = ProjectConfig()
        assert config.dance_gain > 0
        assert config.recruitment_decay > 0
        assert config.dam_wood_required > 0
        assert config.lodge_wood_required > 0

    def test_weight_sum(self):
        """Test quality weights are reasonable"""
        config = ProjectConfig()
        total = (config.resource_weight + config.hydro_impact_weight +
                 config.safety_weight + config.distance_cost_weight)
        assert total > 0


class TestPhysarumConfig:
    """Tests for PhysarumConfig"""

    def test_default_values(self):
        """Test default physarum configuration"""
        config = PhysarumConfig()
        assert config.initial_conductivity > 0
        assert config.min_conductivity >= 0
        assert config.max_conductivity > config.min_conductivity

    def test_adaptation_rates(self):
        """Test adaptation rates are positive"""
        config = PhysarumConfig()
        assert config.reinforcement_rate >= 0
        assert config.decay_rate >= 0


class TestOvermindConfig:
    """Tests for OvermindConfig"""

    def test_default_values(self):
        """Test default overmind configuration"""
        config = OvermindConfig()
        assert config.update_interval > 0
        assert 0 <= config.smoothing_factor <= 1

    def test_bound_ranges(self):
        """Test signal bounds are valid ranges"""
        config = OvermindConfig()
        # Bounds should be tuples with min < max
        for field in fields(config):
            if field.name.endswith('_bounds'):
                bounds = getattr(config, field.name)
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    assert bounds[0] <= bounds[1]


class TestSemanticConfig:
    """Tests for SemanticConfig"""

    def test_default_values(self):
        """Test default semantic configuration"""
        config = SemanticConfig()
        assert config.max_vertices > 0
        assert config.max_edges > 0
        assert config.vertex_decay_rate >= 0


class TestCommunicationConfig:
    """Tests for CommunicationConfig"""

    def test_default_values(self):
        """Test default communication configuration"""
        config = CommunicationConfig()
        assert config.max_bandwidth_per_step > 0
        assert config.max_message_queue > 0
        assert 0 < config.quorum_fraction <= 1

    def test_sync_interval_positive(self):
        """Test sync interval is positive"""
        config = CommunicationConfig()
        assert config.sync_interval > 0


class TestMemoryConfig:
    """Tests for MemoryConfig"""

    def test_default_values(self):
        """Test default memory configuration"""
        config = MemoryConfig()
        assert config.max_events > 0
        assert config.k_neighbors > 0


class TestTimeScaleConfig:
    """Tests for TimeScaleConfig"""

    def test_default_values(self):
        """Test default time scale configuration"""
        config = TimeScaleConfig()
        assert config.hydrology_substeps >= 1
        assert config.pheromone_update_interval > 0


class TestRewardConfig:
    """Tests for RewardConfig"""

    def test_default_values(self):
        """Test default reward configuration"""
        config = RewardConfig()
        # All reward weights should be finite
        for field in fields(config):
            if 'weight' in field.name or 'reward' in field.name or 'penalty' in field.name:
                value = getattr(config, field.name)
                if isinstance(value, (int, float)):
                    assert abs(value) < 1000  # Reasonable bound

    def test_curriculum_thresholds(self):
        """Test curriculum thresholds are valid"""
        config = RewardConfig()
        assert config.min_structures_for_global_reward >= 0


class TestPolicyNetworkConfig:
    """Tests for PolicyNetworkConfig"""

    def test_default_values(self):
        """Test default policy network configuration"""
        config = PolicyNetworkConfig()
        assert config.hidden_dim > 0
        assert config.n_heads > 0
        assert 0 < config.dropout < 1


class TestTrainingConfig:
    """Tests for TrainingConfig"""

    def test_default_values(self):
        """Test default training configuration"""
        config = TrainingConfig()
        assert config.n_episodes > 0
        assert config.max_steps_per_episode > 0
        assert config.learning_rate > 0
        assert 0 < config.gamma <= 1  # Discount factor

    def test_ppo_parameters(self):
        """Test PPO parameters are valid"""
        config = TrainingConfig()
        assert config.ppo_epochs > 0
        assert config.clip_epsilon > 0
        assert config.entropy_coef >= 0
        assert config.value_coef > 0


class TestSimulationConfig:
    """Tests for SimulationConfig (master config)"""

    def test_default_initialization(self):
        """Test default simulation configuration"""
        config = SimulationConfig()

        # Check all sub-configs are initialized
        assert isinstance(config.grid, GridConfig)
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.pheromone, PheromoneConfig)
        assert isinstance(config.project, ProjectConfig)
        assert isinstance(config.physarum, PhysarumConfig)
        assert isinstance(config.overmind, OvermindConfig)
        assert isinstance(config.semantic, SemanticConfig)
        assert isinstance(config.communication, CommunicationConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.time_scales, TimeScaleConfig)
        assert isinstance(config.reward, RewardConfig)
        assert isinstance(config.policy, PolicyNetworkConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_scenario_name(self):
        """Test scenario name configuration"""
        config = SimulationConfig(scenario_name="test_scenario")
        assert config.scenario_name == "test_scenario"

    def test_custom_subconfig(self):
        """Test custom sub-configuration"""
        custom_grid = GridConfig(grid_size=32)
        config = SimulationConfig(grid=custom_grid)
        assert config.grid.grid_size == 32


class TestConfigConsistency:
    """Tests for configuration consistency across modules"""

    def test_grid_size_consistency(self):
        """Test grid size is consistent"""
        config = SimulationConfig()
        # Grid size should be used consistently
        assert config.grid.grid_size > 0

    def test_n_agents_consistency(self):
        """Test n_agents is reasonable"""
        config = SimulationConfig()
        # Should have positive agents that fit in grid
        assert config.agent.n_agents > 0
        assert config.agent.n_agents <= config.grid.grid_size ** 2

    def test_time_scale_consistency(self):
        """Test time scales are compatible"""
        config = SimulationConfig()
        # Hydrology substeps should be reasonable
        assert config.time_scales.hydrology_substeps <= 10


class TestConfigSerialization:
    """Tests for config serialization (if applicable)"""

    def test_dataclass_fields(self):
        """Test all configs are proper dataclasses"""
        configs = [
            GridConfig, AgentConfig, InfoCostConfig, PheromoneConfig,
            ProjectConfig, PhysarumConfig, OvermindConfig, SemanticConfig,
            CommunicationConfig, MemoryConfig, TimeScaleConfig, RewardConfig,
            PolicyNetworkConfig, TrainingConfig, SimulationConfig
        ]

        for config_class in configs:
            # Should have fields
            assert len(fields(config_class)) > 0

    def test_config_repr(self):
        """Test configs have readable representation"""
        config = SimulationConfig()
        repr_str = repr(config)
        assert "SimulationConfig" in repr_str
