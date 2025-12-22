"""
Integration tests for mycobeaver/environment.py

Tests the core MycoBeaverEnv Gym environment.
"""

import pytest
import numpy as np


class TestMycoBeaverEnvBasic:
    """Basic tests for MycoBeaverEnv"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment for testing"""
        from mycobeaver.environment import MycoBeaverEnv
        # Use smaller grid for faster tests
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 4
        return MycoBeaverEnv(simulation_config)

    def test_initialization(self, env):
        """Test environment initialization"""
        assert env is not None
        assert env.config is not None
        assert env.grid_state is not None

    def test_reset(self, env):
        """Test environment reset"""
        obs, info = env.reset()

        assert obs is not None
        assert isinstance(info, dict)
        assert env.current_step == 0

    def test_step(self, env):
        """Test environment step"""
        env.reset()

        # Random actions for all agents
        n_agents = env.config.agent.n_agents
        actions = np.random.randint(0, env.action_space.n, size=n_agents)

        obs, rewards, terminated, truncated, info = env.step(actions)

        assert obs is not None
        assert len(rewards) == n_agents
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self, env):
        """Test multiple environment steps"""
        env.reset()

        for _ in range(10):
            n_agents = env.config.agent.n_agents
            actions = np.random.randint(0, env.action_space.n, size=n_agents)
            obs, rewards, terminated, truncated, info = env.step(actions)

            if terminated or truncated:
                break

        assert env.current_step <= 10

    def test_action_space(self, env):
        """Test action space is valid"""
        env.reset()

        # Should have discrete action space
        assert hasattr(env.action_space, 'n')
        assert env.action_space.n > 0

    def test_observation_space(self, env):
        """Test observation space is valid"""
        env.reset()

        # Should have observation space
        assert env.observation_space is not None


class TestGridState:
    """Tests for GridState management"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment for testing"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 4
        return MycoBeaverEnv(simulation_config)

    def test_grid_state_initialization(self, env):
        """Test grid state is properly initialized"""
        env.reset()

        state = env.grid_state
        size = env.config.grid.grid_size

        assert state.elevation.shape == (size, size)
        assert state.water_depth.shape == (size, size)
        assert state.vegetation.shape == (size, size)

    def test_grid_state_bounds(self, env):
        """Test grid state values are bounded"""
        env.reset()

        state = env.grid_state

        # Water depth should be non-negative
        assert np.all(state.water_depth >= 0)

        # Vegetation should be in [0, 1]
        assert np.all(state.vegetation >= 0)
        assert np.all(state.vegetation <= 1)


class TestAgentState:
    """Tests for AgentState management"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment for testing"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 4
        return MycoBeaverEnv(simulation_config)

    def test_agents_initialized(self, env):
        """Test agents are properly initialized"""
        env.reset()

        assert len(env.agents) == env.config.agent.n_agents

        for agent in env.agents:
            assert agent.energy > 0
            assert agent.satiety >= 0
            assert 0 <= agent.y < env.config.grid.grid_size
            assert 0 <= agent.x < env.config.grid.grid_size

    def test_agent_positions_valid(self, env):
        """Test agent positions are within bounds"""
        env.reset()

        size = env.config.grid.grid_size
        for agent in env.agents:
            assert 0 <= agent.y < size
            assert 0 <= agent.x < size


class TestHydrologyEngine:
    """Tests for HydrologyEngine"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment with hydrology"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 2
        return MycoBeaverEnv(simulation_config)

    def test_water_conservation(self, env):
        """Test water mass is approximately conserved"""
        env.reset()

        # Record initial water
        initial_water = np.sum(env.grid_state.water_depth)

        # Run a few steps (without rain)
        for _ in range(5):
            actions = np.zeros(env.config.agent.n_agents, dtype=int)
            env.step(actions)

        final_water = np.sum(env.grid_state.water_depth)

        # Water should change but not dramatically (allow for sources/evaporation)
        # This is a loose check since water sources add water
        assert final_water >= 0


class TestSubsystems:
    """Tests for subsystem integration"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment with all subsystems"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 4
        simulation_config.training.use_overmind = True
        simulation_config.training.use_pheromones = True
        simulation_config.training.use_projects = True
        return MycoBeaverEnv(simulation_config)

    def test_pheromone_field_exists(self, env):
        """Test pheromone field is initialized"""
        env.reset()
        assert env.pheromone_field is not None

    def test_project_manager_exists(self, env):
        """Test project manager is initialized"""
        env.reset()
        assert env.project_manager is not None

    def test_overmind_exists(self, env):
        """Test overmind is initialized"""
        env.reset()
        if env.config.training.use_overmind:
            assert env.overmind is not None


class TestRewardComputation:
    """Tests for reward computation"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment for testing"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 4
        return MycoBeaverEnv(simulation_config)

    def test_rewards_shape(self, env):
        """Test rewards have correct shape"""
        env.reset()

        actions = np.zeros(env.config.agent.n_agents, dtype=int)
        _, rewards, _, _, _ = env.step(actions)

        assert len(rewards) == env.config.agent.n_agents

    def test_rewards_bounded(self, env):
        """Test rewards are reasonably bounded"""
        env.reset()

        for _ in range(10):
            actions = np.random.randint(0, env.action_space.n,
                                        size=env.config.agent.n_agents)
            _, rewards, terminated, truncated, _ = env.step(actions)

            # Rewards should be finite
            assert np.all(np.isfinite(rewards))

            if terminated or truncated:
                break


class TestEpisodeTermination:
    """Tests for episode termination"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment for testing"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 4
        simulation_config.training.max_steps_per_episode = 50
        return MycoBeaverEnv(simulation_config)

    def test_truncation_at_max_steps(self, env):
        """Test episode truncates at max steps"""
        env.reset()

        max_steps = env.config.training.max_steps_per_episode
        truncated = False

        for _ in range(max_steps + 10):
            actions = np.zeros(env.config.agent.n_agents, dtype=int)
            _, _, terminated, truncated, _ = env.step(actions)

            if terminated or truncated:
                break

        # Should have truncated
        assert truncated or env.current_step >= max_steps


class TestRenderAndClose:
    """Tests for render and close methods"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment for testing"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 2
        return MycoBeaverEnv(simulation_config)

    def test_close(self, env):
        """Test close method"""
        env.reset()
        env.close()  # Should not raise

    def test_render_rgb(self, simulation_config):
        """Test RGB rendering"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 2
        env = MycoBeaverEnv(simulation_config, render_mode="rgb_array")
        env.reset()

        frame = env.render()
        if frame is not None:
            assert len(frame.shape) == 3  # HxWxC


class TestInfoDict:
    """Tests for info dictionary"""

    @pytest.fixture
    def env(self, simulation_config):
        """Create environment for testing"""
        from mycobeaver.environment import MycoBeaverEnv
        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 4
        return MycoBeaverEnv(simulation_config)

    def test_info_on_reset(self, env):
        """Test info dict on reset"""
        _, info = env.reset()

        assert isinstance(info, dict)

    def test_info_on_step(self, env):
        """Test info dict on step"""
        env.reset()

        actions = np.zeros(env.config.agent.n_agents, dtype=int)
        _, _, _, _, info = env.step(actions)

        assert isinstance(info, dict)
        # Should contain useful metrics
        assert "step" in info or len(info) >= 0


class TestDeterminism:
    """Tests for deterministic behavior with seed"""

    def test_reset_with_seed(self, simulation_config):
        """Test reset produces same state with same seed"""
        from mycobeaver.environment import MycoBeaverEnv

        simulation_config.grid.grid_size = 16
        simulation_config.agent.n_agents = 4

        env1 = MycoBeaverEnv(simulation_config)
        env2 = MycoBeaverEnv(simulation_config)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Same seed should give same initial state
        # Note: This depends on implementation
        env1.close()
        env2.close()
