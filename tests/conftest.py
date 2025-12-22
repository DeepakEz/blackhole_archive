"""
Pytest configuration and shared fixtures for Blackhole Archive tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def grid_size():
    """Default grid size for tests"""
    return 16


@pytest.fixture
def small_grid_size():
    """Small grid size for fast tests"""
    return 8


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests"""
    return np.random.default_rng(42)


@pytest.fixture
def sample_grid_state(grid_size):
    """Create a sample grid state for testing"""
    from mycobeaver.environment import GridState

    state = GridState(
        elevation=np.random.rand(grid_size, grid_size) * 10,
        water_depth=np.random.rand(grid_size, grid_size) * 0.5,
        vegetation=np.random.rand(grid_size, grid_size),
        soil_moisture=np.random.rand(grid_size, grid_size),
        dam_grid=np.zeros((grid_size, grid_size)),
        dam_integrity=np.zeros((grid_size, grid_size)),
        lodge_grid=np.zeros((grid_size, grid_size)),
        agent_presence=np.zeros((grid_size, grid_size)),
        message_density=np.zeros((grid_size, grid_size)),
    )
    return state


# MycoBeaver specific fixtures

@pytest.fixture
def pheromone_config():
    """Default pheromone configuration"""
    from mycobeaver.config import PheromoneConfig
    return PheromoneConfig()


@pytest.fixture
def physarum_config():
    """Default physarum configuration"""
    from mycobeaver.config import PhysarumConfig
    return PhysarumConfig()


@pytest.fixture
def project_config():
    """Default project configuration"""
    from mycobeaver.config import ProjectConfig
    return ProjectConfig()


@pytest.fixture
def semantic_config():
    """Default semantic configuration"""
    from mycobeaver.config import SemanticConfig
    return SemanticConfig()


@pytest.fixture
def communication_config():
    """Default communication configuration"""
    from mycobeaver.config import CommunicationConfig
    return CommunicationConfig()


@pytest.fixture
def info_cost_config():
    """Default info cost configuration"""
    from mycobeaver.config import InfoCostConfig
    return InfoCostConfig()


@pytest.fixture
def pheromone_field(pheromone_config, grid_size, info_cost_config):
    """Create a pheromone field for testing"""
    from mycobeaver.pheromone import PheromoneField
    return PheromoneField(pheromone_config, grid_size, info_cost_config)


@pytest.fixture
def physarum_network(physarum_config, grid_size):
    """Create a physarum network for testing"""
    from mycobeaver.physarum import PhysarumNetwork
    return PhysarumNetwork(physarum_config, grid_size)


@pytest.fixture
def simulation_config():
    """Default simulation configuration"""
    from mycobeaver.config import SimulationConfig
    return SimulationConfig()


@pytest.fixture
def mock_agent_state():
    """Create a mock agent state for testing"""
    class MockAgentState:
        def __init__(self):
            self.info_energy = 100.0

        def can_afford_info(self, cost: float, min_threshold: float = 1.0) -> bool:
            return self.info_energy >= cost + min_threshold

        def spend_info(self, cost: float) -> bool:
            if self.can_afford_info(cost):
                self.info_energy -= cost
                return True
            return False

    return MockAgentState()
