"""
MycoBeaver Simulator
=====================
A comprehensive multi-agent simulation for bio-inspired ecosystem engineering.

Inspired by:
- Beaver dam-building behavior
- Ant colony pheromone routing
- Bee waggle dance recruitment
- Physarum polycephalum adaptive networks
- Distributed cognitive architectures

This implementation follows the MycoBeaver Simulator Design Plan
and integrates concepts from the Distributed Cognitive Architecture
in Adversarial Information Environments framework.

Modules:
--------
- config: Configuration dataclasses and defaults
- environment: Gym-compatible grid environment with hydrology
- pheromone: Ant-style pheromone routing system
- projects: Bee-style waggle dance recruitment
- physarum: Physarum-inspired adaptive transport network
- overmind: Meta-controller with wisdom signals
- semantic: Ant-inspired knowledge graph with semantic entropy
- communication: Bee-inspired consensus and message passing
- policy: PyTorch neural network policies
- training: PPO-based training pipeline
- main: CLI and simulation runner

Example Usage:
--------------
>>> from mycobeaver import create_default_config, MycoBeaverEnv
>>> config = create_default_config()
>>> env = MycoBeaverEnv(config)
>>> obs, info = env.reset()
>>> actions = {f"agent_{i}": env.action_space[f"agent_{i}"].sample()
...            for i in range(config.n_beavers)}
>>> obs, rewards, terminated, truncated, info = env.step(actions)

Training:
---------
>>> from mycobeaver.training import train_mycobeaver
>>> history = train_mycobeaver(n_episodes=1000)
"""

__version__ = "1.0.0"
__author__ = "MycoBeaver Research Team"

# Configuration
from .config import (
    SimulationConfig,
    GridConfig,
    AgentConfig,
    PheromoneConfig,
    ProjectConfig,
    PhysarumConfig,
    OvermindConfig,
    RewardConfig,
    PolicyNetworkConfig,
    TrainingConfig,
    SemanticConfig,
    CommunicationConfig,
    create_default_config,
    create_small_test_config,
    create_large_scale_config,
    AgentRole,
    ActionType,
    ProjectType,
    ProjectStatus,
)

# Core environment
from .environment import (
    MycoBeaverEnv,
    GridState,
    AgentState,
    HydrologyEngine,
    VegetationEngine,
)

# Subsystems
from .pheromone import (
    PheromoneField,
    MultiPheromoneField,
)

from .projects import (
    ProjectManager,
    Project,
)

from .physarum import (
    PhysarumNetwork,
)

from .overmind import (
    Overmind,
    NeuralOvermind,
    WisdomState,
)

from .semantic import (
    SemanticGraph,
    ColonySemanticSystem,
    Vertex,
    Edge,
    VertexType,
    EdgeType,
)

from .communication import (
    CommunicationSystem,
    ColonyCommunicationHub,
    Message,
    MessageType,
    VectorClock,
    ConsensusProposal,
)

# Main runner utilities
from .main import (
    run_simulation,
    create_benchmark_config,
    visualize_simulation,
    print_config_summary,
)

# Optional PyTorch-dependent modules
try:
    from .policy import (
        ActorCritic,
        PolicyNetwork,
        ValueNetwork,
        MultiAgentPolicy,
        RolloutBuffer,
    )
    from .training import (
        PPOTrainer,
        TrainingMetrics,
        TrainingHistory,
        AblationStudy,
        train_mycobeaver,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    # Version
    "__version__",

    # Configuration
    "SimulationConfig",
    "GridConfig",
    "AgentConfig",
    "PheromoneConfig",
    "ProjectConfig",
    "PhysarumConfig",
    "OvermindConfig",
    "RewardConfig",
    "PolicyNetworkConfig",
    "TrainingConfig",
    "SemanticConfig",
    "CommunicationConfig",
    "create_default_config",
    "create_small_test_config",
    "create_large_scale_config",
    "AgentRole",
    "ActionType",
    "ProjectType",
    "ProjectStatus",

    # Environment
    "MycoBeaverEnv",
    "GridState",
    "AgentState",
    "HydrologyEngine",
    "VegetationEngine",

    # Pheromone
    "PheromoneField",
    "MultiPheromoneField",

    # Projects
    "ProjectManager",
    "Project",

    # Physarum
    "PhysarumNetwork",

    # Overmind
    "Overmind",
    "NeuralOvermind",
    "WisdomState",

    # Semantic
    "SemanticGraph",
    "ColonySemanticSystem",
    "Vertex",
    "Edge",
    "VertexType",
    "EdgeType",

    # Communication
    "CommunicationSystem",
    "ColonyCommunicationHub",
    "Message",
    "MessageType",
    "VectorClock",
    "ConsensusProposal",

    # Main
    "run_simulation",
    "create_benchmark_config",
    "visualize_simulation",
    "print_config_summary",

    # PyTorch (conditional)
    "TORCH_AVAILABLE",
]

# Add PyTorch exports if available
if TORCH_AVAILABLE:
    __all__.extend([
        "ActorCritic",
        "PolicyNetwork",
        "ValueNetwork",
        "MultiAgentPolicy",
        "RolloutBuffer",
        "PPOTrainer",
        "TrainingMetrics",
        "TrainingHistory",
        "AblationStudy",
        "train_mycobeaver",
    ])
