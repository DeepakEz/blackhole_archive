"""
MycoBeaver Architectural Contracts
===================================
PHASE 1: Strict interface contracts that define component boundaries.

These contracts are NON-NEGOTIABLE. They prevent instability by enforcing
clear separation of concerns between system components.

Key Principle: Overmind is a MODULATOR, not a CONTROLLER.
- It adjusts rates, budgets, and constraints
- It NEVER touches rewards, policies, or agent actions directly
"""

from typing import TypedDict, Protocol, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# =============================================================================
# OVERMIND CONTRACT - The most critical boundary
# =============================================================================

class OvermindSignals(TypedDict):
    """
    STRICT INTERFACE: The ONLY way Overmind communicates with the system.

    Overmind IS allowed to control:
    - PPO learning rate (global or per-agent group)
    - Entropy coefficient (exploration pressure)
    - Communication bandwidth (max messages / step)
    - Pheromone evaporation & diffusion rates
    - Project quorum thresholds
    - Physarum adaptation speed

    Overmind is NOT allowed to:
    - Modify rewards directly
    - Modify policy logits
    - Inject gradients
    - Change agent actions

    This TypedDict is READ-ONLY for all consumers.
    Only Overmind can produce it.
    """
    # Training rate modulation
    lr_scale: float           # Multiplier for base learning rate [0.1, 10.0]
    entropy_scale: float      # Multiplier for entropy coefficient [0.1, 10.0]

    # Communication budget
    comm_budget: int          # Max messages per step per agent [1, 100]

    # Pheromone dynamics
    pheromone_decay: float    # Evaporation rate [0.01, 0.5]
    pheromone_diffusion: float  # Diffusion rate [0.0, 0.3]

    # Physarum dynamics
    physarum_tau: float       # Adaptation time constant [0.1, 10.0]
    physarum_flow_exp: float  # Flow reinforcement exponent [0.5, 2.0]

    # Project recruitment
    quorum_multiplier: float  # Quorum threshold multiplier [0.5, 2.0]
    recruitment_decay: float  # Signal decay rate [0.01, 0.5]
    dance_gain: float         # Waggle dance signal strength [0.1, 5.0]

    # Semantic system
    semantic_temperature: float  # Exploration vs exploitation [0.1, 10.0]


# Default signal values - system starts here
DEFAULT_OVERMIND_SIGNALS: OvermindSignals = {
    "lr_scale": 1.0,
    "entropy_scale": 1.0,
    "comm_budget": 10,
    "pheromone_decay": 0.1,
    "pheromone_diffusion": 0.05,
    "physarum_tau": 1.0,
    "physarum_flow_exp": 1.0,
    "quorum_multiplier": 1.0,
    "recruitment_decay": 0.1,
    "dance_gain": 1.0,
    "semantic_temperature": 1.0,
}


# Signal bounds for validation
SIGNAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "lr_scale": (0.1, 10.0),
    "entropy_scale": (0.1, 10.0),
    "comm_budget": (1, 100),
    "pheromone_decay": (0.01, 0.5),
    "pheromone_diffusion": (0.0, 0.3),
    "physarum_tau": (0.1, 10.0),
    "physarum_flow_exp": (0.5, 2.0),
    "quorum_multiplier": (0.5, 2.0),
    "recruitment_decay": (0.01, 0.5),
    "dance_gain": (0.1, 5.0),
    "semantic_temperature": (0.1, 10.0),
}


def validate_signals(signals: OvermindSignals) -> Tuple[bool, str]:
    """
    Validate that signals are within allowed bounds.

    Returns:
        (is_valid, error_message)
    """
    for key, (low, high) in SIGNAL_BOUNDS.items():
        value = signals.get(key)
        if value is None:
            return False, f"Missing required signal: {key}"
        if not (low <= value <= high):
            return False, f"Signal {key}={value} out of bounds [{low}, {high}]"
    return True, ""


def clamp_signals(signals: OvermindSignals) -> OvermindSignals:
    """
    Clamp signals to valid bounds (safety mechanism).
    """
    clamped = {}
    for key, value in signals.items():
        if key in SIGNAL_BOUNDS:
            low, high = SIGNAL_BOUNDS[key]
            if key == "comm_budget":
                clamped[key] = int(np.clip(value, low, high))
            else:
                clamped[key] = float(np.clip(value, low, high))
        else:
            clamped[key] = value
    return clamped


# =============================================================================
# OBSERVATION CONTRACTS - What each component can see
# =============================================================================

class GlobalObservation(TypedDict):
    """
    Global state observation available to Overmind.
    This is what Overmind bases its modulation decisions on.
    """
    # Water statistics
    water_mean: float
    water_variance: float
    water_max: float
    flood_fraction: float
    drought_fraction: float

    # Colony health
    n_alive: int
    n_total: int
    avg_energy: float
    avg_satiety: float

    # Infrastructure
    n_structures: int
    n_active_projects: int
    n_completed_projects: int

    # Network health
    avg_pheromone: float
    avg_conductivity: float
    semantic_coherence: float

    # Historical (for trend detection)
    reward_trend: float  # Smoothed reward derivative
    survival_trend: float


class AgentObservation(TypedDict):
    """
    Local observation for a single agent.
    Agents CANNOT see global state directly.
    """
    local_grid: np.ndarray      # (C, H, W) local view
    global_features: np.ndarray  # Broadcast signals (recruitment, wisdom)
    internal_state: np.ndarray   # Own energy, satiety, etc.


# =============================================================================
# ACTION CONTRACTS - What each component can do
# =============================================================================

class AgentActionSpace(Enum):
    """
    Enumeration of valid agent actions.
    Agents can ONLY take these discrete actions.
    """
    STAY = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_EAST = 3
    MOVE_WEST = 4
    MOVE_NE = 5
    MOVE_NW = 6
    MOVE_SE = 7
    MOVE_SW = 8
    REST = 9
    FORAGE = 10
    BUILD_DAM = 11
    BUILD_LODGE = 12
    CARRY_RESOURCE = 13
    DROP_RESOURCE = 14
    ADVERTISE_PROJECT = 15


# =============================================================================
# PROTOCOL DEFINITIONS - Interface contracts
# =============================================================================

class SignalProducer(Protocol):
    """
    Protocol for components that produce OvermindSignals.
    Only Overmind should implement this.
    """
    def produce_signals(self, observation: GlobalObservation) -> OvermindSignals:
        """Produce modulation signals from global observation."""
        ...


class SignalConsumer(Protocol):
    """
    Protocol for components that consume OvermindSignals.
    These components READ signals but never modify them.
    """
    def apply_signals(self, signals: OvermindSignals) -> None:
        """Apply modulation signals. Read-only access."""
        ...


class RewardComputer(Protocol):
    """
    Protocol for reward computation.

    CRITICAL: Overmind CANNOT implement this protocol.
    Rewards are computed by environment logic only.
    """
    def compute_reward(
        self,
        state_before: Any,
        action: int,
        state_after: Any
    ) -> float:
        """Compute reward for a transition. Pure function of states."""
        ...


# =============================================================================
# IMMUTABLE REWARD STRUCTURE
# =============================================================================

@dataclass(frozen=True)
class RewardStructure:
    """
    FROZEN dataclass - reward weights cannot be modified at runtime.

    This ensures Overmind cannot manipulate rewards indirectly
    by adjusting weights. All values are set at initialization.
    """
    # Individual rewards
    alive_per_step: float = 0.01
    death_penalty: float = -1.0
    forage_reward: float = 0.05
    build_action: float = 0.1

    # Global rewards
    dam_completion: float = 5.0
    lodge_completion: float = 10.0
    wetland_cell: float = 0.001
    vegetation_multiplier: float = 0.01

    # Penalties
    flood_penalty_per_cell: float = 0.01
    drought_penalty_per_cell: float = 0.005

    # Mixing weights
    individual_weight: float = 0.3
    global_weight: float = 0.7


# =============================================================================
# VALIDATION GUARDS
# =============================================================================

class ContractViolationError(Exception):
    """Raised when an architectural contract is violated."""
    pass


def assert_overmind_cannot_modify_rewards(
    reward_before: float,
    reward_after: float,
    signals: OvermindSignals
) -> None:
    """
    Guard: Verify that Overmind signals did not modify rewards.
    Call this in debug mode to catch violations.
    """
    if abs(reward_before - reward_after) > 1e-10:
        raise ContractViolationError(
            f"Overmind signals caused reward modification: "
            f"{reward_before} -> {reward_after}. "
            f"This violates the architectural contract."
        )


def assert_overmind_cannot_modify_actions(
    action_logits_before: np.ndarray,
    action_logits_after: np.ndarray,
    signals: OvermindSignals
) -> None:
    """
    Guard: Verify that Overmind signals did not modify action logits.
    """
    if not np.allclose(action_logits_before, action_logits_after, atol=1e-10):
        raise ContractViolationError(
            f"Overmind signals caused action logit modification. "
            f"This violates the architectural contract."
        )


# =============================================================================
# SIGNAL ROUTER - Safe distribution of Overmind signals
# =============================================================================

@dataclass
class SignalRouter:
    """
    Routes Overmind signals to consumers with validation.

    This is the ONLY way signals should flow from Overmind to the system.
    Direct access is prohibited.
    """
    _current_signals: OvermindSignals = field(
        default_factory=lambda: DEFAULT_OVERMIND_SIGNALS.copy()
    )
    _locked: bool = False

    def update_signals(self, new_signals: OvermindSignals) -> None:
        """
        Update signals from Overmind.

        Only callable by Overmind, only when not locked.
        """
        if self._locked:
            raise ContractViolationError(
                "Cannot update signals while router is locked"
            )

        # Validate
        valid, error = validate_signals(new_signals)
        if not valid:
            # Clamp instead of failing (graceful degradation)
            new_signals = clamp_signals(new_signals)

        self._current_signals = new_signals

    def get_lr_scale(self) -> float:
        """Get learning rate scale. Read-only."""
        return self._current_signals["lr_scale"]

    def get_entropy_scale(self) -> float:
        """Get entropy coefficient scale. Read-only."""
        return self._current_signals["entropy_scale"]

    def get_comm_budget(self) -> int:
        """Get communication budget. Read-only."""
        return self._current_signals["comm_budget"]

    def get_pheromone_params(self) -> Tuple[float, float]:
        """Get pheromone decay and diffusion. Read-only."""
        return (
            self._current_signals["pheromone_decay"],
            self._current_signals["pheromone_diffusion"]
        )

    def get_physarum_params(self) -> Tuple[float, float]:
        """Get physarum tau and flow exponent. Read-only."""
        return (
            self._current_signals["physarum_tau"],
            self._current_signals["physarum_flow_exp"]
        )

    def get_project_params(self) -> Tuple[float, float, float]:
        """Get quorum, decay, and dance gain. Read-only."""
        return (
            self._current_signals["quorum_multiplier"],
            self._current_signals["recruitment_decay"],
            self._current_signals["dance_gain"]
        )

    def get_semantic_temperature(self) -> float:
        """Get semantic temperature. Read-only."""
        return self._current_signals["semantic_temperature"]

    def lock(self) -> None:
        """Lock router during critical sections."""
        self._locked = True

    def unlock(self) -> None:
        """Unlock router after critical sections."""
        self._locked = False

    def get_all_signals(self) -> OvermindSignals:
        """Get copy of all signals. For logging/debugging only."""
        return self._current_signals.copy()
