"""
AGENT PLASTICITY SYSTEM
Substrate for emergent intelligence through adaptive behavior

Three mechanisms for learning under adversarial pressure:
1. Threat Memory - Learn from deaths, remember danger patterns
2. Strategy Switching - Behavioral modes that shift based on experience
3. Social Learning - Copy behaviors from successful survivors

Without plasticity, selection pressure just kills - with it, it teaches.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque
from enum import Enum, auto
import logging

logger = logging.getLogger("Plasticity")


# =============================================================================
# STRATEGY SPACE
# =============================================================================

class Strategy(Enum):
    """Behavioral strategies agents can adopt"""
    # Exploration vs Exploitation
    EXPLORER = auto()      # Seek novelty, high risk/reward
    EXPLOITER = auto()     # Stick to known, low risk/reward

    # Risk tolerance
    BOLD = auto()          # Ignore threats, maximize productivity
    CAUTIOUS = auto()      # Avoid threats, sacrifice productivity

    # Social mode
    SOLO = auto()          # Independent action
    SOCIAL = auto()        # Follow/copy others

    # Energy management
    GREEDY = auto()        # Maximize immediate gain
    CONSERVING = auto()    # Minimize energy expenditure


@dataclass
class StrategyWeights:
    """Agent's current strategy mixture (not discrete choice)"""
    exploration: float = 0.5    # 0=exploit, 1=explore
    risk_tolerance: float = 0.5  # 0=cautious, 1=bold
    sociality: float = 0.3      # 0=solo, 1=social
    greed: float = 0.5          # 0=conserving, 1=greedy

    def as_vector(self) -> np.ndarray:
        return np.array([self.exploration, self.risk_tolerance,
                        self.sociality, self.greed])

    def from_vector(self, v: np.ndarray):
        self.exploration = np.clip(v[0], 0, 1)
        self.risk_tolerance = np.clip(v[1], 0, 1)
        self.sociality = np.clip(v[2], 0, 1)
        self.greed = np.clip(v[3], 0, 1)


# =============================================================================
# THREAT MEMORY (Collective & Individual)
# =============================================================================

@dataclass
class ThreatEvent:
    """Record of a threat occurrence"""
    position: np.ndarray          # Where it happened
    time: float                   # When it happened
    threat_type: str              # What kind of threat
    severity: float               # How much damage (0-1)
    agents_affected: int          # How many died/injured
    conditions: Dict[str, float]  # Environmental conditions at the time


class ThreatMemory:
    """
    Collective memory of threats - shared knowledge from deaths.

    Agents that die contribute their final state to collective memory,
    teaching survivors what to avoid.
    """

    def __init__(self, decay_rate: float = 0.01, max_memories: int = 100):
        self.memories: deque = deque(maxlen=max_memories)
        self.decay_rate = decay_rate

        # Spatial danger map (discretized)
        self.danger_grid: Dict[Tuple[int, int, int], float] = {}
        self.grid_resolution = 2.0  # Size of each grid cell

        # Threat type statistics
        self.threat_counts: Dict[str, int] = {}
        self.threat_lethality: Dict[str, float] = {}  # Deaths per occurrence

    def _position_to_grid(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert position to grid cell"""
        return tuple((position[1:4] / self.grid_resolution).astype(int))

    def record_death(self, agent, death_cause: str, conditions: Dict[str, float],
                    current_time: float):
        """Record an agent death to collective memory"""
        event = ThreatEvent(
            position=agent.position.copy(),
            time=current_time,
            threat_type=death_cause,
            severity=1.0,  # Death is max severity
            agents_affected=1,
            conditions=conditions
        )
        self.memories.append(event)

        # Update spatial danger map
        grid_cell = self._position_to_grid(agent.position)
        self.danger_grid[grid_cell] = self.danger_grid.get(grid_cell, 0) + 1.0

        # Update threat statistics
        self.threat_counts[death_cause] = self.threat_counts.get(death_cause, 0) + 1

        logger.debug(f"Recorded death: {death_cause} at {grid_cell}")

    def record_injury(self, agent, injury_cause: str, severity: float,
                     conditions: Dict[str, float], current_time: float):
        """Record an agent injury (non-fatal threat)"""
        event = ThreatEvent(
            position=agent.position.copy(),
            time=current_time,
            threat_type=injury_cause,
            severity=severity,
            agents_affected=1,
            conditions=conditions
        )
        self.memories.append(event)

        # Update spatial danger map (weighted by severity)
        grid_cell = self._position_to_grid(agent.position)
        self.danger_grid[grid_cell] = self.danger_grid.get(grid_cell, 0) + severity * 0.5

    def get_danger_level(self, position: np.ndarray) -> float:
        """Get remembered danger level at a position"""
        grid_cell = self._position_to_grid(position)

        # Check this cell and neighbors
        danger = 0.0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor = (grid_cell[0] + dx, grid_cell[1] + dy, grid_cell[2] + dz)
                    if neighbor in self.danger_grid:
                        # Distance-weighted danger
                        dist = abs(dx) + abs(dy) + abs(dz)
                        weight = 1.0 / (1.0 + dist)
                        danger += self.danger_grid[neighbor] * weight

        return danger

    def get_threat_avoidance_vector(self, position: np.ndarray) -> np.ndarray:
        """Get direction to move away from remembered threats"""
        avoidance = np.zeros(3)
        grid_cell = self._position_to_grid(position)

        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                for dz in [-2, -1, 0, 1, 2]:
                    neighbor = (grid_cell[0] + dx, grid_cell[1] + dy, grid_cell[2] + dz)
                    if neighbor in self.danger_grid:
                        danger = self.danger_grid[neighbor]
                        # Direction away from danger
                        direction = np.array([-dx, -dy, -dz], dtype=float)
                        if np.linalg.norm(direction) > 0:
                            direction /= np.linalg.norm(direction)
                        avoidance += danger * direction

        if np.linalg.norm(avoidance) > 0:
            avoidance /= np.linalg.norm(avoidance)

        return avoidance

    def decay(self, dt: float):
        """Memories fade over time"""
        # Decay spatial danger map
        cells_to_remove = []
        for cell, danger in self.danger_grid.items():
            new_danger = danger * (1.0 - self.decay_rate * dt)
            if new_danger < 0.01:
                cells_to_remove.append(cell)
            else:
                self.danger_grid[cell] = new_danger

        for cell in cells_to_remove:
            del self.danger_grid[cell]

    def get_most_dangerous_threats(self, n: int = 3) -> List[Tuple[str, int]]:
        """Get the N most common death causes"""
        sorted_threats = sorted(self.threat_counts.items(),
                               key=lambda x: x[1], reverse=True)
        return sorted_threats[:n]


# =============================================================================
# INDIVIDUAL PLASTICITY STATE
# =============================================================================

@dataclass
class PlasticityState:
    """
    Per-agent plasticity state - the substrate for individual adaptation.
    """
    # Strategy weights (continuous, not discrete)
    strategy: StrategyWeights = field(default_factory=StrategyWeights)

    # Personal threat memory (supplements collective)
    personal_threats: List[Tuple[np.ndarray, str, float]] = field(default_factory=list)
    threat_sensitivity: float = 0.5  # How much to weight threat memory

    # Learning rates (can evolve!)
    strategy_learning_rate: float = 0.1
    social_learning_rate: float = 0.2

    # Performance tracking for credit assignment
    recent_energy_history: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_contribution_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Social learning state
    role_model_id: Optional[str] = None
    role_model_strategy: Optional[StrategyWeights] = None

    # Adaptation counters
    strategy_switches: int = 0
    near_death_experiences: int = 0

    def get_performance_trend(self) -> float:
        """Calculate recent performance trend (-1 to 1)"""
        if len(self.recent_energy_history) < 5:
            return 0.0

        history = list(self.recent_energy_history)
        recent = np.mean(history[-5:])
        earlier = np.mean(history[:5])

        if earlier == 0:
            return 0.0
        return (recent - earlier) / max(abs(earlier), 0.1)


# =============================================================================
# SOCIAL LEARNING SYSTEM
# =============================================================================

class SocialLearningSystem:
    """
    Agents learn from successful survivors.

    Key insight: Don't just copy successful agents' positions,
    copy their STRATEGIES (how they behave).
    """

    def __init__(self, observation_radius: float = 10.0):
        self.observation_radius = observation_radius

        # Track agent performance for role model selection
        self.agent_fitness: Dict[str, float] = {}
        self.agent_strategies: Dict[str, StrategyWeights] = {}

    def update_agent_fitness(self, agent_id: str, fitness: float,
                            strategy: StrategyWeights):
        """Update fitness records for an agent"""
        # Exponential moving average of fitness
        alpha = 0.3
        if agent_id in self.agent_fitness:
            self.agent_fitness[agent_id] = (
                alpha * fitness + (1 - alpha) * self.agent_fitness[agent_id]
            )
        else:
            self.agent_fitness[agent_id] = fitness

        self.agent_strategies[agent_id] = strategy

    def remove_agent(self, agent_id: str):
        """Remove dead agent from fitness tracking"""
        self.agent_fitness.pop(agent_id, None)
        self.agent_strategies.pop(agent_id, None)

    def find_role_model(self, agent_id: str, agent_position: np.ndarray,
                       all_agents: List[Any]) -> Optional[str]:
        """Find a nearby high-performing agent to learn from"""
        best_model = None
        best_fitness = -float('inf')

        for other in all_agents:
            if other.id == agent_id:
                continue
            if other.state != "active":
                continue

            # Check distance
            dist = np.linalg.norm(agent_position[1:] - other.position[1:])
            if dist > self.observation_radius:
                continue

            # Check fitness
            other_fitness = self.agent_fitness.get(other.id, 0)
            if other_fitness > best_fitness:
                best_fitness = other_fitness
                best_model = other.id

        return best_model if best_fitness > 0 else None

    def learn_from_model(self, learner_state: PlasticityState,
                        model_id: str) -> bool:
        """Copy strategy from role model (with noise)"""
        if model_id not in self.agent_strategies:
            return False

        model_strategy = self.agent_strategies[model_id]
        learner_strategy = learner_state.strategy

        # Blend toward model's strategy
        lr = learner_state.social_learning_rate
        noise = 0.05  # Small random variation to maintain diversity

        new_vec = (
            (1 - lr) * learner_strategy.as_vector() +
            lr * model_strategy.as_vector() +
            noise * np.random.randn(4)
        )
        learner_strategy.from_vector(new_vec)

        learner_state.role_model_id = model_id
        learner_state.role_model_strategy = model_strategy

        return True


# =============================================================================
# STRATEGY ADAPTATION
# =============================================================================

class StrategyAdapter:
    """
    Adapts agent strategies based on experience.

    Key principle: Strategies that lead to survival and productivity
    should be reinforced; those that lead to death should be avoided.
    """

    def __init__(self):
        pass

    def adapt_to_threat(self, state: PlasticityState, threat_type: str,
                       severity: float):
        """Adapt strategy in response to threat/injury"""
        # Threats make agents more cautious
        state.strategy.risk_tolerance -= 0.1 * severity * state.strategy_learning_rate
        state.strategy.risk_tolerance = max(0.1, state.strategy.risk_tolerance)

        # Near-death experiences increase caution significantly
        if severity > 0.5:
            state.near_death_experiences += 1
            state.strategy.exploration -= 0.1  # Become more conservative
            state.strategy.sociality += 0.1    # Seek safety in numbers

        state.strategy_switches += 1

    def adapt_to_success(self, state: PlasticityState, reward: float):
        """Adapt strategy in response to success"""
        # Success reinforces current strategy
        # Small drift toward more exploration (successful agents can afford risk)
        if reward > 0.1:
            state.strategy.risk_tolerance += 0.02 * state.strategy_learning_rate
            state.strategy.exploration += 0.01 * state.strategy_learning_rate

        # Clip to valid range
        state.strategy.risk_tolerance = min(0.9, state.strategy.risk_tolerance)
        state.strategy.exploration = min(0.9, state.strategy.exploration)

    def adapt_to_stagnation(self, state: PlasticityState, duration: int):
        """Adapt when agent isn't making progress"""
        if duration > 10:
            # Increase exploration to break out of local optima
            state.strategy.exploration += 0.05 * state.strategy_learning_rate
            state.strategy.sociality += 0.03  # Look for role models
            state.strategy_switches += 1

    def get_behavior_modifiers(self, state: PlasticityState,
                               threat_memory: ThreatMemory,
                               position: np.ndarray) -> Dict[str, float]:
        """
        Get behavior modifiers based on plasticity state.

        Returns multipliers that modify base agent behavior.
        """
        modifiers = {}

        # Exploration modifier: affects probability of trying new areas
        modifiers['exploration_prob'] = 0.1 + 0.4 * state.strategy.exploration

        # Risk modifier: affects willingness to enter dangerous areas
        danger = threat_memory.get_danger_level(position)
        danger_threshold = 0.1 + 0.9 * state.strategy.risk_tolerance
        modifiers['danger_avoidance'] = min(1.0, danger / danger_threshold)

        # Social modifier: affects tendency to follow others
        modifiers['social_following'] = state.strategy.sociality

        # Energy management: affects work vs rest balance
        modifiers['work_intensity'] = 0.5 + 0.5 * state.strategy.greed

        # Threat avoidance direction
        if danger > 0.1 and state.strategy.risk_tolerance < 0.5:
            avoidance = threat_memory.get_threat_avoidance_vector(position)
            modifiers['avoidance_vector'] = avoidance
        else:
            modifiers['avoidance_vector'] = np.zeros(3)

        return modifiers


# =============================================================================
# MAIN PLASTICITY SYSTEM
# =============================================================================

class AgentPlasticitySystem:
    """
    Coordinates all plasticity mechanisms.

    This is the substrate that allows selection pressure to create
    emergent intelligence rather than just extinction.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Core systems
        self.threat_memory = ThreatMemory(
            decay_rate=self.config.get('threat_decay_rate', 0.005),
            max_memories=self.config.get('max_threat_memories', 200)
        )
        self.social_learning = SocialLearningSystem(
            observation_radius=self.config.get('social_radius', 10.0)
        )
        self.strategy_adapter = StrategyAdapter()

        # Per-agent plasticity states
        self.agent_states: Dict[str, PlasticityState] = {}

        # System metrics
        self.metrics = {
            'deaths_recorded': 0,
            'strategies_adapted': 0,
            'social_learning_events': 0,
            'near_death_experiences': 0
        }

    def initialize_agent(self, agent_id: str,
                        initial_strategy: Optional[StrategyWeights] = None):
        """Initialize plasticity state for a new agent"""
        if initial_strategy is None:
            # Random initial strategy (diversity is good!)
            initial_strategy = StrategyWeights(
                exploration=0.3 + 0.4 * np.random.random(),
                risk_tolerance=0.3 + 0.4 * np.random.random(),
                sociality=0.2 + 0.3 * np.random.random(),
                greed=0.4 + 0.2 * np.random.random()
            )

        self.agent_states[agent_id] = PlasticityState(strategy=initial_strategy)

    def on_agent_death(self, agent, death_cause: str,
                      conditions: Dict[str, float], current_time: float):
        """Called when an agent dies - records to collective memory"""
        self.threat_memory.record_death(agent, death_cause, conditions, current_time)
        self.social_learning.remove_agent(agent.id)

        # Remove plasticity state
        self.agent_states.pop(agent.id, None)

        self.metrics['deaths_recorded'] += 1
        logger.info(f"Agent {agent.id} died: {death_cause}")

    def on_agent_injured(self, agent, injury_cause: str, severity: float,
                        conditions: Dict[str, float], current_time: float):
        """Called when an agent is injured"""
        self.threat_memory.record_injury(agent, injury_cause, severity,
                                        conditions, current_time)

        # Adapt strategy in response to threat
        if agent.id in self.agent_states:
            state = self.agent_states[agent.id]
            self.strategy_adapter.adapt_to_threat(state, injury_cause, severity)
            self.metrics['strategies_adapted'] += 1

            if severity > 0.5:
                self.metrics['near_death_experiences'] += 1

    def on_agent_success(self, agent, reward: float):
        """Called when an agent achieves something"""
        if agent.id in self.agent_states:
            state = self.agent_states[agent.id]
            self.strategy_adapter.adapt_to_success(state, reward)
            state.recent_contribution_history.append(reward)

    def update_agent(self, agent, all_agents: List[Any], dt: float) -> Dict[str, float]:
        """
        Update plasticity for an agent and return behavior modifiers.

        Returns:
            Dict of modifiers that should influence agent behavior
        """
        if agent.id not in self.agent_states:
            self.initialize_agent(agent.id)

        state = self.agent_states[agent.id]

        # Track energy history
        state.recent_energy_history.append(agent.energy)

        # Check for stagnation
        trend = state.get_performance_trend()
        if trend < -0.1:
            # Declining performance - need to adapt
            self.strategy_adapter.adapt_to_stagnation(state, len(state.recent_energy_history))

        # Social learning: find and learn from role models
        if state.strategy.sociality > 0.3 and np.random.random() < 0.1:
            model = self.social_learning.find_role_model(
                agent.id, agent.position, all_agents
            )
            if model:
                self.social_learning.learn_from_model(state, model)
                self.metrics['social_learning_events'] += 1

        # Update fitness for social learning system
        fitness = agent.energy + getattr(agent, 'contribution_score', 0) * 0.5
        self.social_learning.update_agent_fitness(agent.id, fitness, state.strategy)

        # Get behavior modifiers
        modifiers = self.strategy_adapter.get_behavior_modifiers(
            state, self.threat_memory, agent.position
        )

        return modifiers

    def update_system(self, dt: float):
        """System-wide updates"""
        # Decay threat memory over time
        self.threat_memory.decay(dt)

    def get_agent_strategy(self, agent_id: str) -> Optional[StrategyWeights]:
        """Get an agent's current strategy"""
        if agent_id in self.agent_states:
            return self.agent_states[agent_id].strategy
        return None

    def get_dangerous_areas(self) -> Dict[Tuple, float]:
        """Get the spatial danger map for visualization"""
        return self.threat_memory.danger_grid.copy()

    def get_metrics(self) -> Dict:
        """Get system metrics"""
        return {
            **self.metrics,
            'active_agents': len(self.agent_states),
            'danger_zones': len(self.threat_memory.danger_grid),
            'top_threats': self.threat_memory.get_most_dangerous_threats(3)
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AGENT PLASTICITY SYSTEM - Test")
    print("=" * 80)

    # Initialize system
    plasticity = AgentPlasticitySystem()

    # Create mock agents
    class MockAgent:
        def __init__(self, id, position):
            self.id = id
            self.position = position
            self.energy = 1.0
            self.state = "active"
            self.contribution_score = 0.0

    agents = [
        MockAgent("ant_0", np.array([0, 5, 0, 0])),
        MockAgent("ant_1", np.array([0, 6, 1, 0])),
        MockAgent("ant_2", np.array([0, 4, -1, 0])),
    ]

    # Initialize agents
    for agent in agents:
        plasticity.initialize_agent(agent.id)

    print(f"\nInitialized {len(agents)} agents with random strategies:")
    for agent in agents:
        strategy = plasticity.get_agent_strategy(agent.id)
        print(f"  {agent.id}: explore={strategy.exploration:.2f}, "
              f"risk={strategy.risk_tolerance:.2f}, "
              f"social={strategy.sociality:.2f}")

    # Simulate some deaths
    print(f"\nSimulating deaths...")
    plasticity.on_agent_death(
        agents[0], "energy_drought",
        {'energy': 0.0, 'r': 5.0},
        current_time=10.0
    )

    # Check danger map
    danger = plasticity.threat_memory.get_danger_level(agents[1].position)
    print(f"\nDanger at ant_1's position: {danger:.3f}")

    # Get behavior modifiers
    modifiers = plasticity.update_agent(agents[1], agents[1:], dt=0.1)
    print(f"\nBehavior modifiers for ant_1:")
    for key, value in modifiers.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: [{value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f}]")
        else:
            print(f"  {key}: {value:.3f}")

    print(f"\n{'='*80}")
    print("Agent Plasticity System initialized successfully!")
    print(f"Metrics: {plasticity.get_metrics()}")
    print(f"{'='*80}")
