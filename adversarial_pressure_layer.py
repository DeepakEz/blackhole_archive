"""
ADVERSARIAL PRESSURE LAYER (APL)
Phase II: Selection pressure, trade-offs, non-stationarity, causal coupling

Design Goals:
1. Selection pressure: survival is no longer guaranteed
2. Trade-offs: every gain has a cost (energy, time, risk)
3. Non-stationarity: environment and threats change
4. Causal coupling: pressure changes all layers
5. Recoverability with scars: recovery is possible but costly

The APL acts as a "Dungeon Master" with rules, not random damage.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum, auto
from collections import deque
import logging

logger = logging.getLogger("APL")


# =============================================================================
# THREAT TAXONOMY
# =============================================================================

class ThreatCategory(Enum):
    """Categories of adversarial pressure"""
    RESOURCE = auto()      # Energy & materials
    STRUCTURAL = auto()    # Infrastructure attacks
    INFORMATION = auto()   # Semantic graph pressure
    COMMUNICATION = auto() # Packet/protocol pressure
    PREDATOR = auto()      # Agent survival pressure


class ThreatSeverity(Enum):
    """Severity levels for scheduling"""
    BACKGROUND = 1   # Always on, mild
    EVENT = 2        # Episodic shocks
    BOSS = 3         # Rare, high-impact


@dataclass
class ThreatEvent:
    """A specific adversarial event"""
    name: str
    category: ThreatCategory
    severity: ThreatSeverity
    cost: float  # Pressure budget cost
    duration: float  # How long the effect lasts
    cooldown: float  # Minimum time between occurrences

    # Effect parameters
    effect_strength: float = 1.0
    target_selector: str = "random"  # How to select targets
    cascade_to: List[ThreatCategory] = field(default_factory=list)

    # Tracking
    last_triggered: float = -1000.0
    times_triggered: int = 0


# =============================================================================
# STATE EXTENSIONS (New variables for Phase II)
# =============================================================================

@dataclass
class StructureState:
    """Extended state for structures under APL"""
    integrity: float = 1.0          # 0-1, health
    maintenance_due: float = 0.0    # Accumulated maintenance debt
    placement_risk: float = 0.0     # Flood/erosion susceptibility
    last_maintained: float = 0.0    # When last maintained

    def decay(self, dt: float, decay_rate: float = 0.01):
        """Structures degrade without maintenance"""
        self.integrity -= decay_rate * dt * (1.0 + self.maintenance_due)
        self.maintenance_due += 0.005 * dt
        self.integrity = max(0.0, min(1.0, self.integrity))

    def maintain(self, energy_cost: float, current_time: float) -> float:
        """Maintain structure, returns actual cost"""
        cost = energy_cost * (1.0 + self.maintenance_due)
        self.maintenance_due = 0.0
        self.integrity = min(1.0, self.integrity + 0.2)
        self.last_maintained = current_time
        return cost


@dataclass
class VertexState:
    """Extended state for semantic graph vertices under APL"""
    strength: float = 1.0           # Vertex stability
    confidence: float = 0.5         # Validation level
    validation_count: int = 0       # Times validated by different sources
    decay_timer: float = 0.0        # Time since last reinforcement
    is_noise: bool = False          # Was this injected noise?

    def decay(self, dt: float, base_rate: float = 0.02):
        """Vertices decay without reinforcement"""
        self.decay_timer += dt
        decay_factor = base_rate * (1.0 + 0.1 * self.decay_timer)
        self.strength -= decay_factor * dt
        self.confidence -= 0.5 * decay_factor * dt
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))

    def reinforce(self, amount: float = 0.1):
        """Reinforce vertex from agent interaction"""
        self.strength = min(1.0, self.strength + amount)
        self.confidence = min(1.0, self.confidence + 0.5 * amount)
        self.validation_count += 1
        self.decay_timer = 0.0


@dataclass
class EdgeState:
    """Extended state for semantic graph edges under APL"""
    strength: float = 1.0
    reliability: float = 1.0  # How often packets succeed on this edge
    congestion: float = 0.0   # Current congestion level

    def decay(self, dt: float, base_rate: float = 0.03):
        self.strength -= base_rate * dt
        self.strength = max(0.0, min(1.0, self.strength))


@dataclass
class PacketState:
    """Extended state for packets under APL"""
    priority: int = 1           # 1-5, higher = more important
    ttl: float = 100.0          # Time to live
    redundancy_level: int = 1   # Copies in flight
    error_rate: float = 0.0     # Accumulated error probability
    hops: int = 0               # Number of hops taken

    def age(self, dt: float):
        self.ttl -= dt
        self.error_rate = min(0.9, self.error_rate + 0.01 * self.hops)


@dataclass
class AgentState:
    """Extended state for agents under APL"""
    fatigue: float = 0.0        # 0-1, affects efficiency
    injury: float = 0.0         # 0-1, affects survival
    role_switch_cost: float = 0.0  # Penalty for changing roles
    stress: float = 0.0         # Affects decision quality

    def accumulate_fatigue(self, work_done: float, dt: float):
        """Fatigue increases with work"""
        self.fatigue += 0.01 * work_done * dt
        self.fatigue = min(1.0, self.fatigue)

    def rest(self, dt: float, rest_rate: float = 0.05):
        """Recovery during low activity"""
        self.fatigue -= rest_rate * dt
        self.injury -= 0.5 * rest_rate * dt
        self.fatigue = max(0.0, self.fatigue)
        self.injury = max(0.0, self.injury)

    def efficiency(self) -> float:
        """Current efficiency based on fatigue/injury"""
        return max(0.1, 1.0 - 0.5 * self.fatigue - 0.3 * self.injury)


# =============================================================================
# APL CORE: PRESSURE BUDGET & THREAT MANAGEMENT
# =============================================================================

class AdversarialPressureLayer:
    """
    The APL observes system state and generates adversarial pressure.

    Pressure Budget increases when system is "too stable":
    - 100% survival for too long
    - Low entropy in decisions
    - Plateaued learning
    - Energy rises while work rises

    Pressure Budget decreases when system is near collapse:
    - Rising mortality
    - Energy crash
    - Packet backlog exploding
    - Graph fragmentation
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config

        # Pressure budget system
        self.pressure_budget = 5.0  # Start LOW - ramp up gradually
        self.max_pressure_budget = 50.0  # Cap lower to prevent wipeout
        self.min_pressure_budget = 0.5

        # Threat level (EMA)
        self.threat_level = 0.1  # Start low
        self.threat_ema_alpha = 0.1

        # Emergency safety
        self.emergency_mode = False
        self.emergency_cooldown = 0.0

        # History tracking
        self.shock_history: deque = deque(maxlen=50)
        self.recovery_cooldown = 0.0

        # System state tracking
        self.stability_score = 0.5
        self.exploit_detector = ExploitDetector()

        # Initialize threat library
        self.threats = self._initialize_threats()

        # Active effects
        self.active_effects: Dict[str, Dict] = {}

        # Metrics
        self.metrics = {
            'total_pressure_applied': 0.0,
            'events_triggered': 0,
            'structures_damaged': 0,
            'vertices_decayed': 0,
            'packets_lost': 0,
            'agents_injured': 0,
            'near_collapses': 0,
            'recovery_count': 0
        }

    def _initialize_threats(self) -> Dict[str, ThreatEvent]:
        """Initialize the threat library"""
        threats = {}

        # =====================================================================
        # A) RESOURCE ADVERSARIES
        # =====================================================================

        threats['energy_drought'] = ThreatEvent(
            name="Energy Drought",
            category=ThreatCategory.RESOURCE,
            severity=ThreatSeverity.EVENT,
            cost=15.0,
            duration=20.0,
            cooldown=50.0,  # Longer cooldown
            effect_strength=0.2,  # Was 50% - reduced to 20%
            cascade_to=[ThreatCategory.STRUCTURAL, ThreatCategory.PREDATOR]
        )

        threats['materials_scarcity'] = ThreatEvent(
            name="Materials Scarcity",
            category=ThreatCategory.RESOURCE,
            severity=ThreatSeverity.EVENT,
            cost=12.0,
            duration=25.0,
            cooldown=35.0,
            effect_strength=0.6,
        )

        threats['maintenance_tax'] = ThreatEvent(
            name="Maintenance Tax",
            category=ThreatCategory.RESOURCE,
            severity=ThreatSeverity.BACKGROUND,
            cost=2.0,
            duration=float('inf'),  # Always on
            cooldown=0.0,
            effect_strength=0.005,  # 0.5% per time unit (was 2%!)
        )

        # =====================================================================
        # B) STRUCTURAL ADVERSARIES
        # =====================================================================

        threats['flood_event'] = ThreatEvent(
            name="Flood / Dam Break",
            category=ThreatCategory.STRUCTURAL,
            severity=ThreatSeverity.EVENT,
            cost=20.0,
            duration=5.0,
            cooldown=80.0,  # Longer cooldown
            effect_strength=0.1,  # Was 30% - reduced to 10%
            target_selector="low_placement",
            cascade_to=[ThreatCategory.RESOURCE, ThreatCategory.COMMUNICATION]
        )

        threats['erosion'] = ThreatEvent(
            name="Erosion",
            category=ThreatCategory.STRUCTURAL,
            severity=ThreatSeverity.BACKGROUND,
            cost=1.0,
            duration=float('inf'),
            cooldown=0.0,
            effect_strength=0.003,  # Was 0.01 - too aggressive
        )

        threats['structural_decay'] = ThreatEvent(
            name="Structural Decay",
            category=ThreatCategory.STRUCTURAL,
            severity=ThreatSeverity.BACKGROUND,
            cost=1.5,
            duration=float('inf'),
            cooldown=0.0,
            effect_strength=0.004,  # Was 0.015 - too aggressive
        )

        # =====================================================================
        # C) INFORMATION ADVERSARIES
        # =====================================================================

        threats['concept_drift'] = ThreatEvent(
            name="Concept Drift",
            category=ThreatCategory.INFORMATION,
            severity=ThreatSeverity.EVENT,
            cost=18.0,
            duration=30.0,
            cooldown=40.0,
            effect_strength=0.4,  # 40% of semantics shift
            cascade_to=[ThreatCategory.COMMUNICATION]
        )

        threats['noise_injection'] = ThreatEvent(
            name="Noise Injection",
            category=ThreatCategory.INFORMATION,
            severity=ThreatSeverity.BACKGROUND,
            cost=3.0,
            duration=float('inf'),
            cooldown=0.0,
            effect_strength=0.01,  # Was 5% - reduced to 1%
        )

        threats['semantic_decay'] = ThreatEvent(
            name="Semantic Decay",
            category=ThreatCategory.INFORMATION,
            severity=ThreatSeverity.BACKGROUND,
            cost=2.0,
            duration=float('inf'),
            cooldown=0.0,
            effect_strength=0.005,  # Was 0.02 - reduced to 0.5%
        )

        threats['contradiction_storm'] = ThreatEvent(
            name="Contradiction Storm",
            category=ThreatCategory.INFORMATION,
            severity=ThreatSeverity.BOSS,
            cost=30.0,
            duration=15.0,
            cooldown=80.0,
            effect_strength=0.5,
            cascade_to=[ThreatCategory.COMMUNICATION, ThreatCategory.STRUCTURAL]
        )

        # =====================================================================
        # D) COMMUNICATION ADVERSARIES
        # =====================================================================

        threats['congestion'] = ThreatEvent(
            name="Congestion",
            category=ThreatCategory.COMMUNICATION,
            severity=ThreatSeverity.BACKGROUND,
            cost=2.0,
            duration=float('inf'),
            cooldown=0.0,
            effect_strength=0.02,  # Was 0.1 - reduced to 2%
        )

        threats['packet_loss'] = ThreatEvent(
            name="Packet Loss",
            category=ThreatCategory.COMMUNICATION,
            severity=ThreatSeverity.EVENT,
            cost=10.0,
            duration=15.0,
            cooldown=25.0,
            effect_strength=0.2,  # 20% drop rate
        )

        threats['route_jamming'] = ThreatEvent(
            name="Route Jamming",
            category=ThreatCategory.COMMUNICATION,
            severity=ThreatSeverity.EVENT,
            cost=12.0,
            duration=10.0,
            cooldown=30.0,
            effect_strength=0.3,
            target_selector="high_traffic"
        )

        # =====================================================================
        # E) PREDATOR PRESSURE
        # =====================================================================

        threats['mortality_zone'] = ThreatEvent(
            name="Mortality Risk Zone",
            category=ThreatCategory.PREDATOR,
            severity=ThreatSeverity.EVENT,
            cost=15.0,
            duration=20.0,
            cooldown=60.0,  # Longer cooldown
            effect_strength=0.03,  # Was 0.1 - reduced to 3%
            target_selector="spatial"
        )

        threats['fatigue_wave'] = ThreatEvent(
            name="Fatigue Wave",
            category=ThreatCategory.PREDATOR,
            severity=ThreatSeverity.EVENT,
            cost=8.0,
            duration=25.0,
            cooldown=50.0,  # Longer cooldown
            effect_strength=0.1,  # Was 0.3 - reduced to 10%
        )

        threats['epidemic'] = ThreatEvent(
            name="Localized Epidemic",
            category=ThreatCategory.PREDATOR,
            severity=ThreatSeverity.BOSS,
            cost=25.0,
            duration=30.0,
            cooldown=150.0,  # Much longer cooldown
            effect_strength=0.15,  # Was 0.4 - reduced to 15%
            target_selector="cluster",
            cascade_to=[ThreatCategory.RESOURCE]
        )

        return threats

    # =========================================================================
    # PRESSURE BUDGET MANAGEMENT
    # =========================================================================

    def update_pressure_budget(self, system_state: Dict) -> float:
        """
        Update pressure budget based on system state.

        Increases when too stable, decreases when near collapse.
        CRITICAL: Must respond FAST to prevent wipeout.
        """
        # Extract metrics
        survival_rate = system_state.get('survival_rate', 1.0)
        energy_trend = system_state.get('energy_trend', 0.0)  # Positive = gaining
        work_efficiency = system_state.get('work_efficiency', 1.0)
        behavioral_entropy = system_state.get('behavioral_entropy', 0.5)
        graph_health = system_state.get('graph_health', 0.5)
        packet_backlog = system_state.get('packet_backlog', 0)

        # =====================================================================
        # EMERGENCY SHUTOFF - Immediate response to critical survival
        # =====================================================================
        if survival_rate < 0.5:
            # CRITICAL: More than half dead - emergency mode
            self.emergency_mode = True
            self.emergency_cooldown = 50.0  # Long cooldown to let system recover
            self.pressure_budget = self.min_pressure_budget
            self.threat_level = 0.05
            self.metrics['near_collapses'] += 1
            logger.warning(f"APL EMERGENCY MODE: survival={survival_rate:.2f}, shutting down pressure")
            return self.pressure_budget

        if survival_rate < 0.7:
            # Severe stress - rapidly reduce pressure
            budget_delta = -10.0 * (0.7 - survival_rate)
            self.pressure_budget += budget_delta
            self.pressure_budget = max(self.min_pressure_budget, self.pressure_budget)
            self.metrics['near_collapses'] += 1
            logger.info(f"APL severe stress: survival={survival_rate:.2f}, budget={self.pressure_budget:.1f}")
            return self.pressure_budget

        # Handle emergency cooldown
        if self.emergency_mode:
            self.emergency_cooldown -= 1.0
            if self.emergency_cooldown <= 0 and survival_rate > 0.8:
                self.emergency_mode = False
                logger.info("APL exiting emergency mode")
            else:
                # Stay in low-pressure mode
                return self.pressure_budget

        # =====================================================================
        # NORMAL OPERATION
        # =====================================================================

        # Calculate stability score
        stability_factors = [
            survival_rate,
            0.5 + 0.5 * np.tanh(energy_trend),  # Normalized energy trend
            min(1.0, work_efficiency),
            1.0 - behavioral_entropy,  # Low entropy = high stability
            graph_health
        ]
        self.stability_score = np.mean(stability_factors)

        # Calculate collapse risk - be more sensitive
        collapse_factors = [
            1.0 - survival_rate,
            max(0, -energy_trend) / 5.0,  # Energy crash (more sensitive)
            min(1.0, packet_backlog / 50.0),  # Backlog (lower threshold)
            1.0 - graph_health  # Graph fragmentation
        ]
        collapse_risk = np.mean(collapse_factors)

        # Budget adjustment - slower increases, faster decreases
        if collapse_risk > 0.3:  # Lower threshold for reducing pressure
            # Reduce pressure proportionally
            budget_delta = -5.0 * collapse_risk
            self.metrics['near_collapses'] += 1
        elif self.stability_score > 0.85:  # Higher threshold for increasing
            # Very stable - slowly increase pressure
            budget_delta = 1.0 * (self.stability_score - 0.7)
        else:
            # Normal - maintain or slight decrease for safety
            budget_delta = 0.2 * (self.stability_score - 0.5)

        self.pressure_budget += budget_delta
        self.pressure_budget = np.clip(
            self.pressure_budget,
            self.min_pressure_budget,
            self.max_pressure_budget
        )

        # Update threat level EMA
        self.threat_level = (
            self.threat_ema_alpha * (self.pressure_budget / self.max_pressure_budget) +
            (1 - self.threat_ema_alpha) * self.threat_level
        )

        return self.pressure_budget

    # =========================================================================
    # THREAT SELECTION & TRIGGERING
    # =========================================================================

    def select_threats(self, current_time: float, system_state: Dict) -> List[str]:
        """
        Select which threats to trigger based on budget and need.

        score(event) = need(event) * novelty(event) * safety(event)
        """
        selected = []
        available_budget = self.pressure_budget

        # Detect current exploits
        exploits = self.exploit_detector.detect(system_state)

        for threat_name, threat in self.threats.items():
            # Check cooldown
            if current_time - threat.last_triggered < threat.cooldown:
                continue

            # Check budget
            if threat.cost > available_budget:
                continue

            # Calculate selection score
            need = self._calculate_need(threat, exploits, system_state)
            novelty = self._calculate_novelty(threat, current_time)
            safety = self._calculate_safety(threat, system_state)

            score = need * novelty * safety

            # Probabilistic selection based on score
            if np.random.random() < score * 0.3:
                selected.append(threat_name)
                available_budget -= threat.cost

        return selected

    def _calculate_need(self, threat: ThreatEvent, exploits: List[str],
                       system_state: Dict) -> float:
        """Calculate how much this threat addresses current exploits"""
        need = 0.3  # Base need

        # Match threats to exploits
        exploit_matches = {
            'early_build_plateau': [ThreatCategory.STRUCTURAL, ThreatCategory.RESOURCE],
            'graph_clique': [ThreatCategory.INFORMATION],
            'idle_agents': [ThreatCategory.PREDATOR],
            'packet_spam': [ThreatCategory.COMMUNICATION],
            'energy_hoarding': [ThreatCategory.RESOURCE],
        }

        for exploit in exploits:
            if exploit in exploit_matches:
                if threat.category in exploit_matches[exploit]:
                    need += 0.3

        return min(1.0, need)

    def _calculate_novelty(self, threat: ThreatEvent, current_time: float) -> float:
        """Avoid repeating the same pressure pattern"""
        # Check recent history
        recent_same_category = sum(
            1 for event in self.shock_history
            if event['category'] == threat.category
        )

        novelty = 1.0 - 0.1 * recent_same_category

        # Time since last trigger
        time_factor = min(1.0, (current_time - threat.last_triggered) / 100.0)
        novelty *= (0.5 + 0.5 * time_factor)

        return max(0.1, novelty)

    def _calculate_safety(self, threat: ThreatEvent, system_state: Dict) -> float:
        """
        Avoid total wipe - keep pressure learnable.

        CRITICAL: Must scale ALL threats based on survival, not just BOSS/EVENT.
        """
        survival_rate = system_state.get('survival_rate', 1.0)
        energy_ratio = system_state.get('energy_ratio', 0.5)

        # Emergency mode - almost no threats
        if self.emergency_mode:
            return 0.01  # Only 1% chance for any threat

        # CRITICAL survival thresholds - scale ALL threats
        if survival_rate < 0.6:
            # Below 60% survival - almost no threats
            return 0.05

        if survival_rate < 0.7:
            # 60-70% survival - minimal threats
            if threat.severity == ThreatSeverity.BOSS:
                return 0.0  # No boss events
            if threat.severity == ThreatSeverity.EVENT:
                return 0.1
            # Background still at 0.2
            return 0.2

        if survival_rate < 0.8:
            # 70-80% survival - reduced threats
            if threat.severity == ThreatSeverity.BOSS:
                return 0.1
            if threat.severity == ThreatSeverity.EVENT:
                return 0.3
            # Background at 0.5
            return 0.5

        if survival_rate < 0.9:
            # 80-90% survival - light reduction
            if threat.severity == ThreatSeverity.BOSS:
                return 0.3
            if threat.severity == ThreatSeverity.EVENT:
                return 0.6
            return 0.8

        # Energy-based reduction (on top of survival)
        safety = 1.0
        if energy_ratio < 0.3:
            if threat.category == ThreatCategory.RESOURCE:
                safety *= 0.3
            elif threat.category == ThreatCategory.PREDATOR:
                safety *= 0.5

        return safety

    def trigger_threat(self, threat_name: str, current_time: float,
                      system_state: Dict) -> Dict:
        """
        Trigger a specific threat and return the effects to apply.
        """
        if threat_name not in self.threats:
            return {}

        threat = self.threats[threat_name]
        threat.last_triggered = current_time
        threat.times_triggered += 1

        # Record in history
        self.shock_history.append({
            'name': threat_name,
            'category': threat.category,
            'time': current_time,
            'severity': threat.severity
        })

        # Generate effects
        effects = {
            'name': threat_name,
            'category': threat.category,
            'severity': threat.severity,
            'strength': threat.effect_strength,
            'duration': threat.duration,
            'start_time': current_time,
            'target_selector': threat.target_selector,
            'cascade_to': threat.cascade_to
        }

        # Add to active effects
        self.active_effects[threat_name] = effects

        self.metrics['events_triggered'] += 1
        self.metrics['total_pressure_applied'] += threat.cost

        logger.info(f"APL triggered: {threat_name} (strength={threat.effect_strength:.2f})")

        return effects

    # =========================================================================
    # EFFECT APPLICATION
    # =========================================================================

    def apply_effects(self, current_time: float, spacetime, semantic_graph,
                     agents: Dict, dt: float, system_state: Optional[Dict] = None) -> Dict:
        """
        Apply all active effects to the system.
        Returns damage report.

        CRITICAL: Effects are scaled by survival rate to prevent wipeout.
        """
        damage_report = {
            'structures_damaged': 0,
            'vertices_decayed': 0,
            'edges_weakened': 0,
            'packets_lost': 0,
            'agents_affected': 0,
            'energy_drained': 0.0
        }

        # Get survival rate for effect scaling
        survival_rate = 1.0
        if system_state:
            survival_rate = system_state.get('survival_rate', 1.0)

        # Emergency mode - suspend ALL effects
        if self.emergency_mode:
            logger.info("APL emergency mode - all effects suspended")
            return damage_report

        # Calculate effect scaling based on survival
        # This ensures effects weaken as the system struggles
        if survival_rate < 0.5:
            effect_scale = 0.0  # No effects below 50%
        elif survival_rate < 0.7:
            effect_scale = 0.2  # 20% effect below 70%
        elif survival_rate < 0.8:
            effect_scale = 0.5  # 50% effect below 80%
        elif survival_rate < 0.9:
            effect_scale = 0.75  # 75% effect below 90%
        else:
            effect_scale = 1.0  # Full effect above 90%

        # Process each active effect
        expired = []
        for name, effect in self.active_effects.items():
            # Check expiration
            if current_time - effect['start_time'] > effect['duration']:
                expired.append(name)
                continue

            # Apply based on category
            category = effect['category']
            scaled_strength = effect['strength'] * effect_scale  # SCALE by survival!

            # Skip if effectively no strength
            if scaled_strength < 0.001:
                continue

            # Create a scaled effect copy for the apply functions
            scaled_effect = effect.copy()
            scaled_effect['strength'] = scaled_strength

            if category == ThreatCategory.RESOURCE:
                damage = self._apply_resource_effect(scaled_effect, spacetime, agents, dt)
                damage_report['energy_drained'] += damage

            elif category == ThreatCategory.STRUCTURAL:
                damage = self._apply_structural_effect(scaled_effect, spacetime, dt)
                damage_report['structures_damaged'] += damage

            elif category == ThreatCategory.INFORMATION:
                v_damage, e_damage = self._apply_information_effect(
                    scaled_effect, semantic_graph, dt
                )
                damage_report['vertices_decayed'] += v_damage
                damage_report['edges_weakened'] += e_damage

            elif category == ThreatCategory.COMMUNICATION:
                damage = self._apply_communication_effect(scaled_effect, semantic_graph, dt)
                damage_report['packets_lost'] += damage

            elif category == ThreatCategory.PREDATOR:
                damage = self._apply_predator_effect(scaled_effect, agents, dt)
                damage_report['agents_affected'] += damage

        # Remove expired effects
        for name in expired:
            del self.active_effects[name]
            logger.info(f"APL effect expired: {name}")

        # Update metrics
        self.metrics['structures_damaged'] += damage_report['structures_damaged']
        self.metrics['vertices_decayed'] += damage_report['vertices_decayed']
        self.metrics['packets_lost'] += damage_report['packets_lost']
        self.metrics['agents_injured'] += damage_report['agents_affected']

        return damage_report

    def _apply_resource_effect(self, effect: Dict, spacetime, agents: Dict,
                               dt: float) -> float:
        """Apply resource pressure effects"""
        total_drain = 0.0
        strength = effect['strength']
        name = effect['name']

        if 'drought' in name.lower():
            # Reduce energy regeneration / increase consumption
            for colony_agents in agents.values():
                for agent in colony_agents:
                    if agent.state == "active":
                        drain = strength * 0.01 * dt
                        agent.energy -= drain
                        total_drain += drain

        elif 'maintenance' in name.lower():
            # Maintenance tax - structures cost energy
            if hasattr(spacetime, 'structural_field'):
                structure_count = np.sum(spacetime.structural_field > 0.1)
                tax = strength * structure_count * 0.001 * dt
                # Distribute tax across agents
                for colony_agents in agents.values():
                    for agent in colony_agents:
                        if agent.state == "active":
                            agent.energy -= tax / max(1, len(agents))
                            total_drain += tax / max(1, len(agents))

        return total_drain

    def _apply_structural_effect(self, effect: Dict, spacetime, dt: float) -> int:
        """Apply structural damage effects"""
        damaged = 0
        strength = effect['strength']
        name = effect['name']

        if not hasattr(spacetime, 'structural_field'):
            return 0

        if 'flood' in name.lower():
            # Damage structures in vulnerable positions (low r values)
            vulnerable_mask = spacetime.structural_field > 0.1
            # Structures closer to horizon are more vulnerable
            r_weights = 1.0 / (spacetime.r[:, None, None] + 1)
            damage_prob = strength * r_weights * vulnerable_mask

            damage_mask = np.random.random(damage_prob.shape) < damage_prob
            spacetime.structural_field[damage_mask] *= 0.5
            damaged = int(np.sum(damage_mask))

        elif 'erosion' in name.lower() or 'decay' in name.lower():
            # Gradual decay of all structures
            decay = strength * dt
            spacetime.structural_field *= (1.0 - decay)
            damaged = int(np.sum(spacetime.structural_field > 0.01))

        return damaged

    def _apply_information_effect(self, effect: Dict, semantic_graph,
                                  dt: float) -> Tuple[int, int]:
        """Apply information/semantic pressure effects"""
        vertices_affected = 0
        edges_affected = 0
        strength = effect['strength']
        name = effect['name']

        if not hasattr(semantic_graph, 'graph'):
            return 0, 0

        if 'drift' in name.lower():
            # Concept drift - shift edge weights and vertex positions
            for v in semantic_graph.graph.nodes():
                if np.random.random() < strength * dt:
                    # Shift vertex properties
                    if 'position' in semantic_graph.graph.nodes[v]:
                        pos = semantic_graph.graph.nodes[v]['position']
                        drift = 0.1 * strength * np.random.randn(len(pos))
                        semantic_graph.graph.nodes[v]['position'] = pos + drift
                        vertices_affected += 1

            for u, v in semantic_graph.graph.edges():
                if np.random.random() < strength * dt:
                    # Weaken edges
                    if (u, v) in semantic_graph.pheromones:
                        semantic_graph.pheromones[(u, v)] *= (1.0 - strength)
                        edges_affected += 1

        elif 'noise' in name.lower():
            # Inject noise vertices
            if np.random.random() < strength * dt * 10:
                # Create a noise vertex
                if hasattr(semantic_graph, 'add_vertex'):
                    pos = np.random.randn(4) * 5
                    pos[0] = 0  # Time component
                    pos[1] = max(3, pos[1])  # Keep outside horizon
                    vid = semantic_graph.add_vertex(pos, salience=0.1)
                    # Mark as noise if we have extended state
                    semantic_graph.graph.nodes[vid]['is_noise'] = True
                    vertices_affected += 1

        elif 'decay' in name.lower():
            # Semantic decay
            decay_rate = strength * dt
            to_remove = []
            for v in semantic_graph.graph.nodes():
                salience = semantic_graph.graph.nodes[v].get('salience', 0.5)
                new_salience = salience * (1.0 - decay_rate)
                semantic_graph.graph.nodes[v]['salience'] = new_salience
                if new_salience < 0.05:
                    to_remove.append(v)
                    vertices_affected += 1

        elif 'contradiction' in name.lower():
            # Contradiction storm - create conflicting subgraphs
            vertices = list(semantic_graph.graph.nodes())
            if len(vertices) > 10:
                # Select random vertices to contradict
                n_contradict = int(len(vertices) * strength)
                contradicted = np.random.choice(vertices,
                                               min(n_contradict, len(vertices)),
                                               replace=False)
                for v in contradicted:
                    # Weaken all edges from this vertex
                    for u in semantic_graph.graph.neighbors(v):
                        if (v, u) in semantic_graph.pheromones:
                            semantic_graph.pheromones[(v, u)] *= 0.3
                            edges_affected += 1
                    vertices_affected += 1

        return vertices_affected, edges_affected

    def _apply_communication_effect(self, effect: Dict, semantic_graph,
                                    dt: float) -> int:
        """Apply communication pressure effects"""
        packets_affected = 0
        strength = effect['strength']
        name = effect['name']

        if not hasattr(semantic_graph, 'packet_queues'):
            return 0

        if 'congestion' in name.lower():
            # Increase effective queue delay (handled by graph)
            pass  # Congestion effect is tracked in edge states

        elif 'loss' in name.lower():
            # Drop packets probabilistically
            for v, queue in semantic_graph.packet_queues.items():
                to_remove = []
                for i, packet in enumerate(queue):
                    if np.random.random() < strength * dt:
                        to_remove.append(i)
                        packets_affected += 1
                # Remove in reverse order
                for i in reversed(to_remove):
                    queue.pop(i)

        elif 'jamming' in name.lower():
            # Block certain edges temporarily
            edges = list(semantic_graph.graph.edges())
            n_jam = int(len(edges) * strength)
            jammed = np.random.choice(len(edges), min(n_jam, len(edges)),
                                     replace=False)
            for idx in jammed:
                edge = edges[idx]
                if edge in semantic_graph.pheromones:
                    semantic_graph.pheromones[edge] = 0.01
                    packets_affected += 1

        return packets_affected

    def _apply_predator_effect(self, effect: Dict, agents: Dict,
                               dt: float) -> int:
        """Apply predator/survival pressure effects"""
        agents_affected = 0
        strength = effect['strength']
        name = effect['name']

        if 'mortality' in name.lower():
            # Apply mortality risk to agents in certain zones
            for colony_agents in agents.values():
                for agent in colony_agents:
                    if agent.state == "active":
                        # Risk increases near horizon
                        if hasattr(agent, 'position') and len(agent.position) > 1:
                            r = agent.position[1]
                            risk = strength * (3.0 / max(3.0, r))
                            if np.random.random() < risk * dt:
                                agent.energy -= 0.2
                                agents_affected += 1

        elif 'fatigue' in name.lower():
            # Increase fatigue across agents
            for colony_agents in agents.values():
                for agent in colony_agents:
                    if agent.state == "active":
                        if hasattr(agent, 'apl_state'):
                            agent.apl_state.fatigue += strength * dt
                        agents_affected += 1

        elif 'epidemic' in name.lower():
            # Affect a cluster of agents
            all_agents = [a for agents_list in agents.values()
                         for a in agents_list if a.state == "active"]
            if len(all_agents) > 0:
                # Pick a random agent as epicenter
                epicenter = np.random.choice(all_agents)
                epi_pos = epicenter.position if hasattr(epicenter, 'position') else np.zeros(4)

                # Affect nearby agents
                for agent in all_agents:
                    if hasattr(agent, 'position'):
                        dist = np.linalg.norm(agent.position - epi_pos)
                        if dist < 5.0:  # Epidemic radius
                            infection_prob = strength * (1.0 - dist / 5.0)
                            if np.random.random() < infection_prob:
                                agent.energy -= 0.3
                                agents_affected += 1

        return agents_affected

    # =========================================================================
    # MAIN UPDATE LOOP
    # =========================================================================

    def update(self, current_time: float, dt: float, spacetime, semantic_graph,
               agents: Dict, system_state: Dict) -> Dict:
        """
        Main APL update - called each simulation step.

        Returns:
            Dict with damage report and triggered events
        """
        # Update pressure budget
        self.update_pressure_budget(system_state)

        # Select and trigger new threats
        selected_threats = self.select_threats(current_time, system_state)
        triggered = []
        for threat_name in selected_threats:
            effects = self.trigger_threat(threat_name, current_time, system_state)
            if effects:
                triggered.append(threat_name)

        # Apply all active effects (scaled by survival!)
        damage_report = self.apply_effects(
            current_time, spacetime, semantic_graph, agents, dt, system_state
        )

        # Check for recovery state
        if damage_report['agents_affected'] > 0 or damage_report['structures_damaged'] > 0:
            self.recovery_cooldown = 10.0  # Give system time to recover
        else:
            self.recovery_cooldown = max(0, self.recovery_cooldown - dt)

        return {
            'pressure_budget': self.pressure_budget,
            'threat_level': self.threat_level,
            'stability_score': self.stability_score,
            'triggered_events': triggered,
            'active_effects': list(self.active_effects.keys()),
            'damage_report': damage_report,
            'in_recovery': self.recovery_cooldown > 0
        }

    def get_metrics(self) -> Dict:
        """Return APL metrics for analysis"""
        return {
            **self.metrics,
            'current_budget': self.pressure_budget,
            'current_threat_level': self.threat_level,
            'active_effects_count': len(self.active_effects),
            'shock_history_count': len(self.shock_history)
        }


# =============================================================================
# EXPLOIT DETECTOR
# =============================================================================

class ExploitDetector:
    """Detects exploitative behaviors that APL should counter"""

    def __init__(self):
        self.history = {
            'build_rate': deque(maxlen=20),
            'energy_history': deque(maxlen=20),
            'vertex_count': deque(maxlen=20),
            'packet_count': deque(maxlen=20)
        }

    def detect(self, system_state: Dict) -> List[str]:
        """Detect current exploits"""
        exploits = []

        # Update history
        self.history['build_rate'].append(system_state.get('build_rate', 0))
        self.history['energy_history'].append(system_state.get('total_energy', 0))
        self.history['vertex_count'].append(system_state.get('n_vertices', 0))
        self.history['packet_count'].append(system_state.get('n_packets', 0))

        # Detect: Early build plateau (build fast then coast)
        if len(self.history['build_rate']) >= 10:
            recent_builds = list(self.history['build_rate'])[-5:]
            earlier_builds = list(self.history['build_rate'])[:5]
            if np.mean(earlier_builds) > 10 and np.mean(recent_builds) < 2:
                exploits.append('early_build_plateau')

        # Detect: Graph clique (dense but not diverse)
        n_vertices = system_state.get('n_vertices', 0)
        n_edges = system_state.get('n_edges', 0)
        if n_vertices > 0:
            edge_density = n_edges / (n_vertices * (n_vertices - 1) + 1)
            if edge_density > 0.5 and n_vertices < 50:
                exploits.append('graph_clique')

        # Detect: Idle agents (high survival but low productivity)
        survival_rate = system_state.get('survival_rate', 1.0)
        productivity = system_state.get('work_per_agent', 0)
        if survival_rate > 0.95 and productivity < 0.1:
            exploits.append('idle_agents')

        # Detect: Energy hoarding (energy rises while work rises)
        if len(self.history['energy_history']) >= 10:
            energy_trend = np.mean(np.diff(list(self.history['energy_history'])))
            if energy_trend > 1.0:
                exploits.append('energy_hoarding')

        return exploits


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ADVERSARIAL PRESSURE LAYER - Test")
    print("=" * 80)

    # Initialize APL
    apl = AdversarialPressureLayer()

    # Mock system state
    system_state = {
        'survival_rate': 1.0,
        'energy_trend': 0.5,
        'work_efficiency': 1.2,
        'behavioral_entropy': 0.3,
        'graph_health': 0.8,
        'packet_backlog': 5,
        'energy_ratio': 0.7,
        'build_rate': 15,
        'total_energy': 350,
        'n_vertices': 25,
        'n_edges': 100,
        'n_packets': 10,
        'work_per_agent': 0.5
    }

    print(f"\nInitial state:")
    print(f"  Pressure Budget: {apl.pressure_budget:.1f}")
    print(f"  Threat Level: {apl.threat_level:.3f}")

    # Simulate a few updates
    for t in range(10):
        # Update budget
        apl.update_pressure_budget(system_state)

        # Select threats
        threats = apl.select_threats(float(t * 10), system_state)

        print(f"\nt={t*10}:")
        print(f"  Budget: {apl.pressure_budget:.1f}")
        print(f"  Stability: {apl.stability_score:.3f}")
        print(f"  Threats selected: {threats}")

        # Simulate slight degradation
        system_state['survival_rate'] *= 0.99
        system_state['energy_trend'] -= 0.1

    print(f"\n{'='*80}")
    print("APL initialized successfully!")
    print(f"Total threats defined: {len(apl.threats)}")
    print(f"{'='*80}")
