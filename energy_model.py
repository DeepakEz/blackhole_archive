"""
BLACKHOLE ARCHIVE: Unified Energy Model
========================================
Shared thermodynamic constants across all simulation engines.
Ensures consistent energy economics - no perpetual motion.

Design Philosophy:
------------------
This module enforces thermodynamic consistency throughout the simulation.
All energy flows must satisfy:
1. Conservation: Energy in = Energy out + Work done
2. Second Law: Processes have entropy cost (irreversibility)
3. No perpetual motion: Cannot gain more than invested

Physical Analogies:
------------------
The simulation uses biological and physical analogies:
- Agents are like cells with metabolic costs
- Structures are like physical infrastructure (dams, nests)
- Packets are like molecules with transport costs
- Pheromones/fields decay like chemical gradients
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EnergyConstants:
    """
    Central energy constants for the simulation.

    All values are calibrated to maintain thermodynamic consistency:
    - Energy inputs (rewards) < Energy outputs (costs) for any cycle
    - Decay rates set to reasonable timescales (τ ~ 10-100 timesteps)
    - Thresholds set to balance activity vs. conservation

    Derivation Notes:
    -----------------
    Values were calibrated through simulation experiments to achieve:
    1. Agents survive 200-400 timesteps without external energy
    2. Structures persist for ~100 timesteps without maintenance
    3. Packet transport is marginally profitable (encourages coordination)
    4. Build probability creates interesting spatial patterns
    """

    # =========================================================================
    # MOVEMENT COSTS (per dt)
    # =========================================================================
    # Physical basis: Kinetic energy + friction losses
    # E_move ∝ m * v² * μ where μ is effective friction coefficient
    #
    # Calibration: Agent should deplete 1.0 energy over ~400 timesteps
    # of continuous movement: 1.0 / 400 ≈ 0.0025
    MOVE_COST_BASE: float = 0.0025

    # With packet: Additional mass increases kinetic energy
    # E_packet = E_base * (1 + m_packet/m_agent)
    # Assuming packet is 60% of agent mass: 0.0025 * 1.6 ≈ 0.004
    MOVE_COST_WITH_PACKET: float = 0.004

    # =========================================================================
    # BEAVER CONSTRUCTION
    # =========================================================================
    # Physical basis: Work required to modify spacetime/build structure
    # Net cost ensures building is an investment, not free

    # BUILD_COST: Energy required for construction
    # Sized so agent can build ~15 structures before depletion
    # 1.0 energy / 0.06 per build ≈ 16 builds
    BUILD_COST: float = 0.06

    # BUILD_REWARD: Partial energy recovery from successful construction
    # Represents efficiency gains from learning, material reuse
    # Net cost = 0.06 - 0.02 = 0.04 per structure
    # This ensures building is costly but not punitive
    BUILD_REWARD: float = 0.02

    # BUILD_COOLDOWN: Recovery time between builds (time units)
    # Physical basis: Material preparation, structural curing
    # At dt=0.01: 0.75 / 0.01 = 75 timesteps ≈ 0.75 simulation time units
    BUILD_COOLDOWN: float = 0.75

    # =========================================================================
    # BUILD TRIGGERS (decision thresholds)
    # =========================================================================

    # CURVATURE_THRESHOLD: Minimum curvature to trigger building
    # Physical basis: Build where spacetime needs stabilization
    # Value 0.01 corresponds to r ≈ 6M (moderate proximity to horizon)
    # K = 48M²/r⁶, sqrt(K) = 0.01 → r ≈ 6M
    CURVATURE_THRESHOLD: float = 0.01

    # ENERGY_THRESHOLD: Minimum energy to attempt building
    # Must cover BUILD_COST with margin: 0.06 + 0.02 margin = 0.08
    ENERGY_THRESHOLD: float = 0.08

    # LOCAL_SATURATION_MAX: Maximum structures in neighborhood
    # Prevents over-building in one location (spatial efficiency)
    LOCAL_SATURATION_MAX: int = 3

    # BUILD_PROBABILITY: Stochastic gate for build attempts
    # Introduces randomness to prevent synchronized building
    # P = 0.35 gives geometric distribution with mean 1/0.35 ≈ 3 attempts
    BUILD_PROBABILITY: float = 0.35

    # =========================================================================
    # PACKET ECONOMICS
    # =========================================================================
    # Design: Packet transport is marginally profitable to encourage
    # coordination, but not so profitable that it dominates other activities

    # Pickup cost: Energy to acquire packet
    PACKET_PICKUP_COST: float = 0.01

    # Delivery reward: Energy gained on successful delivery
    # Net profit = 0.015 - 0.01 = 0.005 per packet (before transport costs)
    # With transport costs (~0.004 per step), profitable over short distances
    PACKET_DELIVERY_REWARD: float = 0.015

    # =========================================================================
    # GLOBAL MATERIAL BUDGET
    # =========================================================================
    # Represents finite resources for construction (exotic matter, etc.)

    # Initial budget: Enough for ~500 structures
    MATERIAL_BUDGET_INITIAL: float = 500.0

    # Cost per structure: 1 unit of material
    MATERIAL_COST_PER_BUILD: float = 1.0

    # =========================================================================
    # DECAY RATES (per dt)
    # =========================================================================
    # Physical basis: Exponential decay N(t) = N₀ * exp(-λt)
    # Half-life τ = ln(2)/λ ≈ 0.693/λ

    # Structural field decay: Structures degrade without maintenance
    # τ = 0.693/0.01 = 69 timesteps (at dt=0.01: 0.69 time units)
    # Structures need maintenance every ~50-100 timesteps
    STRUCTURAL_FIELD_DECAY: float = 0.01

    # Pheromone decay: Chemical signals evaporate
    # τ = 0.693/0.1 = 6.9 timesteps - rapid decay for responsiveness
    # Matches biological pheromone timescales (minutes in real ants)
    PHEROMONE_DECAY: float = 0.1

    # Epistemic stress decay: Contradictions can be resolved
    # τ = 0.693/0.05 = 13.8 timesteps - moderate persistence
    EPISTEMIC_STRESS_DECAY: float = 0.05

    # =========================================================================
    # PACKET THRESHOLDS
    # =========================================================================

    # Pickup salience threshold: Minimum vertex importance for packet pickup
    # Lower value increases packet flow; higher value focuses on important data
    PICKUP_SALIENCE_THRESHOLD: float = 0.45

    # High priority threshold: Packets above this get expedited handling
    HIGH_PRIORITY_THRESHOLD: float = 0.75

    # Packet TTL: Maximum lifetime before packet expires
    # At dt=0.01: 50.0 / 0.01 = 5000 timesteps = 50 time units
    # Packets must be delivered within reasonable time or are lost
    PACKET_TTL: float = 50.0

    # =========================================================================
    # QUEUE LIMITS
    # =========================================================================

    # Maximum packets per queue: Prevents infinite buffering
    MAX_QUEUE_SIZE: int = 10


# Singleton instance
ENERGY = EnergyConstants()


@dataclass
class InstrumentationCounters:
    """
    Per-step counters for debugging.
    Answers: "Why didn't it build/transport?"
    """

    # Beaver build counters
    build_attempts: int = 0
    build_success: int = 0
    blocked_by_curvature: int = 0
    blocked_by_energy: int = 0
    blocked_by_cooldown: int = 0
    blocked_by_saturation: int = 0
    blocked_by_materials: int = 0

    # Bee packet counters
    pickup_attempts: int = 0
    pickup_success: int = 0
    pickup_blocked_no_vertices: int = 0
    pickup_blocked_low_salience: int = 0
    pickup_blocked_no_packets: int = 0

    deliver_attempts: int = 0
    deliver_enqueued: int = 0
    deliver_blocked_not_at_wormhole: int = 0
    deliver_blocked_protocol_full: int = 0
    deliver_blocked_ttl_expired: int = 0

    # Ant counters
    vertices_created: int = 0
    packets_generated: int = 0

    def reset(self):
        """Reset all counters for new step"""
        self.build_attempts = 0
        self.build_success = 0
        self.blocked_by_curvature = 0
        self.blocked_by_energy = 0
        self.blocked_by_cooldown = 0
        self.blocked_by_saturation = 0
        self.blocked_by_materials = 0

        self.pickup_attempts = 0
        self.pickup_success = 0
        self.pickup_blocked_no_vertices = 0
        self.pickup_blocked_low_salience = 0
        self.pickup_blocked_no_packets = 0

        self.deliver_attempts = 0
        self.deliver_enqueued = 0
        self.deliver_blocked_not_at_wormhole = 0
        self.deliver_blocked_protocol_full = 0
        self.deliver_blocked_ttl_expired = 0

        self.vertices_created = 0
        self.packets_generated = 0

    def to_dict(self) -> Dict[str, int]:
        """Export as dictionary"""
        return {
            'build_attempts': self.build_attempts,
            'build_success': self.build_success,
            'blocked_by_curvature': self.blocked_by_curvature,
            'blocked_by_energy': self.blocked_by_energy,
            'blocked_by_cooldown': self.blocked_by_cooldown,
            'blocked_by_saturation': self.blocked_by_saturation,
            'blocked_by_materials': self.blocked_by_materials,
            'pickup_attempts': self.pickup_attempts,
            'pickup_success': self.pickup_success,
            'pickup_blocked_no_vertices': self.pickup_blocked_no_vertices,
            'pickup_blocked_low_salience': self.pickup_blocked_low_salience,
            'pickup_blocked_no_packets': self.pickup_blocked_no_packets,
            'deliver_attempts': self.deliver_attempts,
            'deliver_enqueued': self.deliver_enqueued,
            'deliver_blocked_not_at_wormhole': self.deliver_blocked_not_at_wormhole,
            'deliver_blocked_protocol_full': self.deliver_blocked_protocol_full,
            'deliver_blocked_ttl_expired': self.deliver_blocked_ttl_expired,
            'vertices_created': self.vertices_created,
            'packets_generated': self.packets_generated,
        }

    def format_log_line(self, step: int) -> str:
        """Format counters for logging"""
        build_blocked = (
            self.blocked_by_curvature +
            self.blocked_by_energy +
            self.blocked_by_cooldown +
            self.blocked_by_saturation +
            self.blocked_by_materials
        )

        pickup_blocked = (
            self.pickup_blocked_no_vertices +
            self.pickup_blocked_low_salience +
            self.pickup_blocked_no_packets
        )

        return (
            f"Step {step} | "
            f"Builds: +{self.build_success} "
            f"(blocked: curv={self.blocked_by_curvature}, "
            f"energy={self.blocked_by_energy}, "
            f"sat={self.blocked_by_saturation}, "
            f"mat={self.blocked_by_materials}) | "
            f"Packets: pick={self.pickup_success} "
            f"(blocked: salience={self.pickup_blocked_low_salience}) "
            f"deliver={self.deliver_enqueued} "
            f"drop={self.deliver_blocked_ttl_expired}"
        )
