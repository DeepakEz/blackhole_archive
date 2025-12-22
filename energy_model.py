"""
BLACKHOLE ARCHIVE: Unified Energy Model
========================================
Shared thermodynamic constants across all simulation engines.
Ensures consistent energy economics - no perpetual motion.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EnergyConstants:
    """
    Central energy constants for the simulation.
    These values ensure:
    - Colony must budget energy (no free lunch)
    - Packet work can partially pay for itself
    - No perpetual motion machines
    """

    # Movement costs (per dt)
    MOVE_COST_BASE: float = 0.0025  # Base movement cost per dt
    MOVE_COST_WITH_PACKET: float = 0.004  # Movement with packet (60% higher)

    # Beaver construction
    BUILD_COST: float = 0.06  # Energy cost to build structure
    BUILD_REWARD: float = 0.02  # Energy reward for building (net: -0.04)
    BUILD_COOLDOWN: float = 0.75  # Time units before next build

    # Build triggers (thresholds)
    CURVATURE_THRESHOLD: float = 0.01  # Lowered from 0.1 for reachability
    ENERGY_THRESHOLD: float = 0.08  # Minimum energy to build
    LOCAL_SATURATION_MAX: int = 3  # Max structures in radius=1
    BUILD_PROBABILITY: float = 0.35  # Stochastic gate

    # Packet economics
    PACKET_PICKUP_COST: float = 0.01
    PACKET_DELIVERY_REWARD: float = 0.015

    # Global material budget
    MATERIAL_BUDGET_INITIAL: float = 500.0
    MATERIAL_COST_PER_BUILD: float = 1.0

    # Decay rates
    STRUCTURAL_FIELD_DECAY: float = 0.01  # Per dt
    PHEROMONE_DECAY: float = 0.1  # Per dt
    EPISTEMIC_STRESS_DECAY: float = 0.05  # Per dt

    # Packet thresholds
    PICKUP_SALIENCE_THRESHOLD: float = 0.45  # Lower from 0.7
    HIGH_PRIORITY_THRESHOLD: float = 0.75
    PACKET_TTL: float = 50.0  # Time-to-live

    # Queue limits
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
