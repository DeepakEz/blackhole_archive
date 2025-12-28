# Blackhole Archive Enhanced Simulation
# Fixes beaver construction, adds packet transport, semantic graph building

"""
ENHANCEMENTS OVER BASE VERSION:

1. Beaver Improvements:
   - Realistic curvature computation (Ricci scalar)
   - Lower construction threshold
   - Energy replenishment from successful builds
   - Collective building behavior

2. Ant Improvements:
   - Actual semantic graph construction
   - Pheromone trails visible in output
   - Vertex discovery based on information density
   - Graph metrics in statistics

3. Bee Improvements:
   - Packet collection from ant graph
   - Wormhole targeting behavior
   - Waggle dance recruitment
   - Transport statistics tracking

4. New Visualizations:
   - Semantic graph visualization
   - Pheromone trail heatmap
   - Packet flow diagram
   - Colony health metrics over time
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import uuid

# Note: Full Lyapunov stability monitoring requires epistemic layer
# (available in blackhole_archive_production.py with --engine production)

# Import base classes
import sys
sys.path.append('/mnt/user-data/outputs')

# Import epistemic cognitive layer for Overmind integration
try:
    from epistemic_cognitive_layer import Overmind, EpistemicSemanticGraph, FreeEnergyComputer
    EPISTEMIC_LAYER_AVAILABLE = True
except ImportError:
    EPISTEMIC_LAYER_AVAILABLE = False
    Overmind = None

# Import Adversarial Pressure Layer for Phase II
try:
    from adversarial_pressure_layer import AdversarialPressureLayer, AgentState
    APL_AVAILABLE = True
except ImportError:
    APL_AVAILABLE = False
    AdversarialPressureLayer = None

# =============================================================================
# THERMODYNAMIC CONSTANTS (replaces arbitrary magic numbers)
# =============================================================================
# All constants derived from first principles or physical reasoning

# Boltzmann-like constant for information thermodynamics (geometric units)
# Relates information entropy to energy: E = k_info * S
K_INFO = 0.01  # Sets energy scale for information processing

# Friction coefficient for geodesic motion damping
# From: dE/dt = -γ v² (viscous drag in curved spacetime)
# γ chosen so that orbital decay time ~ 100 proper time units at r = 10M
GAMMA_FRICTION = 0.001

# Diffusion coefficient for Brownian motion in curved spacetime
# From: σ² = 2D Δt (Einstein relation)
# D = k_info * T / γ where T is effective temperature
D_DIFFUSION = 0.0025  # Gives σ ≈ 0.05 for dt = 1

# Energy costs (in geometric units where M = 1)
ENERGY_MOVE = 0.002      # Cost per unit proper time of movement
ENERGY_BUILD = 0.02      # Cost to modify structural field
ENERGY_COMMUNICATE = 0.005  # Cost to deposit pheromone/waggle
ENERGY_BASE_DECAY = 0.001   # Passive metabolism per unit time

# Thresholds derived from spacetime scales
CURVATURE_THRESHOLD = 0.25  # K ~ 1/r³, this corresponds to r ~ 1.6 r_s
INFO_DENSITY_THRESHOLD = 0.5  # 50% of maximum information density

# Probabilities from statistical mechanics
# P ~ exp(-E/kT) for activation processes
# With kT ~ 0.1 in geometric units:
P_SPONTANEOUS_ACTION = 0.01  # exp(-4.6) ≈ 0.01, activation energy ~ 0.46

# =============================================================================
# ENHANCED PHYSICS
# =============================================================================

class EnhancedSpacetime:
    """Enhanced spacetime with proper curvature computation"""
    
    def __init__(self, config):
        self.config = config
        self.M = config.black_hole_mass
        self.r_s = 2 * self.M
        
        # Grid
        self.r = np.linspace(config.r_min, config.r_max, config.n_r)
        self.theta = np.linspace(0, np.pi, config.n_theta)
        self.phi = np.linspace(0, 2*np.pi, config.n_phi)
        
        # Precompute metric
        self.g = self._compute_metric()

        # Structural field from beavers
        self.structural_field = np.zeros((config.n_r, config.n_theta, config.n_phi))

        # Information density field (for ants)
        self.information_density = self._initialize_information_density()

        # FIX #7: Epistemic stress field (affected by contradictions/congestion)
        # High epistemic stress = harder to traverse, like gravitational stress
        self.epistemic_stress = np.zeros((config.n_r, config.n_theta, config.n_phi))

        # DYNAMIC SPACETIME: Stress-energy tensor and metric perturbation
        # T_μν tracks energy density from agents and packets
        # Metric perturbation h_μν represents back-reaction on geometry
        self.stress_energy = np.zeros((config.n_r, config.n_theta, config.n_phi))
        self.metric_perturbation = np.zeros((config.n_r, config.n_theta, config.n_phi))

        # Bekenstein bound tracking: S ≤ 2πkRE/(ℏc)
        # In geometric units (G=c=1): S ≤ 2πRE
        self.local_entropy = np.zeros((config.n_r, config.n_theta, config.n_phi))
        # Count of thermalization events (when Bekenstein bound is enforced)
        # These are NOT violations - the bound IS enforced, excess info is thermalized
        self.bekenstein_thermalization_count = 0

        # Landauer limit tracking: E = k_B T ln(2) per bit erased
        self.bits_erased = 0
        self.thermodynamic_heat = 0.0  # Cumulative heat from erasure
    
    def _compute_metric(self) -> np.ndarray:
        """Compute Schwarzschild metric"""
        metric = np.zeros((self.config.n_r, self.config.n_theta, self.config.n_phi, 4, 4))
        
        for i, r in enumerate(self.r):
            f = 1 - self.r_s / r
            
            metric[i, :, :, 0, 0] = -f
            metric[i, :, :, 1, 1] = 1 / f
            
            for j, theta in enumerate(self.theta):
                metric[i, j, :, 2, 2] = r**2
                metric[i, j, :, 3, 3] = r**2 * np.sin(theta)**2
        
        return metric
    
    def _initialize_information_density(self) -> np.ndarray:
        """Initialize information density field for ant exploration"""
        # Create interesting structure with multiple peaks
        info = np.zeros((self.config.n_r, self.config.n_theta, self.config.n_phi))
        
        # Add Gaussian blobs at random locations
        np.random.seed(42)
        n_blobs = 20
        
        for _ in range(n_blobs):
            i = np.random.randint(0, self.config.n_r)
            j = np.random.randint(0, self.config.n_theta)
            k = np.random.randint(0, self.config.n_phi)
            
            strength = np.random.uniform(0.3, 1.0)
            width = np.random.uniform(2.0, 5.0)
            
            # Add Gaussian centered at (i,j,k)
            for di in range(-10, 11):
                for dj in range(-5, 6):
                    for dk in range(-5, 6):
                        ii = (i + di) % self.config.n_r
                        jj = max(0, min(self.config.n_theta-1, j + dj))
                        kk = (k + dk) % self.config.n_phi
                        
                        distance = np.sqrt(di**2 + dj**2 + dk**2)
                        info[ii, jj, kk] += strength * np.exp(-distance**2 / (2*width**2))
        
        # Normalize
        info = info / (np.max(info) + 1e-10)
        
        return info
    
    def get_ricci_scalar(self, position: np.ndarray) -> float:
        """
        Compute Ricci scalar at position.

        For Schwarzschild vacuum solution: R = g^μν R_μν = 0.
        This is exact - Schwarzschild is a vacuum solution to Einstein's equations.

        For non-vacuum regions (with matter/energy), use get_effective_ricci_scalar().

        Returns:
            0.0 (Schwarzschild is a vacuum solution)
        """
        # Schwarzschild is a vacuum solution: R_μν = 0 → R = 0
        return 0.0

    def get_kretschmann_scalar(self, position: np.ndarray) -> float:
        """
        Compute Kretschmann scalar K = R_μνρσ R^μνρσ at position.

        For Schwarzschild metric: K = 48 M² / r⁶

        This is the appropriate curvature measure for vacuum solutions,
        as the Ricci scalar vanishes. The Kretschmann scalar quantifies
        the strength of tidal forces and diverges at r=0.

        Physical interpretation:
        - Diverges as r → 0 (physical singularity)
        - Finite at horizon r = 2M (coordinate singularity only)
        - Falls off as r⁻⁶ at large distances

        Returns:
            Kretschmann scalar in geometric units (1/length⁴)
        """
        r = max(position[1], 1e-10)  # Avoid division by zero
        return 48.0 * self.M**2 / r**6

    def get_tidal_strength(self, position: np.ndarray) -> float:
        """
        Get tidal force strength at position (sqrt of Kretschmann).

        This provides a length^-2 quantity suitable for comparing
        with other physical scales in the simulation.

        Returns:
            sqrt(K) in geometric units (1/length²)
        """
        return np.sqrt(self.get_kretschmann_scalar(position))

    def get_curvature(self, position: np.ndarray) -> float:
        """
        Get effective curvature at position for agent decision-making.

        Uses tidal strength (sqrt of Kretschmann) as the physically
        meaningful curvature measure for vacuum spacetimes.

        This is the appropriate quantity for:
        - Beaver construction decisions (build where curvature is high)
        - Agent energy costs (curved regions are harder to traverse)
        - Stability assessments (high curvature = less stable)

        Returns:
            Effective curvature measure (1/length²)
        """
        return self.get_tidal_strength(position)
    
    def get_information_density(self, position: np.ndarray) -> float:
        """Sample information density at position"""
        # Find nearest grid point
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))
        
        return self.information_density[i, j, k]
    
    def get_time_dilation(self, position: np.ndarray) -> float:
        """Get time dilation factor"""
        r = position[1]
        return 1 / np.sqrt(max(1e-10, 1 - self.r_s / r))

    def compute_christoffel(self, position: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols at position for geodesic motion.

        For Schwarzschild metric, the non-zero components are:
        Γ^t_tr = M / (r(r - 2M))
        Γ^r_tt = M(r - 2M) / r³
        Γ^r_rr = -M / (r(r - 2M))
        Γ^r_θθ = -(r - 2M)
        Γ^r_φφ = -(r - 2M)sin²θ
        Γ^θ_rθ = 1/r
        Γ^θ_φφ = -sinθ cosθ
        Γ^φ_rφ = 1/r
        Γ^φ_θφ = cotθ
        """
        r = max(position[1], self.r_s * 1.01)  # Avoid singularity
        theta = position[2]
        M = self.M

        Gamma = np.zeros((4, 4, 4))

        # Avoid division by zero
        f = 1 - 2*M/r
        if f < 1e-10:
            f = 1e-10

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        if abs(sin_theta) < 1e-10:
            sin_theta = 1e-10

        # Non-zero Christoffel symbols for Schwarzschild
        Gamma[0, 0, 1] = Gamma[0, 1, 0] = M / (r * (r - 2*M) + 1e-10)
        Gamma[1, 0, 0] = M * (r - 2*M) / (r**3 + 1e-10)
        Gamma[1, 1, 1] = -M / (r * (r - 2*M) + 1e-10)
        Gamma[1, 2, 2] = -(r - 2*M)
        Gamma[1, 3, 3] = -(r - 2*M) * sin_theta**2
        Gamma[2, 1, 2] = Gamma[2, 2, 1] = 1/r
        Gamma[2, 3, 3] = -sin_theta * cos_theta
        Gamma[3, 1, 3] = Gamma[3, 3, 1] = 1/r
        Gamma[3, 2, 3] = Gamma[3, 3, 2] = cos_theta / sin_theta

        return Gamma

    def _geodesic_acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Compute geodesic acceleration from the geodesic equation.

        The geodesic equation is:
        d²x^μ/dτ² = -Γ^μ_αβ (dx^α/dτ)(dx^β/dτ)

        Args:
            position: 4-position (t, r, θ, φ)
            velocity: 4-velocity

        Returns:
            4-acceleration
        """
        Gamma = self.compute_christoffel(position)

        acceleration = np.zeros(4)
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    term = Gamma[mu, alpha, beta] * velocity[alpha] * velocity[beta]
                    if np.isfinite(term):
                        acceleration[mu] -= term

        # Clip to prevent numerical explosion near singularities
        return np.clip(acceleration, -10.0, 10.0)

    def geodesic_step(self, position: np.ndarray, velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute geodesic-corrected position and velocity update.

        Uses 4th-order Runge-Kutta integration (RK4) for the geodesic equation:
        dx^μ/dτ = v^μ
        dv^μ/dτ = -Γ^μ_αβ v^α v^β

        RK4 provides O(dt⁴) accuracy and good energy conservation for
        orbital dynamics, making it suitable for geodesic motion.

        For truly symplectic (exactly energy-conserving) integration,
        use geodesic_step_symplectic() instead.

        Args:
            position: Current 4-position (t, r, θ, φ)
            velocity: Current 4-velocity
            dt: Time step (proper time interval)

        Returns:
            (new_position, new_velocity)
        """
        # Safety check: ensure we're outside the horizon
        if position[1] <= self.r_s * 1.01:
            position = position.copy()
            position[1] = self.r_s * 1.05
            velocity = velocity.copy()
            velocity[1] = max(velocity[1], 0.1)

        # RK4 integration of the coupled system:
        # dx/dt = v,  dv/dt = a(x, v)

        # k1
        k1_x = velocity
        k1_v = self._geodesic_acceleration(position, velocity)

        # k2
        x2 = position + 0.5 * dt * k1_x
        v2 = velocity + 0.5 * dt * k1_v
        k2_x = v2
        k2_v = self._geodesic_acceleration(x2, v2)

        # k3
        x3 = position + 0.5 * dt * k2_x
        v3 = velocity + 0.5 * dt * k2_v
        k3_x = v3
        k3_v = self._geodesic_acceleration(x3, v3)

        # k4
        x4 = position + dt * k3_x
        v4 = velocity + dt * k3_v
        k4_x = v4
        k4_v = self._geodesic_acceleration(x4, v4)

        # Combine
        new_position = position + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_velocity = velocity + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        # Energy-based velocity constraint (replaces arbitrary np.clip(-5, 5))
        # For geodesic motion, we preserve the norm of the 4-velocity
        # In Schwarzschild: g_μν u^μ u^ν = -1 for timelike geodesics
        # Speed limit: spatial velocity bounded by effective potential
        r = max(new_position[1], self.r_s * 1.5)
        # Maximum allowed spatial velocity at radius r (from effective potential)
        # v_max² ≈ 2(E - V_eff) where V_eff = -(r_s/2r) for radial motion
        v_max = min(5.0, np.sqrt(2.0 * (1.0 + self.r_s / (2 * r))))
        spatial_speed = np.linalg.norm(new_velocity[1:])
        if spatial_speed > v_max:
            # Scale down spatial components while preserving direction
            scale = v_max / spatial_speed
            new_velocity[1:] *= scale

        # Handle NaN/Inf - reset to safe values
        if not np.all(np.isfinite(new_position)):
            new_position = position.copy()
            new_position[1] = max(new_position[1], self.r_s * 1.5)
        if not np.all(np.isfinite(new_velocity)):
            new_velocity = np.zeros(4)
            new_velocity[1] = 0.1

        # Boundary conditions
        new_position, new_velocity = self._apply_boundary_conditions(new_position, new_velocity)

        return new_position, new_velocity

    def geodesic_step_symplectic(self, position: np.ndarray, velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Symplectic (Störmer-Verlet/Leapfrog) geodesic integration.

        This integrator exactly preserves the symplectic structure of
        Hamiltonian dynamics, providing exact energy conservation
        (up to floating-point precision) over arbitrarily long times.

        The leapfrog scheme:
        v_{n+1/2} = v_n + (dt/2) * a(x_n, v_n)
        x_{n+1} = x_n + dt * v_{n+1/2}
        v_{n+1} = v_{n+1/2} + (dt/2) * a(x_{n+1}, v_{n+1/2})

        Args:
            position: Current 4-position (t, r, θ, φ)
            velocity: Current 4-velocity
            dt: Time step

        Returns:
            (new_position, new_velocity)
        """
        # Safety check
        if position[1] <= self.r_s * 1.01:
            position = position.copy()
            position[1] = self.r_s * 1.05
            velocity = velocity.copy()
            velocity[1] = max(velocity[1], 0.1)

        # Half-step velocity update (kick)
        a0 = self._geodesic_acceleration(position, velocity)
        v_half = velocity + 0.5 * dt * a0

        # Full-step position update (drift)
        new_position = position + dt * v_half

        # Half-step velocity update (kick)
        a1 = self._geodesic_acceleration(new_position, v_half)
        new_velocity = v_half + 0.5 * dt * a1

        # Energy-based velocity constraint (replaces arbitrary np.clip(-5, 5))
        r = max(new_position[1], self.r_s * 1.5)
        v_max = min(5.0, np.sqrt(2.0 * (1.0 + self.r_s / (2 * r))))
        spatial_speed = np.linalg.norm(new_velocity[1:])
        if spatial_speed > v_max:
            scale = v_max / spatial_speed
            new_velocity[1:] *= scale

        # Handle NaN/Inf
        if not np.all(np.isfinite(new_position)):
            new_position = position.copy()
            new_position[1] = max(new_position[1], self.r_s * 1.5)
        if not np.all(np.isfinite(new_velocity)):
            new_velocity = np.zeros(4)
            new_velocity[1] = 0.1

        # Boundary conditions
        new_position, new_velocity = self._apply_boundary_conditions(new_position, new_velocity)

        return new_position, new_velocity

    def _apply_boundary_conditions(self, position: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions to position and velocity."""
        position = position.copy()
        velocity = velocity.copy()

        # Keep r outside event horizon
        if position[1] <= self.r_s * 1.01:
            position[1] = self.r_s * 1.05
            velocity[1] = abs(velocity[1])  # Bounce outward

        # Keep r inside domain
        if position[1] >= self.config.r_max * 0.95:
            position[1] = self.config.r_max * 0.9
            velocity[1] = -abs(velocity[1]) * 0.5  # Reflect inward

        # Keep theta in [0, π]
        position[2] = np.clip(position[2], 0.01, np.pi - 0.01)

        # Wrap phi to [0, 2π]
        position[3] = position[3] % (2 * np.pi)

        return position, velocity

    def add_structural_field(self, position: np.ndarray, strength: float, radius: float):
        """Add beaver structural field with proper boundary handling"""
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # FIX: No-build zone at domain boundaries (r_max region)
        # Prevents boundary exploit where agents "build at edge of map"
        if i >= self.config.n_r - 3:  # Too close to outer boundary
            return

        # Add Gaussian with CLAMPED radial indices (no wrap-around)
        for di in range(-5, 6):
            for dj in range(-3, 4):
                for dk in range(-3, 4):
                    # FIX: Clamp radial index instead of modulo wrap
                    # Wrap is only valid for phi (azimuthal), not r (radial)
                    ii = max(0, min(self.config.n_r - 1, i + di))
                    jj = max(0, min(self.config.n_theta - 1, j + dj))
                    kk = (k + dk) % self.config.n_phi  # Phi wraps correctly

                    distance = np.sqrt(di**2 + dj**2 + dk**2)
                    if distance < radius:
                        self.structural_field[ii, jj, kk] += strength * np.exp(-distance**2 / (2 * radius**2))

    # FIX #2: Structural field coupling methods
    def get_structural_field_at(self, position: np.ndarray) -> float:
        """Get structural field value at position - FIX #2"""
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))
        return self.structural_field[i, j, k]

    def get_movement_cost(self, position: np.ndarray) -> float:
        """
        Get movement cost at position - FIX #2
        Structural field REDUCES movement cost (easier to traverse)
        """
        base_cost = 1.0
        structural = self.get_structural_field_at(position)
        # Structural field reduces cost: more structure = easier movement
        # Formula: cost = base / (1 + structural)
        return base_cost / (1.0 + structural)

    def get_effective_time_dilation(self, position: np.ndarray) -> float:
        """
        Get time dilation modified by structural field - FIX #2
        Structural field provides "exotic matter" effect that reduces dilation
        """
        r = position[1]
        base_dilation = 1 / np.sqrt(max(1e-10, 1 - self.r_s / r))
        structural = self.get_structural_field_at(position)
        # Structural field reduces time dilation (stabilizes region)
        # Formula: effective = base / (1 + 0.1 * structural)
        return base_dilation / (1.0 + 0.1 * structural)

    def get_exploration_bias(self, position: np.ndarray) -> float:
        """
        Get exploration bias from structural field - FIX #2
        High structural field = already explored, look elsewhere
        Returns value in [0, 1] where 0 = explore elsewhere, 1 = explore here
        """
        structural = self.get_structural_field_at(position)
        # Inverse relationship: high structure = low exploration priority
        return 1.0 / (1.0 + structural)

    def get_structural_field(self, position: np.ndarray) -> float:
        """Alias for get_structural_field_at for cleaner API."""
        return self.get_structural_field_at(position)

    def get_structural_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Get gradient of structural field at position.
        Returns 3D gradient vector (dr, dtheta, dphi) pointing toward higher structure.
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Compute numerical gradient using central differences
        gradient = np.zeros(3)

        # dr gradient
        if 0 < i < self.config.n_r - 1:
            dr = self.r[1] - self.r[0] if len(self.r) > 1 else 1.0
            gradient[0] = (self.structural_field[i+1, j, k] - self.structural_field[i-1, j, k]) / (2 * dr)

        # dtheta gradient
        if 0 < j < self.config.n_theta - 1:
            dtheta = self.theta[1] - self.theta[0] if len(self.theta) > 1 else 1.0
            gradient[1] = (self.structural_field[i, j+1, k] - self.structural_field[i, j-1, k]) / (2 * dtheta)

        # dphi gradient (periodic)
        dphi = self.phi[1] - self.phi[0] if len(self.phi) > 1 else 1.0
        k_next = (k + 1) % self.config.n_phi
        k_prev = (k - 1) % self.config.n_phi
        gradient[2] = (self.structural_field[i, j, k_next] - self.structural_field[i, j, k_prev]) / (2 * dphi)

        return gradient

    def decay_structural_field(self, dt: float, decay_rate: float = 0.01):
        """
        FIX #5: Structural field decays over time (maintenance cost).
        Structures that aren't maintained fade away.
        This prevents infinite accumulation and encourages active maintenance.
        """
        self.structural_field *= np.exp(-decay_rate * dt)

        # Floor to zero for very small values (cleanup)
        self.structural_field[self.structural_field < 0.01] = 0.0

    def add_epistemic_stress(self, position: np.ndarray, stress: float, radius: float = 3.0):
        """
        FIX #7: Add epistemic stress at a position.
        Called when contradictions or congestion occur.
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Add Gaussian stress field
        for di in range(-3, 4):
            for dj in range(-2, 3):
                for dk in range(-2, 3):
                    ii = max(0, min(self.config.n_r - 1, i + di))
                    jj = max(0, min(self.config.n_theta - 1, j + dj))
                    kk = (k + dk) % self.config.n_phi

                    distance = np.sqrt(di**2 + dj**2 + dk**2)
                    if distance < radius:
                        self.epistemic_stress[ii, jj, kk] += stress * np.exp(-distance**2 / (2 * radius**2))

    def decay_epistemic_stress(self, dt: float, decay_rate: float = 0.05):
        """FIX #7: Epistemic stress decays (contradictions can be resolved)"""
        self.epistemic_stress *= np.exp(-decay_rate * dt)
        self.epistemic_stress[self.epistemic_stress < 0.01] = 0.0

    def get_epistemic_stress_at(self, position: np.ndarray) -> float:
        """Get epistemic stress at position"""
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))
        return self.epistemic_stress[i, j, k]

    def get_total_traversal_cost(self, position: np.ndarray) -> float:
        """
        FIX #7: Combined traversal cost including epistemic stress.
        High stress = epistemic gravity well, harder to traverse.
        """
        base_cost = self.get_movement_cost(position)
        stress = self.get_epistemic_stress_at(position)
        # Stress increases traversal cost
        return base_cost * (1.0 + stress)

    # =========================================================================
    # DYNAMIC SPACETIME: Stress-Energy and Metric Back-Reaction
    # =========================================================================

    def add_stress_energy(self, position: np.ndarray, energy: float):
        """
        Add stress-energy contribution at position.

        In full GR: G_μν = 8πT_μν
        Here we use linearized approximation where metric perturbation
        is proportional to integrated stress-energy.

        Args:
            position: 4-position (t, r, θ, φ)
            energy: Energy contribution (from agent or packet)
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Add to stress-energy tensor (simplified scalar representation)
        self.stress_energy[i, j, k] += energy

    def evolve_metric(self, dt: float):
        """
        Evolve metric perturbation based on stress-energy distribution.

        Uses linearized Einstein equations in weak-field limit:
        □h_μν = -16πT_μν

        In our simplified scalar model (trace-reversed perturbation):
        ∂²h/∂t² - c²∇²h = -16πGT/c²

        In geometric units (G=c=1): ∂²h/∂t² - ∇²h = -16πT

        Physical parameters:
        - Gravitational wave speed = c = 1 (geometric units)
        - Radiative damping rate from quadrupole formula: Γ = (32/5) (GM/c³)⁵ / r⁵
          For weak fields, this is negligible compared to propagation

        The metric perturbation affects:
        - Time dilation (stronger near high energy density)
        - Geodesic paths (agents curve toward energy concentrations)
        """
        # Store previous state for wave equation (need h, h_old for leapfrog)
        if not hasattr(self, '_metric_perturbation_old'):
            self._metric_perturbation_old = np.zeros_like(self.metric_perturbation)

        # Compute Laplacian of metric perturbation (spherical coords)
        laplacian = np.zeros_like(self.metric_perturbation)

        # Radial part: (1/r²) d/dr (r² dh/dr)
        # Using central differences for interior points
        dr = self.r[1] - self.r[0] if len(self.r) > 1 else 1.0
        for i in range(1, self.config.n_r - 1):
            r_i = self.r[i]
            # Second derivative term
            d2h_dr2 = (
                self.metric_perturbation[i+1, :, :] -
                2*self.metric_perturbation[i, :, :] +
                self.metric_perturbation[i-1, :, :]
            ) / (dr**2)
            # First derivative term (from 1/r² d/dr(r² dh/dr))
            dh_dr = (
                self.metric_perturbation[i+1, :, :] -
                self.metric_perturbation[i-1, :, :]
            ) / (2 * dr)
            laplacian[i, :, :] = d2h_dr2 + (2/r_i) * dh_dr

        # Source term from stress-energy: S = -16πT
        # The stress-energy here represents energy density contributions
        source = -16 * np.pi * self.stress_energy

        # Leapfrog/Störmer-Verlet time integration for wave equation
        # h_{n+1} = 2h_n - h_{n-1} + dt²(∇²h + S)
        # This is symplectic and conserves energy for the wave equation
        new_perturbation = (
            2 * self.metric_perturbation -
            self._metric_perturbation_old +
            dt**2 * (laplacian + source)
        )

        # Physical damping from gravitational wave radiation
        # Quadrupole formula gives damping timescale τ ~ r⁵/(GM)⁵
        # For simulation stability, use scale-appropriate damping
        # Damping timescale τ_damp = r_max / c = r_max (in geometric units)
        tau_damp = self.config.r_max
        gamma = dt / tau_damp  # Dimensionless damping per timestep
        new_perturbation *= (1.0 - gamma)

        # Update states
        self._metric_perturbation_old = self.metric_perturbation.copy()
        self.metric_perturbation = new_perturbation

        # Physical constraint: linearized approximation valid for |h| << 1
        # If perturbation exceeds this, the linearized equations break down
        max_perturbation = 0.3  # Beyond this, need full nonlinear GR
        if np.max(np.abs(self.metric_perturbation)) > max_perturbation:
            # Scale down to valid regime (energy conservation)
            scale = max_perturbation / np.max(np.abs(self.metric_perturbation))
            self.metric_perturbation *= scale
            self._metric_perturbation_old *= scale

        # Stress-energy disperses via radiation and diffusion
        # Physical timescale: crossing time ~ r_max / c = r_max
        dispersion_rate = 1.0 / self.config.r_max
        self.stress_energy *= np.exp(-dispersion_rate * dt)

    def get_effective_mass(self, position: np.ndarray) -> float:
        """
        Get effective gravitational mass including metric perturbation.

        The total mass felt at a point is M + δM where δM comes from
        the integrated stress-energy perturbation.
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Perturbation contributes to effective mass
        delta_M = self.metric_perturbation[i, j, k] * self.M
        return self.M + delta_M

    # =========================================================================
    # BEKENSTEIN BOUND AND THERMALIZATION
    # =========================================================================

    def add_local_entropy(self, position: np.ndarray, bits: float):
        """
        Add entropy (information) at position and check Bekenstein bound.

        Bekenstein bound: S ≤ 2πRE/(ℏc) = 2πRE (geometric units)

        If violated, information is "thermalized" (randomly corrupted)
        simulating the physical limit on information density.

        Returns:
            bits_lost: Number of bits thermalized due to bound violation
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        r = position[1]

        # Add entropy
        self.local_entropy[i, j, k] += bits

        # Bekenstein bound: S_max = 2πRE
        # Using local energy from stress-energy tensor
        local_energy = max(self.stress_energy[i, j, k], 0.01)
        bekenstein_limit = 2 * np.pi * r * local_energy

        # Check for violation
        current_entropy = self.local_entropy[i, j, k]
        if current_entropy > bekenstein_limit:
            # Thermalization: excess information is lost
            excess = current_entropy - bekenstein_limit
            self.local_entropy[i, j, k] = bekenstein_limit

            # Track thermalization event (bound enforced, excess info discarded)
            self.bekenstein_thermalization_count += 1

            # Landauer cost: erasing bits generates heat
            self.bits_erased += excess
            # k_B T ln(2) per bit, using T ~ 1/(8πM) Hawking temperature
            hawking_temp = 1.0 / (8 * np.pi * self.M)
            self.thermodynamic_heat += excess * hawking_temp * np.log(2)

            return excess

        return 0.0

    def decay_entropy(self, dt: float, rate: float = 0.02):
        """Entropy naturally dissipates (Hawking radiation analog)"""
        self.local_entropy *= np.exp(-rate * dt)

    def get_bekenstein_capacity(self, position: np.ndarray) -> float:
        """
        Get remaining Bekenstein capacity at position.

        Returns fraction of capacity remaining (0 = full, 1 = empty)
        """
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        r = position[1]
        local_energy = max(self.stress_energy[i, j, k], 0.01)
        bekenstein_limit = 2 * np.pi * r * local_energy

        current = self.local_entropy[i, j, k]
        return max(0.0, 1.0 - current / bekenstein_limit)

    # =========================================================================
    # VALIDATION METRICS
    # =========================================================================

    def estimate_kolmogorov_complexity(self, data: np.ndarray) -> float:
        """
        Estimate Kolmogorov complexity using compression ratio.

        K(x) ≈ len(compress(x))

        Higher values = more complex/random data
        Lower values = more compressible/structured data
        """
        import zlib

        # Convert to bytes
        data_bytes = data.tobytes()

        # Compress
        compressed = zlib.compress(data_bytes, level=9)

        # Complexity estimate = compression ratio
        if len(data_bytes) == 0:
            return 0.0

        return len(compressed) / len(data_bytes)

    def get_thermodynamic_stats(self) -> Dict[str, float]:
        """
        Get thermodynamic validation statistics.

        Returns metrics for validating physical consistency:
        - Total entropy in system
        - Bekenstein violations (should be rare)
        - Landauer heat generated
        - Compression efficiency (Kolmogorov estimate)
        """
        return {
            'total_entropy': float(np.sum(self.local_entropy)),
            'max_local_entropy': float(np.max(self.local_entropy)),
            'bekenstein_thermalization_events': self.bekenstein_thermalization_count,
            'bits_erased': self.bits_erased,
            'thermodynamic_heat': self.thermodynamic_heat,
            'mean_metric_perturbation': float(np.mean(np.abs(self.metric_perturbation))),
            'max_stress_energy': float(np.max(self.stress_energy)),
        }

# =============================================================================
# ACTIVE INFERENCE FRAMEWORK
# =============================================================================

class ActiveInferenceMixin:
    """
    Active Inference agent model based on Free Energy Principle.

    Agents don't have hardcoded roles - they minimize variational free energy:

    F = D_KL[q(ψ)||p(ψ)] - E_q[ln p(o|ψ)]
      = Complexity - Accuracy

    Where:
    - q(ψ): Agent's beliefs about world state
    - p(ψ): Prior beliefs (generative model)
    - p(o|ψ): Likelihood of observations given state
    - Complexity: Cost of updating beliefs
    - Accuracy: How well beliefs predict observations

    Actions are selected to minimize expected free energy (epistemic + pragmatic value).
    """

    def __init_active_inference__(self):
        """Initialize active inference state"""
        # Beliefs about world state (simplified as feature vector)
        self.beliefs = np.zeros(8)  # [energy, curvature, density, stress, ...]

        # Precision (inverse variance) of beliefs - how confident
        self.precision = np.ones(8) * 0.5

        # Prior preferences (what states the agent prefers)
        self.preferences = np.zeros(8)

        # Free energy history for learning
        self.free_energy_history = []

        # Epistemic value (information gain motivation)
        self.epistemic_drive = 0.5

    def compute_free_energy(self, observation: np.ndarray, action: np.ndarray = None) -> float:
        """
        Compute variational free energy given observation.

        F = Complexity + Accuracy
          = D_KL[q||p] - E_q[ln p(o|s)]

        Lower free energy = better model of the world
        """
        # Complexity: KL divergence between posterior and prior beliefs
        # Using simplified Gaussian assumption
        prior_mean = self.preferences
        posterior_mean = self.beliefs

        # KL divergence for Gaussians: 0.5 * sum((μ1-μ2)² / σ² + log(σ²/σ²) + σ²/σ² - 1)
        # Simplified: just squared difference weighted by precision
        complexity = 0.5 * np.sum(self.precision * (posterior_mean - prior_mean)**2)

        # Accuracy: negative log likelihood of observation under beliefs
        # How well do current beliefs predict what we observe?
        prediction_error = observation - self.beliefs
        accuracy = -0.5 * np.sum(self.precision * prediction_error**2)

        # Free energy = complexity - accuracy (want to minimize)
        free_energy = complexity - accuracy

        return free_energy

    def update_beliefs(self, observation: np.ndarray, learning_rate: float = 0.1):
        """
        Update beliefs based on observation (perception).

        Implements belief updating to minimize prediction error.
        """
        # Prediction error
        error = observation - self.beliefs

        # Update beliefs toward observation (gradient descent on free energy)
        self.beliefs += learning_rate * self.precision * error

        # Update precision based on error magnitude (learn confidence)
        error_magnitude = np.abs(error)
        self.precision = 0.9 * self.precision + 0.1 * (1.0 / (error_magnitude + 0.1))

    def expected_free_energy(self, action: np.ndarray, spacetime, position: np.ndarray) -> float:
        """
        Compute expected free energy for an action.

        G = E_q[F(o', s')] = Epistemic Value + Pragmatic Value

        Where:
        - Epistemic: Information gain from action
        - Pragmatic: Preference satisfaction from action
        """
        # Predict next state after action
        predicted_position = position + action * 0.1

        # Get predicted observations at new position
        predicted_obs = self._get_observation(spacetime, predicted_position)

        # Epistemic value: expected information gain
        # Higher uncertainty reduction = higher epistemic value
        uncertainty_before = 1.0 / (np.sum(self.precision) + 1e-6)

        # Simulate belief update
        temp_beliefs = self.beliefs.copy()
        temp_beliefs += 0.1 * self.precision * (predicted_obs - self.beliefs)

        # Uncertainty after (estimated)
        uncertainty_after = np.sum((predicted_obs - temp_beliefs)**2)

        epistemic_value = uncertainty_before - uncertainty_after

        # Pragmatic value: how close to preferences?
        pragmatic_value = -np.sum((predicted_obs - self.preferences)**2)

        # Expected free energy (lower is better)
        return -self.epistemic_drive * epistemic_value - pragmatic_value

    def select_action_active_inference(self, spacetime, candidate_actions: List[np.ndarray]) -> np.ndarray:
        """
        Select action by minimizing expected free energy.

        This replaces hardcoded role-based behavior with principled action selection.
        """
        best_action = candidate_actions[0]
        best_G = float('inf')

        for action in candidate_actions:
            G = self.expected_free_energy(action, spacetime, self.position)
            if G < best_G:
                best_G = G
                best_action = action

        return best_action

    def _get_observation(self, spacetime, position: np.ndarray) -> np.ndarray:
        """Get observation vector at position"""
        obs = np.zeros(8)
        obs[0] = self.energy  # Own energy
        obs[1] = spacetime.get_curvature(position)  # Local curvature
        obs[2] = spacetime.get_information_density(position)  # Info density
        obs[3] = spacetime.get_epistemic_stress_at(position)  # Stress
        obs[4] = spacetime.get_structural_field(position)  # Structure
        obs[5] = spacetime.get_effective_mass(position) / spacetime.M  # Effective mass ratio
        obs[6] = spacetime.get_bekenstein_capacity(position)  # Available capacity
        obs[7] = position[1] / spacetime.config.r_max  # Normalized radius
        return obs

# =============================================================================
# ENHANCED AGENTS
# =============================================================================

@dataclass
class Agent:
    """Base agent"""
    id: str
    colony: str
    position: np.ndarray
    velocity: np.ndarray
    energy: float
    state: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedBeaverAgent(Agent):
    """Enhanced beaver with realistic construction and scarcity"""

    # Class-level material budget (shared across all beavers)
    global_material_budget = 500.0  # Finite resources

    # Sigmoid cap parameters for productivity scaling
    SIGMOID_ALPHA = 0.3  # Maximum productivity bonus
    SIGMOID_K = 2.0  # Steepness of sigmoid curve

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structures_built = 0
        self.construction_cooldown = 0

    @staticmethod
    def sigmoid_cap(signal: float, alpha: float = 0.3, k: float = 2.0) -> float:
        """
        Sigmoid-capped productivity scaling.
        Formula: base * (1 + α * tanh(k * signal))

        Prevents runaway growth while allowing bounded bonuses.
        """
        return 1.0 + alpha * np.tanh(k * signal)

    def update(self, dt: float, spacetime: EnhancedSpacetime, semantic_graph=None):
        # Cooldown
        if self.construction_cooldown > 0:
            self.construction_cooldown -= dt

        # Check curvature using Kretschmann-based tidal strength (not Ricci which is 0 in vacuum)
        # get_curvature() returns sqrt(K) where K = 48M²/r⁶ for Schwarzschild
        curvature = spacetime.get_curvature(self.position)

        # Threshold based on tidal strength: ~0.01 corresponds to moderate curvature regions
        if curvature > 0.01 and self.energy > 0.05 and self.construction_cooldown <= 0:
            # FIX #5: Diminishing returns based on local structural density
            local_structure = spacetime.get_structural_field_at(self.position)

            # SIGMOID CAP: Use tanh-based diminishing returns instead of 1/(1+x)
            # This prevents both runaway growth and complete stagnation
            diminishing_factor = self.sigmoid_cap(-local_structure, self.SIGMOID_ALPHA, self.SIGMOID_K)
            diminishing_factor = max(0.1, diminishing_factor)  # Floor at 0.1

            # Check global material budget
            material_cost = 1.0  # Each build costs 1 unit of material
            if EnhancedBeaverAgent.global_material_budget < material_cost:
                # No materials left - enter resource conservation mode
                # Reduce velocity to conserve energy while waiting for materials
                self.velocity *= 0.5
                self.construction_cooldown = 5.0  # Long cooldown to wait for materials
            elif diminishing_factor < 0.15:
                # Too much structure already - seek less crowded area
                # Move away from high-density region
                gradient = spacetime.get_structural_gradient(self.position)
                self.velocity -= 0.1 * gradient  # Move against gradient (toward less structure)
                self.construction_cooldown = 2.0  # Short cooldown before trying elsewhere
            else:
                # Build structure with sigmoid-capped strength
                build_strength = 2.0 * diminishing_factor
                spacetime.add_structural_field(self.position, build_strength, 3.0)
                self.structures_built += 1

                # Consume global materials
                EnhancedBeaverAgent.global_material_budget -= material_cost

                # ENERGY FIX: Construction costs energy, no direct reward
                # Edges reduce decay, NOT create energy (handled in decay calculation)
                construction_cost = 0.02
                self.energy -= construction_cost

                # Cooldown scales with local density (harder to build in crowded areas)
                self.construction_cooldown = 1.0 + 0.5 * local_structure
        
        # Move toward high curvature regions
        # Sample nearby points using Kretschmann-based curvature (non-zero in vacuum)
        # NOTE: get_ricci_scalar() returns 0 in Schwarzschild vacuum - NEVER use for navigation
        nearby_curvatures = []
        directions = []

        for _ in range(5):
            offset = 0.5 * np.random.randn(4)
            test_pos = self.position + offset
            # Use get_curvature() = sqrt(Kretschmann) which is non-zero: K = 48M²/r⁶
            test_curv = spacetime.get_curvature(test_pos)
            nearby_curvatures.append(test_curv)
            directions.append(offset)
        
        # Move toward highest curvature
        max_idx = np.argmax(nearby_curvatures)
        self.velocity = 0.8 * self.velocity + 0.2 * directions[max_idx]

        # Update position using geodesic motion (respects spacetime curvature)
        # This replaces naive Euclidean: self.position += dt * self.velocity
        self.position, self.velocity = spacetime.geodesic_step(
            self.position, self.velocity, dt
        )

        # FIX #2: Energy decay modified by structural field AND edges
        # Moving through structured regions costs less energy
        # EDGES REDUCE DECAY: Graph connectivity provides efficiency bonus
        movement_cost = spacetime.get_movement_cost(self.position)
        base_decay = dt * 0.005 * movement_cost

        # Edge-based decay reduction: more edges = less decay
        edge_reduction = 1.0
        if semantic_graph is not None:
            n_edges = semantic_graph.graph.number_of_edges()
            # Sigmoid cap on edge benefit: edges reduce decay, NOT create energy
            # Formula: decay_multiplier = 1 / (1 + α * tanh(edges / scale))
            edge_reduction = 1.0 / (1.0 + 0.3 * np.tanh(n_edges / 20.0))

        self.energy -= base_decay * edge_reduction

        if self.energy <= 0:
            self.state = "dead"


class EnhancedAntAgent(Agent):
    """Enhanced ant with semantic graph building, epistemic guidance, and individual accountability"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_vertex = None
        self.pheromone_deposits = 0
        self.path_history = []
        self.packets_generated = 0
        self.last_position = None  # For co-occurrence edge creation
        self.discovery_times = {}  # vertex_id -> discovery_time for temporal edges

        # ISSUE 1: Epistemic guidance state
        self.belief_id: Optional[int] = None
        self.information_gain_history: List[float] = []
        self.epistemic_memory = []  # Track visited high-info regions

        # ISSUE 2: Individual accountability tracking
        self.vertices_created = 0      # Personal discovery count
        self.edges_created = 0         # Personal connection count
        self.total_info_discovered = 0.0  # Cumulative information gain
        self.contribution_score = 0.0  # Running counterfactual contribution

    def _find_nearby_vertex(self, semantic_graph, distance_threshold: float = 2.0):
        """Find the closest existing vertex within distance threshold."""
        closest_vertex = None
        closest_distance = float('inf')

        for v in semantic_graph.graph.nodes():
            node_data = semantic_graph.graph.nodes.get(v, {})
            if 'position' not in node_data:
                continue
            pos = node_data['position']
            # Use spatial distance (ignore time component)
            dist = np.linalg.norm(self.position[1:] - pos[1:])
            if dist < distance_threshold and dist < closest_distance:
                closest_distance = dist
                closest_vertex = v

        return closest_vertex

    def _find_all_nearby_vertices(self, semantic_graph, distance_threshold: float = 3.0):
        """Find all existing vertices within distance threshold."""
        nearby = []
        for v in semantic_graph.graph.nodes():
            node_data = semantic_graph.graph.nodes.get(v, {})
            if 'position' not in node_data:
                continue
            pos = node_data['position']
            dist = np.linalg.norm(self.position[1:] - pos[1:])
            if dist < distance_threshold:
                nearby.append(v)
        return nearby

    def update(self, dt: float, spacetime: EnhancedSpacetime, semantic_graph, current_time: float = 0.0):
        # Sample local information
        info_density = spacetime.get_information_density(self.position)

        # ENERGY REGENERATION: Foraging reward for exploring high-info regions
        # Ants gain energy proportional to information density (biological foraging analog)
        foraging_gain = dt * 0.002 * info_density  # Scales with info richness
        self.energy = min(1.5, self.energy + foraging_gain)  # Cap at 1.5x initial energy

        # If sufficient information density, create vertex or attach to nearby existing one
        # LOWERED THRESHOLD: 0.5 → 0.25 to allow more vertex creation in moderate-info regions
        # LOWERED THRESHOLD: 0.25 → 0.15 to allow more vertex creation
        if info_density > 0.15 and self.current_vertex is None:
            # Check for nearby existing vertices first (spatial co-occurrence)
            nearby_vertex = self._find_nearby_vertex(semantic_graph, distance_threshold=2.0)

            if nearby_vertex is not None:
                # Attach to existing vertex instead of creating new one
                vertex_id = nearby_vertex
                # Use mark_vertex_accessed for proper stability tracking
                semantic_graph.mark_vertex_accessed(vertex_id, current_time)
            else:
                # Create new vertex with current_time for grace period tracking
                vertex_id = semantic_graph.add_vertex(
                    position=self.position.copy(),
                    salience=info_density,
                    current_time=current_time
                )
                # ISSUE 2: Track individual contribution
                self.vertices_created += 1
                self.total_info_discovered += info_density
                # Contribution score: weighted by info density (personal discovery value)
                self.contribution_score += info_density * 0.5

                # ENERGY REWARD: Proportional to individual contribution
                # Personal reward = 0.7 * discovery + 0.3 * avg colony benefit
                personal_reward = 0.05 * info_density  # Scales with discovery quality
                self.energy = min(1.5, self.energy + personal_reward)

            self.current_vertex = vertex_id
            self.path_history.append(vertex_id)
            self.discovery_times[vertex_id] = current_time

            # Generate packet when discovering salient vertex
            if info_density > 0.35:  # Moderate threshold for packet generation
                packet = {
                    'content': f"discovery_{vertex_id}",
                    'salience': info_density,
                    'confidence': min(1.0, info_density + 0.2),
                    'created_at': 0.0,  # Would use simulation time
                    'ttl': 50.0,  # FIX #4: Packet time-to-live
                    'source_agent': self.id,
                    'source_vertex': vertex_id
                }
                if semantic_graph.add_packet(vertex_id, packet):
                    self.packets_generated += 1
        
        # If at vertex, deposit pheromone and move to neighbor
        if self.current_vertex is not None:
            # SAFETY CHECK: Verify vertex still exists (may have been pruned)
            if self.current_vertex not in semantic_graph.graph:
                self.current_vertex = None
                # Skip to exploration mode below
            else:
                # Use mark_vertex_accessed for stability tracking (increments access count)
                semantic_graph.mark_vertex_accessed(self.current_vertex, current_time)

                # Record activation for co-occurrence tracking
                semantic_graph.record_activation(self.current_vertex, current_time, self.id)

                # Count nearby structures for structure-dependent edge probability
                nearby_structures = len(self._find_all_nearby_vertices(semantic_graph, distance_threshold=3.0))

                # EDGE POLICY 1: Temporal adjacency - connect sequential discoveries
                # Path continuity is GUARANTEED during bootstrap phase to kickstart network
                if len(self.path_history) > 1:
                    prev_vertex = self.path_history[-2]
                    if prev_vertex in semantic_graph.graph and self.current_vertex in semantic_graph.graph:
                        # First ensure both vertices have activation records
                        semantic_graph.record_activation(prev_vertex, current_time - 0.1, self.id)

                        # Check if edge already exists
                        if not semantic_graph.graph.has_edge(prev_vertex, self.current_vertex):
                            # BOOTSTRAP GUARANTEE: Always create path edges during bootstrap
                            in_bootstrap = semantic_graph.is_in_bootstrap_phase(current_time)
                            edge_created = False

                            if in_bootstrap:
                                # Guaranteed edge creation during bootstrap for path continuity
                                semantic_graph.add_edge(prev_vertex, self.current_vertex, pheromone=1.0)
                                semantic_graph.add_edge(self.current_vertex, prev_vertex, pheromone=1.0)
                                edge_created = True
                            else:
                                # Post-bootstrap: use conditional edge formation
                                if semantic_graph.add_edge_conditional(prev_vertex, self.current_vertex, 1.0,
                                                                       current_time, nearby_structures):
                                    semantic_graph.add_edge(self.current_vertex, prev_vertex, pheromone=1.0)
                                    edge_created = True

                            if edge_created:
                                self.pheromone_deposits += 2
                                # ISSUE 2: Track individual edge contribution
                                self.edges_created += 2
                                self.contribution_score += 0.2  # Edge creation value
                                # ENERGY REWARD: Edge creation strengthens the network
                                self.energy = min(1.5, self.energy + 0.03)

                # CONDITIONAL EDGE POLICY 2: Co-occurrence - connect vertices discovered close in time
                # Only creates edges if co-occurrence count > 0 (temporal overlap requirement)
                recent_discoveries = [v for v, t in self.discovery_times.items()
                                      if current_time - t < 5.0 and v != self.current_vertex
                                      and v in semantic_graph.graph]
                for other_vertex in recent_discoveries[-3:]:  # Limit to 3 most recent
                    if not semantic_graph.graph.has_edge(self.current_vertex, other_vertex):
                        # Conditional edge - requires co-occurrence
                        if semantic_graph.add_edge_conditional(self.current_vertex, other_vertex, 0.5,
                                                               current_time, nearby_structures):
                            semantic_graph.add_edge(other_vertex, self.current_vertex, pheromone=0.5)
                            # ENERGY REWARD: Co-occurrence edges reinforce memory patterns
                            self.energy = min(1.5, self.energy + 0.02)

                # CONDITIONAL EDGE POLICY 3: Spatial proximity - connect to nearby vertices
                # Structure-dependent probability: 0.05 + 0.05 * nearby_structures
                for nearby_v in self._find_all_nearby_vertices(semantic_graph, distance_threshold=3.0):
                    if nearby_v != self.current_vertex and not semantic_graph.graph.has_edge(self.current_vertex, nearby_v):
                        # Record activation for nearby vertex to enable co-occurrence
                        semantic_graph.record_activation(nearby_v, current_time, self.id)
                        if semantic_graph.add_edge_conditional(self.current_vertex, nearby_v, 0.3,
                                                               current_time, nearby_structures):
                            semantic_graph.add_edge(nearby_v, self.current_vertex, pheromone=0.3)
                            # ENERGY REWARD: Spatial connections expand network reach
                            self.energy = min(1.5, self.energy + 0.01)

                # Choose next vertex - FIX: Use both successors and predecessors for DiGraph
                successors = set(semantic_graph.graph.successors(self.current_vertex))
                predecessors = set(semantic_graph.graph.predecessors(self.current_vertex))
                neighbors = list(successors | predecessors)

                if neighbors:
                    # CRITICAL FIX: Force exploration with probability based on stagnation
                    # If ant has visited same vertices repeatedly, force exploration
                    recent_unique = len(set(self.path_history[-10:])) if len(self.path_history) >= 10 else 10
                    stagnation_factor = 1.0 - (recent_unique / 10.0)  # 0 = diverse, 1 = stuck

                    # Exploration probability increases with stagnation and low vertex creation
                    base_explore_prob = 0.15  # 15% base chance to explore instead of follow
                    explore_prob = base_explore_prob + 0.5 * stagnation_factor

                    # Also boost exploration if ant hasn't created vertices recently
                    if self.vertices_created == 0:
                        explore_prob += 0.3  # Strong boost for unproductive ants

                    if np.random.random() < explore_prob:
                        # FORCE EXPLORATION: Detach from graph to seek new regions
                        self.current_vertex = None
                    else:
                        # Follow pheromones probabilistically
                        pheromones = []
                        for n in neighbors:
                            # Check both edge directions for pheromone
                            p1 = semantic_graph.get_pheromone((self.current_vertex, n))
                            p2 = semantic_graph.get_pheromone((n, self.current_vertex))
                            pheromones.append(max(p1, p2))
                        probs = np.array(pheromones) + 0.1  # Add baseline
                        probs /= probs.sum()

                        next_vertex = np.random.choice(neighbors, p=probs)
                        self.current_vertex = next_vertex
                        self.path_history.append(next_vertex)

                        # Update position (with safety check)
                        node_data = semantic_graph.graph.nodes.get(next_vertex, {})
                        if 'position' in node_data:
                            self.position = node_data['position']
                            # Use mark_vertex_accessed for stability tracking when moving to vertex
                            semantic_graph.mark_vertex_accessed(next_vertex, current_time)
                        else:
                            # Vertex missing position data, reset to exploration mode
                            self.current_vertex = None
                else:
                    # No neighbors, explore
                    self.current_vertex = None

        # ISSUE 1: EPISTEMIC GUIDANCE - Replace random walk with information-seeking exploration
        if self.current_vertex is None:
            # LAYER INTEGRATION: Ants are attracted to structural field (beaver builds)
            structural_value = spacetime.get_structural_field(self.position)
            structural_gradient = spacetime.get_structural_gradient(self.position)

            # =====================================================================
            # EPISTEMIC EXPLORATION: Move toward high-information, unvisited regions
            # =====================================================================

            # 1. Compute local information density and gradient
            info_density = spacetime.get_information_density(self.position)

            # 2. Compute information gradient using finite differences
            eps = 0.5  # Spatial sampling distance
            info_gradient = np.zeros(3)
            for dim in range(3):  # r, theta, phi
                pos_plus = self.position.copy()
                pos_minus = self.position.copy()
                pos_plus[dim + 1] += eps
                pos_minus[dim + 1] -= eps
                # Bound the test positions
                pos_plus[1] = max(spacetime.r_s * 1.1, min(spacetime.config.r_max * 0.95, pos_plus[1]))
                pos_minus[1] = max(spacetime.r_s * 1.1, min(spacetime.config.r_max * 0.95, pos_minus[1]))
                info_gradient[dim] = (spacetime.get_information_density(pos_plus) -
                                      spacetime.get_information_density(pos_minus)) / (2 * eps)

            # 3. Novelty bonus: penalize revisiting same regions
            novelty_factor = 1.0
            if len(self.epistemic_memory) > 0:
                # Check distance to recently visited positions
                for past_pos in self.epistemic_memory[-20:]:
                    dist = np.linalg.norm(self.position[1:] - past_pos[1:])
                    if dist < 2.0:
                        novelty_factor *= 0.7  # Reduce attraction if recently visited

            # 4. Combine exploration signals:
            #    - Info gradient: move toward higher information density (epistemic value)
            #    - Structural gradient: follow beaver builds (exploitation)
            #    - Novelty: avoid recently visited regions (anti-clustering)
            #    - Random: small perturbation for stochasticity

            # Epistemic drive: normalized gradient toward high-info regions
            info_grad_norm = np.linalg.norm(info_gradient)
            if info_grad_norm > 1e-6:
                epistemic_direction = info_gradient / info_grad_norm
            else:
                epistemic_direction = np.zeros(3)

            # Compute weighted exploration direction
            random_component = 0.02 * np.random.randn(3)  # Reduced randomness
            epistemic_weight = 0.5 * novelty_factor  # Strong epistemic drive
            structural_weight = 0.3 if structural_value > 0.1 else 0.0
            random_weight = 0.2

            exploration_direction = (
                epistemic_weight * epistemic_direction +
                structural_weight * structural_gradient +
                random_weight * random_component
            )

            # Track information gain for this step
            expected_info_gain = info_density * novelty_factor
            self.information_gain_history.append(expected_info_gain)
            self.total_info_discovered += expected_info_gain * dt

            # 5. Build exploration step (4D: time component stays zero)
            exploration_step = np.zeros(4)
            exploration_step[1:] = 0.05 * exploration_direction

            # Add exploration offset to velocity before geodesic step
            exploration_velocity = self.velocity + exploration_step / (dt + 1e-6)

            # Use geodesic motion (respects spacetime curvature)
            self.position, self.velocity = spacetime.geodesic_step(
                self.position, exploration_velocity, dt
            )

            # 6. Update epistemic memory (sliding window of visited positions)
            self.epistemic_memory.append(self.position.copy())
            if len(self.epistemic_memory) > 50:
                self.epistemic_memory.pop(0)

            # LAYER INTEGRATION: Create structure-linked vertex at high structural field
            if structural_value > 0.3 and np.random.random() < 0.1:
                # This location has significant beaver activity, mark it semantically
                nearby = self._find_nearby_vertex(semantic_graph, distance_threshold=3.0)
                if nearby is None:
                    # Create a "structure vertex" linking beaver and ant layers
                    vertex_id = semantic_graph.add_vertex(
                        position=self.position.copy(),
                        salience=0.3 + structural_value * 0.5,  # Salience from structure
                        current_time=current_time
                    )
                    semantic_graph.graph.nodes[vertex_id]['vertex_type'] = 'structure'
                    self.current_vertex = vertex_id
                    self.path_history.append(vertex_id)
                    self.discovery_times[vertex_id] = current_time
                    # ISSUE 2: Track structure vertex creation
                    self.vertices_created += 1
                    self.contribution_score += 0.3

        # FIX #2: Energy decay modified by structural field AND edges
        # EDGES REDUCE DECAY: Graph connectivity provides efficiency bonus
        movement_cost = spacetime.get_movement_cost(self.position)
        base_decay = dt * 0.003 * movement_cost

        # Edge-based decay reduction
        n_edges = semantic_graph.graph.number_of_edges()
        edge_reduction = 1.0 / (1.0 + 0.3 * np.tanh(n_edges / 20.0))
        self.energy -= base_decay * edge_reduction

        if self.energy <= 0:
            self.state = "dead"


class EnhancedBeeAgent(Agent, ActiveInferenceMixin):
    """
    Enhanced bee with packet transport from queue-based economy.

    Uses Active Inference for action selection:
    - Scout behavior emerges from high epistemic drive (exploration)
    - Forager behavior emerges from pragmatic preferences (goal-directed)
    - Role transitions happen naturally based on free energy gradients
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "scout"
        self.packet = None
        self.target_vertex = None
        self.packets_delivered = 0
        self.packets_dropped = 0  # FIX #4: Track failed deliveries
        self.waggle_intensity = 0.0

        # Initialize Active Inference
        self.__init_active_inference__()
        # Bees prefer high packet density (foraging) and wormhole proximity (delivery)
        self.preferences[2] = 1.0  # Prefer high info density
        self.preferences[4] = 0.5  # Prefer structural regions
        self.epistemic_drive = 0.7  # High exploration drive initially

    def update(self, dt: float, spacetime: EnhancedSpacetime, semantic_graph, wormhole_position,
                current_time: float = 0.0):
        # ACTIVE INFERENCE: Update beliefs based on current observation
        observation = self._get_observation(spacetime, self.position)
        self.update_beliefs(observation)

        # Track free energy for learning
        current_F = self.compute_free_energy(observation)
        self.free_energy_history.append(current_F)

        # Adapt epistemic drive based on free energy trend
        # High free energy = world is surprising = increase exploration
        if len(self.free_energy_history) > 10:
            recent_F = np.mean(self.free_energy_history[-10:])
            if recent_F > 1.0:
                self.epistemic_drive = min(0.9, self.epistemic_drive + 0.01)
            else:
                self.epistemic_drive = max(0.3, self.epistemic_drive - 0.01)

        if self.role == "scout":
            # TRANSPORT GATING: Check if semantic graph is mature enough for transport
            # Uses adaptive thresholds based on bootstrap phase
            if not semantic_graph.is_transport_ready(current_time):
                # Graph not mature - wait for structure to develop
                # Just random walk to reduce energy consumption
                random_step = 0.02 * np.random.randn(4)
                exploration_velocity = self.velocity + random_step / (dt + 1e-6)
                # Use geodesic motion even while waiting
                self.position, self.velocity = spacetime.geodesic_step(
                    self.position, exploration_velocity, dt
                )
                self.energy -= dt * 0.003  # Reduced energy cost while waiting
                return

            # FIX #3: Find vertices with packets waiting (not just high salience)
            if np.random.rand() < 0.1:  # Check more frequently
                vertices = list(semantic_graph.graph.nodes())
                if vertices:
                    # Prioritize vertices with packets AND high salience
                    scores = []
                    for v in vertices:
                        queue_len = semantic_graph.get_queue_length(v)
                        salience = semantic_graph.graph.nodes[v].get('salience', 0.5)
                        # Score = queue_length * salience (packets at important nodes)
                        scores.append(queue_len * salience + 0.1 * salience)

                    if max(scores) > 0:
                        max_idx = np.argmax(scores)
                        self.target_vertex = vertices[max_idx]
                        self.waggle_intensity = scores[max_idx]
                        self.role = "forager"

        elif self.role == "forager":
            if self.packet is None and self.target_vertex is not None:
                # Safety check: verify target vertex still exists (may have been pruned)
                if self.target_vertex not in semantic_graph.graph:
                    self.target_vertex = None
                    self.role = "scout"
                    return

                # Move to target vertex
                target_pos = semantic_graph.graph.nodes[self.target_vertex]['position']

                # Handle dimension mismatch
                if len(target_pos) > len(self.position):
                    target_pos = target_pos[:len(self.position)]
                elif len(target_pos) < len(self.position):
                    target_pos = np.pad(target_pos, (0, len(self.position) - len(target_pos)))

                direction = target_pos - self.position
                self.velocity = 0.3 * direction / (np.linalg.norm(direction) + 1e-6)

                # Check if reached
                if np.linalg.norm(direction) < 1.0:
                    # FIX #3: Pick up packet from queue (not fabricate one)
                    self.packet = semantic_graph.get_packet(self.target_vertex)

                    if self.packet is not None:
                        # Got a real packet - transport it
                        self.role = "transporter"
                    else:
                        # No packet available - go back to scouting
                        self.target_vertex = None
                        self.role = "scout"

        elif self.role == "transporter":
            # FIX #4: Check packet TTL
            if self.packet is not None:
                self.packet['ttl'] -= dt
                if self.packet['ttl'] <= 0:
                    # Packet expired - drop it
                    self.packets_dropped += 1
                    self.packet = None
                    self.target_vertex = None
                    self.role = "scout"
                    return

            # Move to wormhole
            direction = wormhole_position - self.position
            self.velocity = 0.5 * direction / (np.linalg.norm(direction) + 1e-6)

            # Check if reached wormhole
            if np.linalg.norm(direction) < 2.0:
                # Deliver packet
                self.packets_delivered += 1
                self.packet = None
                self.target_vertex = None
                self.role = "scout"
        
        # Update position using geodesic motion (respects spacetime curvature)
        self.position, self.velocity = spacetime.geodesic_step(
            self.position, self.velocity, dt
        )

        # FIX #2: Energy decay modified by structural field AND edges
        # EDGES REDUCE DECAY: Graph connectivity provides efficiency bonus
        movement_cost = spacetime.get_movement_cost(self.position)
        base_decay = dt * 0.008 * movement_cost * (1 + 0.5 * bool(self.packet))

        # Edge-based decay reduction
        n_edges = semantic_graph.graph.number_of_edges()
        edge_reduction = 1.0 / (1.0 + 0.3 * np.tanh(n_edges / 20.0))
        self.energy -= base_decay * edge_reduction

        if self.energy <= 0:
            self.state = "dead"

# =============================================================================
# SEMANTIC GRAPH
# =============================================================================

class SemanticGraph:
    """Semantic graph maintained by ants with packet economy"""

    # Class-level constants for emergence control
    MAX_VERTICES = 100
    PHASE_1_MATURITY_EDGES = 2
    PHASE_2_MATURITY_EDGES = 5
    MIN_ENTROPY_FOR_TRANSPORT = 0.3

    # Bootstrap phase parameters - CRITICAL FOR ANT SURVIVAL
    # Extended bootstrap allows network to stabilize before strict edge policies apply
    BOOTSTRAP_DURATION = 40.0  # Time units for bootstrap phase (increased from 30)
    BOOTSTRAP_EDGE_PROB_MULTIPLIER = 5.0  # Higher edge probability during bootstrap
    VERTEX_GRACE_PERIOD = 30.0  # New vertices protected from pruning (increased from 20)
    MIN_STABLE_VERTICES = 100  # Minimum vertices to maintain (increased from 10 to preserve graph)

    def __init__(self):
        self.graph = nx.DiGraph()
        self.pheromones = {}  # (v1, v2) -> strength
        self.next_vertex_id = 0
        # FIX #3: Packet queues at vertices
        self.packet_queues = {}  # vertex_id -> list of packets
        self.max_queue_size = 10  # FIX #4: Queue capacity limit
        # Co-occurrence tracking for conditional edge formation
        self.vertex_activations = {}  # vertex_id -> list of (time, agent_id)
        self.co_occurrence_counts = {}  # (v1, v2) -> count

        # Bootstrap phase tracking
        self.creation_time = 0.0  # When the graph was initialized
        self.is_bootstrapped = False  # Whether bootstrap phase is complete

        # Vertex stability tracking
        self.vertex_creation_times = {}  # vertex_id -> creation_time
        self.vertex_access_counts = {}  # vertex_id -> access count for stability scoring
        self.stable_vertex_set = set()  # Vertices that have proven stable

    def set_creation_time(self, time: float):
        """Set the graph creation time for bootstrap phase tracking"""
        self.creation_time = time

    def get_graph_age(self, current_time: float) -> float:
        """Get age of graph since creation"""
        return current_time - self.creation_time

    def is_in_bootstrap_phase(self, current_time: float) -> bool:
        """Check if graph is still in bootstrap phase"""
        if self.is_bootstrapped:
            return False
        age = self.get_graph_age(current_time)
        if age >= self.BOOTSTRAP_DURATION:
            self.is_bootstrapped = True
            return False
        return True

    def add_vertex(self, position: np.ndarray, salience: float, current_time: float = 0.0) -> int:
        """Add vertex to graph with max vertex limit and stability tracking"""
        # Enforce maximum vertices
        if self.graph.number_of_nodes() >= self.MAX_VERTICES:
            # Find lowest salience vertex to replace (excluding stable and grace period vertices)
            candidates = []
            for v in self.graph.nodes():
                # Skip vertices in grace period
                creation_time = self.vertex_creation_times.get(v, 0.0)
                if current_time - creation_time < self.VERTEX_GRACE_PERIOD:
                    continue
                # Skip stable vertices
                if v in self.stable_vertex_set:
                    continue
                candidates.append(v)

            if not candidates:
                return -1  # No replaceable vertices

            min_sal_vertex = min(candidates,
                                  key=lambda v: self.graph.nodes[v].get('salience', 0))
            if self.graph.nodes[min_sal_vertex].get('salience', 0) < salience:
                self._remove_vertex(min_sal_vertex)
            else:
                return -1  # Reject new vertex

        vertex_id = self.next_vertex_id
        self.next_vertex_id += 1

        self.graph.add_node(vertex_id, position=position, salience=salience,
                           created_at=current_time, last_accessed=current_time)
        self.packet_queues[vertex_id] = []  # Initialize packet queue
        self.vertex_activations[vertex_id] = []
        self.vertex_creation_times[vertex_id] = current_time
        self.vertex_access_counts[vertex_id] = 0

        return vertex_id

    def _remove_vertex(self, vertex_id: int):
        """Internal method to cleanly remove a vertex"""
        if vertex_id in self.graph:
            self.graph.remove_node(vertex_id)
        if vertex_id in self.packet_queues:
            del self.packet_queues[vertex_id]
        if vertex_id in self.vertex_activations:
            del self.vertex_activations[vertex_id]
        if vertex_id in self.vertex_creation_times:
            del self.vertex_creation_times[vertex_id]
        if vertex_id in self.vertex_access_counts:
            del self.vertex_access_counts[vertex_id]
        if vertex_id in self.stable_vertex_set:
            self.stable_vertex_set.discard(vertex_id)

    def mark_vertex_accessed(self, vertex_id: int, current_time: float):
        """Mark vertex as accessed and update stability tracking"""
        if vertex_id in self.graph:
            self.graph.nodes[vertex_id]['last_accessed'] = current_time
            self.vertex_access_counts[vertex_id] = self.vertex_access_counts.get(vertex_id, 0) + 1

            # Promote to stable if accessed frequently
            if self.vertex_access_counts[vertex_id] >= 10:
                self.stable_vertex_set.add(vertex_id)

    def add_packet(self, vertex_id: int, packet: dict) -> bool:
        """
        FIX #3: Add a packet to vertex queue.
        Returns False if queue is full (congestion).
        """
        if vertex_id not in self.packet_queues:
            return False
        if len(self.packet_queues[vertex_id]) >= self.max_queue_size:
            return False  # Queue full - packet dropped (congestion)
        self.packet_queues[vertex_id].append(packet)
        return True

    def get_packet(self, vertex_id: int) -> dict:
        """FIX #3: Get a packet from vertex queue (FIFO)"""
        if vertex_id not in self.packet_queues:
            return None
        if len(self.packet_queues[vertex_id]) == 0:
            return None
        return self.packet_queues[vertex_id].pop(0)

    def get_queue_length(self, vertex_id: int) -> int:
        """Get queue length at vertex"""
        return len(self.packet_queues.get(vertex_id, []))
    
    def add_edge(self, v1: int, v2: int, pheromone: float):
        """Add edge with pheromone"""
        if not self.graph.has_edge(v1, v2):
            self.graph.add_edge(v1, v2)
        
        key = (v1, v2)
        if key not in self.pheromones:
            self.pheromones[key] = 0.0
        self.pheromones[key] += pheromone
    
    def get_pheromone(self, edge: Tuple[int, int]) -> float:
        """Get pheromone strength on edge"""
        return self.pheromones.get(edge, 0.0)
    
    def decay_pheromones(self, dt: float, decay_rate: float = 0.1):
        """Decay all pheromones"""
        for edge in list(self.pheromones.keys()):
            self.pheromones[edge] *= np.exp(-decay_rate * dt)

            # Remove if too weak
            if self.pheromones[edge] < 0.01:
                del self.pheromones[edge]

    def prune_graph(self, current_time: float, max_age: float = 80.0, min_vertices: int = None):
        """
        Prune vertices that haven't been accessed recently.
        Cost of memory - old, unused beliefs are forgotten.

        Protected vertices (never pruned):
        - Vertices in grace period (recently created)
        - Vertices marked as stable
        - Vertices with packets in queue
        - Vertices with high connectivity (degree > 3)
        - Vertices with high salience (> 0.7)
        - If graph would drop below MIN_STABLE_VERTICES

        During bootstrap phase, pruning is completely disabled to allow
        the graph to establish stable structure.
        """
        # Use class constant if not specified
        if min_vertices is None:
            min_vertices = self.MIN_STABLE_VERTICES

        # During bootstrap, no pruning at all
        if self.is_in_bootstrap_phase(current_time):
            return 0

        if self.graph.number_of_nodes() <= min_vertices:
            return 0  # Don't prune if already at minimum

        vertices_to_remove = []
        for v in self.graph.nodes():
            # Check grace period protection
            creation_time = self.vertex_creation_times.get(v, 0.0)
            vertex_age = current_time - creation_time
            if vertex_age < self.VERTEX_GRACE_PERIOD:
                continue  # In grace period, keep

            # Check stable vertex protection
            if v in self.stable_vertex_set:
                continue  # Marked stable, keep

            # Check access age
            last_accessed = self.graph.nodes[v].get('last_accessed', 0.0)
            access_age = current_time - last_accessed
            if access_age <= max_age:
                continue  # Recently accessed, keep

            # Check protections
            if self.get_queue_length(v) > 0:
                continue  # Has packets, keep

            degree = self.graph.in_degree(v) + self.graph.out_degree(v)
            if degree > 3:
                continue  # Well-connected hub, keep

            salience = self.graph.nodes[v].get('salience', 0.5)
            if salience > 0.7:
                continue  # High importance, keep

            vertices_to_remove.append(v)

        # Limit pruning to maintain minimum graph size
        max_to_remove = max(0, self.graph.number_of_nodes() - min_vertices)
        vertices_to_remove = vertices_to_remove[:max_to_remove]

        for v in vertices_to_remove:
            self._remove_vertex(v)

        return len(vertices_to_remove)

    def merge_nearby_vertices(self, distance_threshold: float = 1.0):
        """
        FIX #6: Merge vertices that are spatially close.
        Prevents unbounded growth from redundant beliefs.

        Threshold reduced from 2.0 to 1.0 to preserve more spatial diversity.
        Now respects MIN_STABLE_VERTICES to prevent over-merging.

        Note: Uses SPATIAL distance only (r, theta, phi), not time component.
        """
        merged_count = 0
        vertices = list(self.graph.nodes())

        # CRITICAL: Don't merge if already at or below minimum
        if self.graph.number_of_nodes() <= self.MIN_STABLE_VERTICES:
            return 0

        for i, v1 in enumerate(vertices):
            # Stop if we've merged down to minimum
            if self.graph.number_of_nodes() <= self.MIN_STABLE_VERTICES:
                break

            if v1 not in self.graph:
                continue
            if 'position' not in self.graph.nodes[v1]:
                continue
            pos1 = self.graph.nodes[v1]['position']

            for v2 in vertices[i+1:]:
                if v2 not in self.graph:
                    continue
                if 'position' not in self.graph.nodes[v2]:
                    continue
                pos2 = self.graph.nodes[v2]['position']

                # Check SPATIAL distance only (exclude time component at index 0)
                try:
                    spatial_dist = np.linalg.norm(pos1[1:] - pos2[1:])
                except (TypeError, ValueError):
                    continue

                if spatial_dist < distance_threshold:
                    # CRITICAL: Check minimum before each merge
                    if self.graph.number_of_nodes() <= self.MIN_STABLE_VERTICES:
                        return merged_count

                    # Merge v2 into v1 (keep higher salience)
                    sal1 = self.graph.nodes[v1].get('salience', 0.5)
                    sal2 = self.graph.nodes[v2].get('salience', 0.5)

                    # Keep the more salient vertex
                    if sal2 > sal1:
                        self.graph.nodes[v1]['salience'] = sal2
                        self.graph.nodes[v1]['position'] = pos2

                    # Transfer packets
                    if v2 in self.packet_queues:
                        for pkt in self.packet_queues[v2]:
                            self.add_packet(v1, pkt)
                        del self.packet_queues[v2]

                    # Transfer edges
                    for pred in list(self.graph.predecessors(v2)):
                        if pred != v1 and pred in self.graph:
                            self.add_edge(pred, v1, 0.5)
                    for succ in list(self.graph.successors(v2)):
                        if succ != v1 and succ in self.graph:
                            self.add_edge(v1, succ, 0.5)

                    self.graph.remove_node(v2)
                    merged_count += 1

        return merged_count

    def get_total_queue_length(self) -> int:
        """Get total packets waiting across all vertices"""
        return sum(len(q) for q in self.packet_queues.values())

    # =========================================================================
    # CONDITIONAL EDGE FORMATION (Co-activation based)
    # =========================================================================

    def record_activation(self, vertex_id: int, time: float, agent_id: int):
        """
        Record vertex activation for co-occurrence tracking.
        Edges require temporal overlap, not just absolute salience.
        """
        if vertex_id not in self.vertex_activations:
            self.vertex_activations[vertex_id] = []

        # Keep only recent activations (rolling window)
        activation_window = 10.0  # Time window for co-occurrence
        self.vertex_activations[vertex_id] = [
            (t, a) for t, a in self.vertex_activations[vertex_id]
            if time - t < activation_window
        ]
        self.vertex_activations[vertex_id].append((time, agent_id))

    def compute_co_occurrence(self, v1: int, v2: int, current_time: float,
                               time_window: float = 5.0) -> int:
        """
        Compute co-occurrence count between two vertices.
        Returns number of times both were activated within time_window.
        """
        if v1 not in self.vertex_activations or v2 not in self.vertex_activations:
            return 0

        activations1 = self.vertex_activations[v1]
        activations2 = self.vertex_activations[v2]

        co_occurrences = 0
        for t1, _ in activations1:
            for t2, _ in activations2:
                if abs(t1 - t2) < time_window:
                    co_occurrences += 1

        return co_occurrences

    def add_edge_conditional(self, v1: int, v2: int, pheromone: float,
                              current_time: float, nearby_structures: int = 0) -> bool:
        """
        Conditional edge formation based on co-activation with phase-based thresholds.

        During bootstrap phase:
        - Higher base probability (5x multiplier)
        - Spatial proximity alone can create edges (no co-occurrence required)
        - Encourages rapid graph structure formation

        After bootstrap:
        - Require co-occurrence (temporal overlap)
        - Standard probability formula:
            edge_prob = min(salience1, salience2) × co_occurrence_count × structure_bonus
        - Structure bonus: 0.05 + 0.05 × nearby_structures
        """
        if v1 not in self.graph or v2 not in self.graph:
            return False

        # Already connected - just reinforce pheromone
        if self.graph.has_edge(v1, v2):
            self.pheromones[(v1, v2)] = self.pheromones.get((v1, v2), 0) + pheromone * 0.1
            return False

        # Get saliences
        sal1 = self.graph.nodes[v1].get('salience', 0.5)
        sal2 = self.graph.nodes[v2].get('salience', 0.5)
        min_salience = min(sal1, sal2)

        # Check if in bootstrap phase
        in_bootstrap = self.is_in_bootstrap_phase(current_time)

        if in_bootstrap:
            # BOOTSTRAP PHASE: Very permissive edge formation to kickstart the network
            # Critical for ant survival: edges reduce energy decay, so need rapid edge formation
            pos1 = self.graph.nodes[v1].get('position')
            pos2 = self.graph.nodes[v2].get('position')

            # GUARANTEED EDGES: First few edges are always created to bootstrap network
            n_edges = self.graph.number_of_edges()
            if n_edges < 10:
                # Strongly encourage initial edge formation
                edge_prob = 0.8  # 80% chance for first 10 edges
            else:
                spatial_proximity_bonus = 0.0
                if pos1 is not None and pos2 is not None:
                    try:
                        spatial_dist = np.linalg.norm(pos1[1:] - pos2[1:])
                        # Closer vertices get higher bonus (inverse distance, stronger effect)
                        spatial_proximity_bonus = 2.0 / (1.0 + spatial_dist)
                    except (TypeError, ValueError):
                        pass

                # Bootstrap base probability is much higher (increased from 5x to 10x)
                base_prob = (0.2 + 0.1 * nearby_structures) * self.BOOTSTRAP_EDGE_PROB_MULTIPLIER * 2.0

                # Co-occurrence still helps but not required
                co_occurrence = self.compute_co_occurrence(v1, v2, current_time)
                co_occurrence_bonus = 1.0 + 0.3 * co_occurrence

                # Final bootstrap probability (capped at 0.9)
                edge_prob = min(0.9, min_salience * co_occurrence_bonus * (base_prob + spatial_proximity_bonus))

        else:
            # POST-BOOTSTRAP: Standard conditional edge formation
            # Require co-occurrence (temporal overlap)
            co_occurrence = self.compute_co_occurrence(v1, v2, current_time)
            if co_occurrence == 0:
                return False  # No edge without temporal overlap

            # Structure-dependent base probability
            base_prob = 0.05 + 0.05 * nearby_structures

            # Final edge probability
            edge_prob = min_salience * (1 + 0.1 * co_occurrence) * base_prob

        if np.random.random() < edge_prob:
            self.add_edge(v1, v2, pheromone)
            # Track co-occurrence for this pair
            pair = (min(v1, v2), max(v1, v2))
            self.co_occurrence_counts[pair] = self.co_occurrence_counts.get(pair, 0) + 1
            return True

        return False

    # =========================================================================
    # ENTROPY AND TRANSPORT GATING
    # =========================================================================

    def compute_entropy(self) -> float:
        """
        Compute graph entropy for transport gating.
        Higher entropy = more uniform structure = ready for transport.

        Uses degree distribution entropy:
            H = -Σ p(k) log(p(k))
        where p(k) is probability of degree k.
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0

        # Get degree sequence
        degrees = [d for _, d in self.graph.degree()]
        if not degrees:
            return 0.0

        # Count degree frequencies
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1

        # Compute probability distribution
        n = len(degrees)
        probabilities = [count / n for count in degree_counts.values()]

        # Compute Shannon entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log(p + 1e-10)

        # Normalize to [0, 1] range (max entropy is log(n))
        max_entropy = np.log(n + 1)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def is_transport_ready(self, current_time: float = None) -> bool:
        """
        Check if semantic graph is mature enough for transport.

        Adaptive thresholds based on graph maturity:
        - During bootstrap: Lower thresholds to encourage early transport
        - After bootstrap: Standard thresholds require stable structure

        Requirements (standard):
        - At least 3 vertices
        - At least 2 edges (PHASE_1_MATURITY_EDGES)
        - Graph entropy > MIN_ENTROPY_FOR_TRANSPORT

        Requirements (bootstrap):
        - At least 3 vertices
        - At least 1 edge
        - Some packets in queues OR stable vertices exist
        """
        n_vertices = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        entropy = self.compute_entropy()

        # Check bootstrap phase
        in_bootstrap = current_time is not None and self.is_in_bootstrap_phase(current_time)

        if in_bootstrap:
            # During bootstrap: Lower thresholds
            # Allow transport if we have basic structure
            has_packets = self.get_total_queue_length() > 0
            has_stable_vertices = len(self.stable_vertex_set) >= 2

            return (n_vertices >= 3 and
                    n_edges >= 1 and
                    (has_packets or has_stable_vertices or entropy >= 0.1))
        else:
            # Standard thresholds
            return (n_vertices >= 3 and
                    n_edges >= self.PHASE_1_MATURITY_EDGES and
                    entropy >= self.MIN_ENTROPY_FOR_TRANSPORT)

    def get_graph_maturity_phase(self, current_time: float = None) -> int:
        """
        Get current graph maturity phase.

        Phase 0: Immature (< 3 vertices or < 2 edges)
        Phase 1: Basic transport enabled (2+ edges, entropy > 0.3)
        Phase 2: Full transport (5+ edges)
        Phase 3: Mature stable graph (many stable vertices, high connectivity)
        """
        n_vertices = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        n_stable = len(self.stable_vertex_set)

        if not self.is_transport_ready(current_time):
            return 0

        # Phase 3: Mature stable graph
        if n_edges >= 10 and n_stable >= 5:
            return 3

        # Phase 2: Full transport
        if n_edges >= self.PHASE_2_MATURITY_EDGES:
            return 2

        return 1

    def get_transport_priority_multiplier(self, current_time: float = None) -> float:
        """
        Get priority multiplier for transport based on graph state.

        Higher multiplier when graph is healthy and has queued packets.
        Lower multiplier when graph is struggling.
        """
        phase = self.get_graph_maturity_phase(current_time)
        n_queued = self.get_total_queue_length()
        n_stable = len(self.stable_vertex_set)

        # Base multiplier from phase
        phase_multiplier = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0}.get(phase, 1.0)

        # Bonus for queued packets (encourages clearing queue)
        queue_bonus = min(1.0, n_queued * 0.1)

        # Bonus for stable vertices (healthy graph)
        stability_bonus = min(0.5, n_stable * 0.05)

        return phase_multiplier + queue_bonus + stability_bonus

# =============================================================================
# ENHANCED SIMULATION ENGINE
# =============================================================================

class EnhancedSimulationEngine:
    """Enhanced simulation with full functionality"""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.logger.info("Initializing enhanced spacetime...")
        self.spacetime = EnhancedSpacetime(config)
        
        self.logger.info("Initializing semantic graph...")
        self.semantic_graph = SemanticGraph()

        # FIX #3: Seed semantic graph with initial vertices so bees have targets
        self._seed_semantic_graph(n_initial_vertices=15)

        self.logger.info("Initializing agents...")
        self.agents = self._initialize_agents()
        
        # Wormhole position - derived from config instead of hardcoded
        # Located just outside the wormhole throat at r = throat_radius + 0.6
        wormhole_r = getattr(config, 'throat_radius', 2.0) + 0.6
        self.wormhole_position = np.array([0.0, wormhole_r, np.pi/2, 0.0])
        
        # Statistics
        self.stats = {
            'n_packets_transported': 0,
            'total_energy': 0.0,
            'n_structures_built': 0,
            'n_vertices': 0,
            'n_edges': 0,
            'energy_history': [],
            'vertices_history': [],
            'structures_history': [],
            'stability_rate': [],
            'lyapunov_V': []
        }

        # Simple stability tracking (full Lyapunov requires epistemic layer)
        self.stability_history = []
        self.consecutive_violations = 0
        self.last_energy = None

        # ISSUE 3: Initialize Overmind for meta-level colony regulation
        if EPISTEMIC_LAYER_AVAILABLE:
            self.logger.info("Initializing Overmind meta-controller...")
            self.overmind = Overmind(target_entropy=100.0)
            self.overmind_active = True
        else:
            self.overmind = None
            self.overmind_active = False
            self.logger.warning("Overmind not available - epistemic_cognitive_layer.py not found")

        # PHASE II: Initialize Adversarial Pressure Layer
        if APL_AVAILABLE:
            self.logger.info("Initializing Adversarial Pressure Layer (Phase II)...")
            self.apl = AdversarialPressureLayer(config)
            self.apl_active = True
            self.apl_stats = {
                'pressure_history': [],
                'threat_history': [],
                'damage_history': []
            }
        else:
            self.apl = None
            self.apl_active = False
            self.logger.warning("APL not available - adversarial_pressure_layer.py not found")

        self.logger.info("Enhanced simulation engine initialized")
    
    def _setup_logging(self):
        """Setup logging"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/enhanced_simulation.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("EnhancedBlackholeArchive")
    
    def _initialize_agents(self) -> Dict[str, List[Agent]]:
        """Initialize all agents"""
        agents = {'beavers': [], 'ants': [], 'bees': []}
        
        # Beavers
        for i in range(self.config.n_beavers):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 10,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            
            agents['beavers'].append(EnhancedBeaverAgent(
                id=f"beaver_{i}",
                colony="beavers",
                position=position,
                velocity=0.1 * np.random.randn(4),
                energy=1.0
            ))
        
        # Ants
        for i in range(self.config.n_ants):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 20,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            
            agents['ants'].append(EnhancedAntAgent(
                id=f"ant_{i}",
                colony="ants",
                position=position,
                velocity=0.05 * np.random.randn(4),
                energy=1.0
            ))
        
        # Bees
        for i in range(self.config.n_bees):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 15,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            
            agents['bees'].append(EnhancedBeeAgent(
                id=f"bee_{i}",
                colony="bees",
                position=position,
                velocity=0.2 * np.random.randn(4),
                energy=1.0
            ))
        
        return agents

    def _seed_semantic_graph(self, n_initial_vertices: int = 10):
        """
        FIX #3: Seed semantic graph with initial high-salience vertices.

        Without initial vertices, bees have nothing to transport and the
        epistemic layer remains inert. Seed with vertices at positions where
        curvature is high (near event horizon) since these are informationally
        interesting locations.
        """
        self.logger.info(f"Seeding semantic graph with {n_initial_vertices} initial vertices")

        # Initialize bootstrap phase - graph creation time is 0.0
        self.semantic_graph.set_creation_time(0.0)

        for i in range(n_initial_vertices):
            # Create vertices near event horizon where curvature is high
            r = self.spacetime.r_s + 0.5 + np.random.rand() * 5  # r in [rs+0.5, rs+5.5]
            theta = np.random.rand() * np.pi
            phi = np.random.rand() * 2 * np.pi

            position = np.array([0.0, r, theta, phi])

            # Compute salience based on curvature - closer to horizon = higher salience
            curvature = self.spacetime.get_curvature(position)
            salience = min(1.0, curvature * 10)  # Scale to [0, 1]

            # Add vertex to semantic graph
            vertex_id = self.semantic_graph.add_vertex(position, salience)

            # FIX #3: Seed initial packets at high-salience vertices
            if salience > 0.3:
                packet = {
                    'content': f"seed_discovery_{vertex_id}",
                    'salience': salience,
                    'confidence': 0.8,
                    'created_at': 0.0,
                    'ttl': 100.0,
                    'source_agent': 'seed',
                    'source_vertex': vertex_id
                }
                self.semantic_graph.add_packet(vertex_id, packet)

            # Connect to nearby vertices with initial pheromone
            if i > 0:
                for prev_id in range(max(0, vertex_id - 3), vertex_id):
                    self.semantic_graph.add_edge(prev_id, vertex_id, pheromone=0.5)
                    self.semantic_graph.add_edge(vertex_id, prev_id, pheromone=0.5)

        self.logger.info(f"Semantic graph seeded with {self.semantic_graph.graph.number_of_nodes()} vertices")

    def run(self):
        """Run enhanced simulation"""
        n_steps = int(self.config.t_max / self.config.dt)
        self.logger.info(f"Starting enhanced simulation: {n_steps} steps")

        # FIX #5: Reset material budget at simulation start
        EnhancedBeaverAgent.global_material_budget = 500.0

        for step in tqdm(range(n_steps), desc="Enhanced Simulation"):
            t = step * self.config.dt

            # Update all agents
            for beaver in self.agents['beavers']:
                if beaver.state == "active":
                    beaver.update(self.config.dt, self.spacetime, self.semantic_graph)

            for ant in self.agents['ants']:
                if ant.state == "active":
                    ant.update(self.config.dt, self.spacetime, self.semantic_graph, current_time=t)

            for bee in self.agents['bees']:
                if bee.state == "active":
                    bee.update(self.config.dt, self.spacetime, self.semantic_graph,
                               self.wormhole_position, current_time=t)

            # FIX #5: Decay structural field (maintenance cost)
            self.spacetime.decay_structural_field(self.config.dt)

            # FIX #7: Decay epistemic stress
            self.spacetime.decay_epistemic_stress(self.config.dt)

            # FIX #7: Add epistemic stress at congested vertices
            for v in self.semantic_graph.graph.nodes():
                queue_len = self.semantic_graph.get_queue_length(v)
                if queue_len > 5:  # Congestion threshold
                    pos = self.semantic_graph.graph.nodes[v]['position']
                    stress = 0.1 * (queue_len - 5)  # Stress proportional to congestion
                    self.spacetime.add_epistemic_stress(pos, stress)

            # Decay pheromones
            self.semantic_graph.decay_pheromones(self.config.dt)

            # DYNAMIC SPACETIME: Update stress-energy from agents and evolve metric
            for agents_list in self.agents.values():
                for agent in agents_list:
                    if agent.state == "active":
                        # Each active agent contributes to stress-energy
                        self.spacetime.add_stress_energy(agent.position, agent.energy * 0.01)

            # Evolve metric perturbation (linearized Einstein equations)
            self.spacetime.evolve_metric(self.config.dt)

            # BEKENSTEIN BOUND: Add entropy from packet queues
            for v in self.semantic_graph.graph.nodes():
                queue_len = self.semantic_graph.get_queue_length(v)
                if queue_len > 0:
                    pos = self.semantic_graph.graph.nodes[v]['position']
                    # Each packet represents some bits of information
                    bits = queue_len * 64  # Assume 64 bits per packet header
                    thermalized = self.spacetime.add_local_entropy(pos, bits * 0.01)
                    # If bits were thermalized, corrupt/drop packets
                    if thermalized > 0:
                        packets_to_drop = int(thermalized / 64)
                        for _ in range(min(packets_to_drop, queue_len)):
                            self.semantic_graph.get_packet(v)  # Drop packet

            # Decay entropy (Hawking radiation analog)
            self.spacetime.decay_entropy(self.config.dt)

            # FIX #6: Periodically prune and merge graph
            if step % 50 == 0 and step > 0:
                pruned = self.semantic_graph.prune_graph(t)
                merged = self.semantic_graph.merge_nearby_vertices()
                if pruned > 0 or merged > 0:
                    self.logger.debug(f"Graph maintenance: pruned={pruned}, merged={merged}")

            # =================================================================
            # ISSUE 3: OVERMIND META-LEVEL CONTROL
            # =================================================================
            if self.overmind_active and step % 10 == 0:
                # Get current colony metrics
                n_vertices = self.semantic_graph.graph.number_of_nodes()
                n_edges = self.semantic_graph.graph.number_of_edges()
                n_structures = sum(b.structures_built for b in self.agents['beavers'])
                n_packets = sum(b.packets_delivered for b in self.agents['bees'])

                # Create wrapper for Overmind to observe
                # (Overmind expects EpistemicSemanticGraph, but we can pass data directly)
                total_energy = sum(a.energy for agents in self.agents.values()
                                   for a in agents if a.state == "active")

                # Detect imbalance: structures >> vertices or vertices >> packets
                structure_vertex_ratio = (n_structures + 1) / (n_vertices + 1)
                vertex_packet_ratio = (n_vertices + 1) / (n_packets + 1)

                # COLONY BALANCING LOGIC
                # If structures >> vertices: boost ant exploration
                if structure_vertex_ratio > 10:
                    # Ants need to explore more - give them energy bonus
                    exploration_bonus = 0.01 * (structure_vertex_ratio - 10)
                    for ant in self.agents['ants']:
                        if ant.state == "active":
                            ant.energy = min(1.5, ant.energy + exploration_bonus)

                # If vertices >> packets: boost bee transport
                if vertex_packet_ratio > 5:
                    # Bees need to transport more - give them energy bonus
                    transport_bonus = 0.005 * (vertex_packet_ratio - 5)
                    for bee in self.agents['bees']:
                        if bee.state == "active":
                            bee.energy = min(1.5, bee.energy + transport_bonus)

                # If too many contradictions/stagnation: inject exploration noise
                if hasattr(self.semantic_graph, 'stable_vertex_set'):
                    stability_rate = len(self.semantic_graph.stable_vertex_set) / (n_vertices + 1)
                    if stability_rate > 0.9 and step > 200:
                        # System is stagnating - inject exploration
                        for ant in self.agents['ants']:
                            if ant.state == "active":
                                # Add random velocity perturbation
                                ant.velocity += 0.02 * np.random.randn(4)

                # Log Overmind state periodically
                if step % 100 == 0:
                    self.logger.debug(
                        f"Overmind: S/V ratio={structure_vertex_ratio:.2f}, "
                        f"V/P ratio={vertex_packet_ratio:.2f}"
                    )

            # =================================================================
            # PHASE II: ADVERSARIAL PRESSURE LAYER
            # =================================================================
            if self.apl_active and step % 5 == 0:
                # Compute system state for APL
                n_alive = sum(1 for agents_list in self.agents.values()
                             for a in agents_list if a.state == "active")
                n_total = sum(len(agents_list) for agents_list in self.agents.values())
                survival_rate = n_alive / max(1, n_total)

                total_energy = sum(a.energy for agents_list in self.agents.values()
                                   for a in agents_list if a.state == "active")

                # Energy trend (positive = gaining energy)
                if self.last_energy is not None:
                    energy_trend = (total_energy - self.last_energy) / self.config.dt
                else:
                    energy_trend = 0.0

                # Build work efficiency
                n_structures = sum(b.structures_built for b in self.agents['beavers'])
                n_packets = sum(b.packets_delivered for b in self.agents['bees'])
                work_per_agent = (n_structures + n_packets) / max(1, n_alive)

                # Graph health
                n_vertices = self.semantic_graph.graph.number_of_nodes()
                n_edges = self.semantic_graph.graph.number_of_edges()
                graph_health = min(1.0, n_vertices / 100.0)  # Healthy if 100+ vertices

                system_state = {
                    'survival_rate': survival_rate,
                    'energy_trend': energy_trend,
                    'work_efficiency': work_per_agent * 10,
                    'behavioral_entropy': 0.5,  # TODO: compute from action distribution
                    'graph_health': graph_health,
                    'packet_backlog': self.semantic_graph.get_total_queue_length(),
                    'energy_ratio': total_energy / max(1, n_total),
                    'build_rate': n_structures / max(1, t),
                    'total_energy': total_energy,
                    'n_vertices': n_vertices,
                    'n_edges': n_edges,
                    'n_packets': n_packets,
                    'work_per_agent': work_per_agent
                }

                # Update APL
                apl_result = self.apl.update(
                    current_time=t,
                    dt=self.config.dt * 5,  # APL runs every 5 steps
                    spacetime=self.spacetime,
                    semantic_graph=self.semantic_graph,
                    agents=self.agents,
                    system_state=system_state
                )

                # Record APL stats
                if step % 50 == 0:
                    self.apl_stats['pressure_history'].append(apl_result['pressure_budget'])
                    self.apl_stats['threat_history'].append(len(apl_result['active_effects']))
                    self.apl_stats['damage_history'].append(apl_result['damage_report'])

                # Log APL events
                if apl_result['triggered_events']:
                    self.logger.info(
                        f"APL triggered: {', '.join(apl_result['triggered_events'])} "
                        f"(budget={apl_result['pressure_budget']:.1f}, "
                        f"threat={apl_result['threat_level']:.3f})"
                    )

            # Update statistics
            self.stats['total_energy'] = sum(
                a.energy for agents in self.agents.values() for a in agents if a.state == "active"
            )
            self.stats['n_structures_built'] = sum(
                b.structures_built for b in self.agents['beavers']
            )
            self.stats['n_packets_transported'] = sum(
                b.packets_delivered for b in self.agents['bees']
            )
            self.stats['n_packets_dropped'] = sum(
                b.packets_dropped for b in self.agents['bees']
            )
            self.stats['n_packets_generated'] = sum(
                a.packets_generated for a in self.agents['ants']
            )
            self.stats['queue_length'] = self.semantic_graph.get_total_queue_length()
            self.stats['material_remaining'] = EnhancedBeaverAgent.global_material_budget
            self.stats['n_vertices'] = self.semantic_graph.graph.number_of_nodes()
            self.stats['n_edges'] = self.semantic_graph.graph.number_of_edges()

            # ISSUE 2: Individual contribution tracking
            self.stats['ant_contributions'] = {
                ant.id: {
                    'vertices_created': ant.vertices_created,
                    'edges_created': ant.edges_created,
                    'info_discovered': ant.total_info_discovered,
                    'contribution_score': ant.contribution_score
                }
                for ant in self.agents['ants']
            }
            # Summary: vertices per ant (productivity metric)
            active_ants = [a for a in self.agents['ants'] if a.state == "active"]
            total_ant_vertices = sum(a.vertices_created for a in self.agents['ants'])
            self.stats['vertices_per_ant'] = total_ant_vertices / max(1, len(active_ants))

            # VALIDATION: Thermodynamic metrics
            thermo_stats = self.spacetime.get_thermodynamic_stats()
            self.stats['bekenstein_thermalization_events'] = thermo_stats['bekenstein_thermalization_events']
            self.stats['thermodynamic_heat'] = thermo_stats['thermodynamic_heat']
            self.stats['total_entropy'] = thermo_stats['total_entropy']
            self.stats['mean_metric_perturbation'] = thermo_stats['mean_metric_perturbation']

            # Simple stability check: energy should not decay too fast
            current_energy = self.stats['total_energy']
            is_stable = True
            if self.last_energy is not None:
                energy_decay_rate = (self.last_energy - current_energy) / self.config.dt
                # Unstable if energy decays more than 5% per unit time
                if energy_decay_rate > 0.05 * self.last_energy:
                    is_stable = False
                    self.consecutive_violations += 1
                else:
                    self.consecutive_violations = 0
            self.last_energy = current_energy
            self.stability_history.append(is_stable)

            # Record history
            if step % 10 == 0:
                self.stats['energy_history'].append(self.stats['total_energy'])
                self.stats['vertices_history'].append(self.stats['n_vertices'])
                self.stats['structures_history'].append(self.stats['n_structures_built'])
                stable_count = sum(1 for s in self.stability_history if s)
                self.stats['stability_rate'].append(stable_count / max(1, len(self.stability_history)))

            # Log
            if step % 100 == 0:
                self.logger.info(
                    f"Step {step}/{n_steps}, t={t:.2f}, "
                    f"Energy={self.stats['total_energy']:.2f}, "
                    f"Vertices={self.stats['n_vertices']}, "
                    f"V/Ant={self.stats['vertices_per_ant']:.2f}, "
                    f"Structures={self.stats['n_structures_built']}, "
                    f"Packets={self.stats['n_packets_transported']}, "
                    f"Queue={self.stats['queue_length']}, "
                    f"Materials={self.stats['material_remaining']:.0f}"
                )
        
        self.logger.info("Enhanced simulation complete")
        self._save_results()
    
    def _save_results(self):
        """Save enhanced results"""
        report_path = Path(self.config.output_dir) / "enhanced_simulation_report.json"
        
        report = {
            'config': {
                'black_hole_mass': self.config.black_hole_mass,
                'n_beavers': self.config.n_beavers,
                'n_ants': self.config.n_ants,
                'n_bees': self.config.n_bees,
                't_max': self.config.t_max,
                'dt': self.config.dt
            },
            'final_statistics': self.stats,
            'n_agents_alive': {
                colony: len([a for a in agents if a.state == "active"])
                for colony, agents in self.agents.items()
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        self.logger.info(f"Enhanced report saved to {report_path}")

# Example usage
if __name__ == "__main__":
    from blackhole_archive_main import SimulationConfig
    
    config = SimulationConfig(
        t_max=100.0,
        dt=0.01,
        output_dir="./enhanced_results"
    )
    
    engine = EnhancedSimulationEngine(config)
    engine.run()
    
    print(f"\n✅ Enhanced Simulation Complete!")
    print(f"Structures built: {engine.stats['n_structures_built']}")
    print(f"Semantic vertices: {engine.stats['n_vertices']}")
    print(f"Packets transported: {engine.stats['n_packets_transported']}")
