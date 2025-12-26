# Blackhole Archive Project: Simulation Architecture
# Complete Implementation Framework

"""
SYSTEM ARCHITECTURE OVERVIEW

The simulation consists of 5 major subsystems:

1. Physics Engine: Handles spacetime geometry, geodesics, wormhole dynamics
2. Colony Agents: Implements beaver/ant/bee behaviors
3. Communication Protocol: Inter-colony coordination and packet transport
4. Visualization: Real-time 3D rendering of system state
5. Analysis Tools: Data extraction, metrics, and validation

Technology Stack:
- Python 3.11+
- NumPy/SciPy: Numerical computations
- PyTorch: GPU-accelerated field dynamics
- NetworkX: Graph structures for ants
- FastAPI: Inter-process communication
- Plotly/Mayavi: 3D visualization
- HDF5: Data persistence
"""

# =============================================================================
# CORE DEPENDENCIES
# =============================================================================

"""
requirements.txt:

numpy>=1.24.0
scipy>=1.11.0
torch>=2.1.0
networkx>=3.2
fastapi>=0.104.0
uvicorn>=0.24.0
plotly>=5.18.0
mayavi>=4.8.1
h5py>=3.10.0
pandas>=2.1.0
pydantic>=2.5.0
websockets>=12.0
numba>=0.58.0
jax>=0.4.20
jaxlib>=0.4.20
"""

# =============================================================================
# 1. PHYSICS ENGINE
# =============================================================================

"""
blackhole_archive/
├── physics/
│   ├── __init__.py
│   ├── spacetime.py          # Metric, geodesics
│   ├── wormhole.py           # Wormhole throat dynamics
│   ├── fields.py             # Scalar/vector field evolution
│   ├── hawking.py            # Hawking radiation
│   └── integrators.py        # Numerical integration schemes
├── agents/
│   ├── __init__.py
│   ├── base.py               # Base agent class
│   ├── beavers.py            # Beaver colony
│   ├── ants.py               # Ant colony
│   └── bees.py               # Bee colony
├── protocols/
│   ├── __init__.py
│   ├── packets.py            # Packet definitions
│   ├── channels.py           # Communication channels
│   └── synchronization.py    # Clock synchronization
├── visualization/
│   ├── __init__.py
│   ├── spacetime_viz.py      # Metric visualization
│   ├── colony_viz.py         # Agent visualization
│   └── dashboard.py          # Real-time dashboard
├── analysis/
│   ├── __init__.py
│   ├── metrics.py            # System metrics
│   ├── information.py        # Information-theoretic measures
│   └── validation.py         # Theoretical validation
└── main.py                   # Main simulation runner
"""

# =============================================================================
# physics/spacetime.py
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import torch
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from numba import jit

@dataclass
class SpacetimeConfig:
    """Configuration for spacetime simulation"""
    # Black hole parameters
    mass: float  # Solar masses
    spin: float  # Dimensionless spin parameter (0 to 1)
    
    # Grid parameters
    r_min: float  # Minimum radius (in Schwarzschild radii)
    r_max: float  # Maximum radius
    n_r: int      # Number of radial points
    n_theta: int  # Number of theta points
    n_phi: int    # Number of phi points
    
    # Time evolution
    t_max: float  # Maximum simulation time
    dt: float     # Time step
    
    # Wormhole parameters
    throat_radius: float  # Throat radius b_0 (in Schwarzschild radii)
    throat_length: float  # Throat length
    redshift_scale: float # Redshift function scale
    
    # Physical constants (in geometric units G=c=1)
    hbar: float = 1.0
    k_B: float = 1.0

class SchwarzschildMetric:
    """
    Schwarzschild metric in Schwarzschild coordinates (t, r, theta, phi)
    
    ds^2 = -(1 - 2M/r)dt^2 + (1 - 2M/r)^(-1)dr^2 + r^2(dtheta^2 + sin^2(theta)dphi^2)
    """
    
    def __init__(self, mass: float):
        self.M = mass
        self.r_s = 2 * mass  # Schwarzschild radius
        
    def metric_tensor(self, r: float, theta: float) -> np.ndarray:
        """
        Compute metric tensor g_μν at point (r, theta)
        
        Returns:
            4x4 metric tensor
        """
        g = np.zeros((4, 4))
        
        f = 1 - self.r_s / r
        
        g[0, 0] = -f  # g_tt
        g[1, 1] = 1 / f  # g_rr
        g[2, 2] = r**2  # g_θθ
        g[3, 3] = r**2 * np.sin(theta)**2  # g_φφ
        
        return g
    
    def inverse_metric(self, r: float, theta: float) -> np.ndarray:
        """Compute inverse metric g^μν"""
        g_inv = np.zeros((4, 4))
        
        f = 1 - self.r_s / r
        
        g_inv[0, 0] = -1 / f
        g_inv[1, 1] = f
        g_inv[2, 2] = 1 / r**2
        g_inv[3, 3] = 1 / (r**2 * np.sin(theta)**2)
        
        return g_inv
    
    def christoffel_symbols(self, r: float, theta: float) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^μ_νρ
        
        Returns:
            4x4x4 array of connection coefficients
        """
        Gamma = np.zeros((4, 4, 4))
        
        M = self.M
        f = 1 - 2*M/r
        df_dr = 2*M / r**2
        
        # Non-zero components (using symmetry Γ^μ_νρ = Γ^μ_ρν)
        
        # Γ^t_tr = Γ^t_rt
        Gamma[0, 0, 1] = Gamma[0, 1, 0] = M / (r**2 * f)
        
        # Γ^r_tt
        Gamma[1, 0, 0] = M * f / r**2
        
        # Γ^r_rr
        Gamma[1, 1, 1] = -M / (r**2 * f)
        
        # Γ^r_θθ
        Gamma[1, 2, 2] = -(r - 2*M)
        
        # Γ^r_φφ
        Gamma[1, 3, 3] = -(r - 2*M) * np.sin(theta)**2
        
        # Γ^θ_rθ = Γ^θ_θr
        Gamma[2, 1, 2] = Gamma[2, 2, 1] = 1 / r
        
        # Γ^θ_φφ
        Gamma[2, 3, 3] = -np.sin(theta) * np.cos(theta)
        
        # Γ^φ_rφ = Γ^φ_φr
        Gamma[3, 1, 3] = Gamma[3, 3, 1] = 1 / r
        
        # Γ^φ_θφ = Γ^φ_φθ
        Gamma[3, 2, 3] = Gamma[3, 3, 2] = np.cos(theta) / np.sin(theta)
        
        return Gamma

class MorrisThrorneWormhole:
    """
    Morris-Thorne traversable wormhole metric
    
    ds^2 = -e^(2Φ(l))dt^2 + dl^2 + r^2(l)(dθ^2 + sin^2(θ)dφ^2)
    
    where r(l) = sqrt(b_0^2 + l^2) defines the throat shape
    """
    
    def __init__(self, b_0: float, length: float, redshift_scale: float = 1.0):
        self.b_0 = b_0  # Throat radius
        self.L = length  # Throat length
        self.phi_0 = redshift_scale
        
    def shape_function(self, l: float) -> float:
        """Radial coordinate as function of proper distance l"""
        return np.sqrt(self.b_0**2 + l**2)
    
    def redshift_function(self, l: float) -> float:
        """Redshift function Φ(l)"""
        # Smooth function that goes to 0 at infinity
        return self.phi_0 * np.exp(-np.abs(l) / self.L)
    
    def metric_tensor(self, l: float, theta: float) -> np.ndarray:
        """Metric tensor in wormhole coordinates"""
        g = np.zeros((4, 4))
        
        Phi = self.redshift_function(l)
        r = self.shape_function(l)
        
        g[0, 0] = -np.exp(2 * Phi)  # g_tt
        g[1, 1] = 1.0  # g_ll
        g[2, 2] = r**2  # g_θθ
        g[3, 3] = r**2 * np.sin(theta)**2  # g_φφ
        
        return g
    
    def christoffel_symbols(self, l: float, theta: float) -> np.ndarray:
        """Christoffel symbols for wormhole metric"""
        Gamma = np.zeros((4, 4, 4))
        
        # Derivatives
        Phi = self.redshift_function(l)
        dPhi_dl = self._derivative(self.redshift_function, l)
        
        r = self.shape_function(l)
        dr_dl = l / np.sqrt(self.b_0**2 + l**2)
        
        # Non-zero components
        # Γ^t_tl
        Gamma[0, 0, 1] = Gamma[0, 1, 0] = dPhi_dl
        
        # Γ^l_tt
        Gamma[1, 0, 0] = dPhi_dl * np.exp(2 * Phi)
        
        # Γ^l_θθ
        Gamma[1, 2, 2] = -r * dr_dl
        
        # Γ^l_φφ
        Gamma[1, 3, 3] = -r * dr_dl * np.sin(theta)**2
        
        # Γ^θ_lθ
        Gamma[2, 1, 2] = Gamma[2, 2, 1] = dr_dl / r
        
        # Γ^θ_φφ
        Gamma[2, 3, 3] = -np.sin(theta) * np.cos(theta)
        
        # Γ^φ_lφ
        Gamma[3, 1, 3] = Gamma[3, 3, 1] = dr_dl / r
        
        # Γ^φ_θφ
        Gamma[3, 2, 3] = Gamma[3, 3, 2] = np.cos(theta) / np.sin(theta)
        
        return Gamma
    
    @staticmethod
    def _derivative(func: Callable, x: float, h: float = 1e-5) -> float:
        """Numerical derivative using central difference"""
        return (func(x + h) - func(x - h)) / (2 * h)

class GeodesicIntegrator:
    """
    Integrate geodesic equations in curved spacetime
    
    d²x^μ/dλ² + Γ^μ_αβ (dx^α/dλ)(dx^β/dλ) = 0
    """
    
    def __init__(self, metric):
        self.metric = metric
        
    def geodesic_equation(self, lambda_param: float, y: np.ndarray) -> np.ndarray:
        """
        Geodesic equation in first-order form
        
        y = [x^0, x^1, x^2, x^3, dx^0/dλ, dx^1/dλ, dx^2/dλ, dx^3/dλ]
        """
        # Extract position and velocity
        x = y[:4]  # (t, r, theta, phi)
        v = y[4:]  # (dt/dλ, dr/dλ, dθ/dλ, dφ/dλ)
        
        # Compute Christoffel symbols at current position
        # (for Schwarzschild, only depends on r and theta)
        Gamma = self.metric.christoffel_symbols(x[1], x[2])
        
        # Acceleration: d²x^μ/dλ² = -Γ^μ_αβ v^α v^β
        a = np.zeros(4)
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    a[mu] -= Gamma[mu, alpha, beta] * v[alpha] * v[beta]
        
        # Return derivative [v, a]
        return np.concatenate([v, a])
    
    def integrate(self, 
                  x_initial: np.ndarray, 
                  v_initial: np.ndarray,
                  lambda_span: Tuple[float, float],
                  method: str = 'DOP853') -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate geodesic from initial conditions
        
        Args:
            x_initial: Initial position (t, r, θ, φ)
            v_initial: Initial 4-velocity
            lambda_span: (lambda_start, lambda_end)
            method: Integration method
            
        Returns:
            lambda_values, trajectory (8 x N array)
        """
        y0 = np.concatenate([x_initial, v_initial])
        
        sol = solve_ivp(
            self.geodesic_equation,
            lambda_span,
            y0,
            method=method,
            dense_output=True,
            rtol=1e-10,
            atol=1e-12
        )
        
        return sol.t, sol.y

class SpacetimeGrid:
    """
    3D computational grid for field evolution on curved spacetime
    """
    
    def __init__(self, config: SpacetimeConfig, metric):
        self.config = config
        self.metric = metric
        
        # Create grid
        self.r = np.linspace(config.r_min, config.r_max, config.n_r)
        self.theta = np.linspace(0, np.pi, config.n_theta)
        self.phi = np.linspace(0, 2*np.pi, config.n_phi)
        
        # Mesh grid
        self.R, self.THETA, self.PHI = np.meshgrid(self.r, self.theta, self.phi, indexing='ij')
        
        # Metric tensor at each grid point (precomputed for efficiency)
        self._precompute_metric()
        
    def _precompute_metric(self):
        """Precompute metric tensor at all grid points"""
        shape = (self.config.n_r, self.config.n_theta, self.config.n_phi, 4, 4)
        self.g = np.zeros(shape)
        self.g_inv = np.zeros(shape)
        self.Gamma = np.zeros((self.config.n_r, self.config.n_theta, self.config.n_phi, 4, 4, 4))
        
        for i in range(self.config.n_r):
            for j in range(self.config.n_theta):
                r = self.r[i]
                theta = self.theta[j]
                
                self.g[i, j, :] = self.metric.metric_tensor(r, theta)
                self.g_inv[i, j, :] = self.metric.inverse_metric(r, theta)
                self.Gamma[i, j, :] = self.metric.christoffel_symbols(r, theta)
    
    def sqrt_det_g(self, i: int, j: int, k: int) -> float:
        """sqrt(-det(g)) at grid point (i, j, k)"""
        r = self.r[i]
        theta = self.theta[j]
        return r**2 * np.sin(theta) * np.sqrt(1 - 2*self.metric.M/r)

class FieldEvolver:
    """
    Evolve scalar and vector fields on curved spacetime
    
    Uses finite difference methods with covariant derivatives
    """
    
    def __init__(self, grid: SpacetimeGrid):
        self.grid = grid
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evolve_scalar_field(self, 
                           phi: np.ndarray,
                           V: Callable,
                           t: float,
                           dt: float) -> np.ndarray:
        """
        Evolve scalar field equation:
        □φ - V'(φ) = 0
        
        where □ is the d'Alembertian operator
        """
        # Convert to torch tensor
        phi_t = torch.tensor(phi, device=self.device, dtype=torch.float32)
        
        # Compute Laplacian with covariant derivatives
        laplacian = self._covariant_laplacian(phi_t)
        
        # Potential derivative
        dV = torch.tensor(
            np.gradient(V(phi.flatten())).reshape(phi.shape),
            device=self.device
        )
        
        # Time evolution: φ_new = φ_old + dt * (□φ - V'(φ))
        phi_new = phi_t + dt * (laplacian - dV)
        
        return phi_new.cpu().numpy()
    
    def _covariant_laplacian(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Covariant Laplacian: □φ = g^μν ∇_μ∇_ν φ
        """
        # Finite difference approximation
        # This is simplified; full implementation requires careful treatment
        # of connection coefficients
        
        # Get grid spacing
        dr = self.grid.r[1] - self.grid.r[0]
        dtheta = self.grid.theta[1] - self.grid.theta[0]
        dphi = self.grid.phi[1] - self.grid.phi[0]
        
        # Partial derivatives (central difference)
        dphi_dr = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / (2 * dr)
        dphi_dtheta = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / (2 * dtheta)
        dphi_dphi = (torch.roll(phi, -1, 2) - torch.roll(phi, 1, 2)) / (2 * dphi)
        
        # Second derivatives
        d2phi_dr2 = (torch.roll(phi, -1, 0) - 2*phi + torch.roll(phi, 1, 0)) / dr**2
        d2phi_dtheta2 = (torch.roll(phi, -1, 1) - 2*phi + torch.roll(phi, 1, 1)) / dtheta**2
        d2phi_dphi2 = (torch.roll(phi, -1, 2) - 2*phi + torch.roll(phi, 2)) / dphi**2
        
        # Combine with inverse metric (diagonal in Schwarzschild)
        # Full expression includes connection terms
        g_inv = torch.tensor(self.grid.g_inv, device=self.device)
        
        laplacian = (
            g_inv[..., 1, 1] * d2phi_dr2 +
            g_inv[..., 2, 2] * d2phi_dtheta2 +
            g_inv[..., 3, 3] * d2phi_dphi2
        )
        
        return laplacian

# =============================================================================
# physics/hawking.py
# =============================================================================

class HawkingRadiation:
    """
    Hawking radiation emission from black hole
    
    Temperature: T_H = ℏc³ / (8πGMk_B)
    Luminosity: L = ℏc⁶ / (15360πG²M²)
    """
    
    def __init__(self, mass: float, hbar: float = 1.0, k_B: float = 1.0):
        self.M = mass
        self.hbar = hbar
        self.k_B = k_B
        
        # Hawking temperature (geometric units)
        self.T_H = hbar / (8 * np.pi * mass * k_B)
        
        # Luminosity
        self.L = hbar / (15360 * np.pi * mass**2)
        
    def thermal_spectrum(self, omega: np.ndarray) -> np.ndarray:
        """
        Planck distribution for Hawking radiation
        
        n(ω) = 1 / (exp(ℏω/k_B T_H) - 1)
        """
        return 1 / (np.exp(self.hbar * omega / (self.k_B * self.T_H)) - 1)
    
    def sample_photon(self, rng: np.random.Generator) -> Tuple[float, np.ndarray]:
        """
        Sample a Hawking photon (frequency and direction)
        
        Returns:
            omega: Photon frequency
            direction: Unit 3-vector (theta, phi)
        """
        # Sample frequency from thermal distribution
        # Use rejection sampling
        omega_max = 10 * self.k_B * self.T_H / self.hbar
        omega = self._rejection_sample_planck(rng, omega_max)
        
        # Isotropic emission
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        
        direction = np.array([theta, phi])
        
        return omega, direction
    
    def _rejection_sample_planck(self, rng: np.random.Generator, omega_max: float) -> float:
        """Rejection sampling for Planck distribution"""
        # Maximum of Planck function (approximate)
        max_val = 2 * (self.k_B * self.T_H / self.hbar)**3
        
        while True:
            omega = rng.random() * omega_max
            u = rng.random() * max_val
            
            if u < omega**2 / (np.exp(self.hbar * omega / (self.k_B * self.T_H)) - 1):
                return omega
    
    def mass_loss_rate(self) -> float:
        """Rate of mass loss due to Hawking radiation"""
        return -self.L  # dM/dt = -L (in geometric units)

# =============================================================================
# 2. COLONY AGENTS
# =============================================================================

# agents/base.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any
import uuid

class AgentState(Enum):
    """Base agent states"""
    IDLE = "idle"
    ACTIVE = "active"
    COMMUNICATING = "communicating"
    DEAD = "dead"

@dataclass
class Agent:
    """Base class for all agents"""
    id: str
    position: np.ndarray  # Spacetime position (t, r, θ, φ)
    velocity: np.ndarray  # 4-velocity
    state: AgentState
    energy: float
    colony_id: str
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, position: np.ndarray, colony_id: str, **kwargs):
        """Factory method to create agent"""
        return cls(
            id=str(uuid.uuid4()),
            position=position,
            velocity=np.zeros(4),
            state=AgentState.IDLE,
            energy=1.0,
            colony_id=colony_id,
            metadata={},
            **kwargs
        )
    
    @abstractmethod
    def update(self, dt: float, environment: 'Environment'):
        """Update agent state"""
        pass
    
    @abstractmethod
    def communicate(self, other_agent: 'Agent') -> Optional['Message']:
        """Generate message to another agent"""
        pass

# agents/beavers.py

from dataclasses import dataclass, field
from typing import Optional
import networkx as nx

@dataclass
class BeaverAgent(Agent):
    """
    Beaver agent: Constructs and maintains spacetime structures
    
    Capabilities:
    - Build scaffolds at specified spacetime locations
    - Maintain wormhole throat stability
    - Regulate information flow through topology modification
    """
    
    # Beaver-specific attributes
    structural_field: float = 1.0  # σ_B value at agent location
    construction_sites: List[np.ndarray] = field(default_factory=list)
    dam_strength: float = 1.0
    local_curvature: float = 0.0  # Cached curvature for stability computation

    def update(self, dt: float, environment: 'Environment'):
        """Update beaver behavior"""
        # Check local curvature and cache it for stability computation
        curvature = environment.get_curvature(self.position)
        self.local_curvature = curvature
        
        # If curvature is too high (unstable), build structure
        if curvature > environment.config.stability_threshold:
            self.build_scaffold(environment)
        
        # Maintain existing structures
        for site in self.construction_sites:
            self.reinforce_structure(site, environment)
        
        # Move along curvature gradient (beavers seek unstable regions)
        grad_curvature = environment.get_curvature_gradient(self.position)
        self.velocity[1:] += dt * grad_curvature * 0.1  # Small adjustment
        
        # Update position (geodesic motion)
        self.position += dt * self.velocity
        
        # Decay energy
        self.energy -= dt * 0.01
        
        if self.energy <= 0:
            self.state = AgentState.DEAD
    
    def build_scaffold(self, environment: 'Environment'):
        """Construct a structural scaffold at current location"""
        # Add to construction sites
        self.construction_sites.append(self.position.copy())
        
        # Modify structural field in environment
        environment.add_structural_field(
            self.position,
            strength=self.dam_strength,
            radius=2.0
        )
        
        # Energy cost
        self.energy -= 0.1
    
    def reinforce_structure(self, site: np.ndarray, environment: 'Environment'):
        """Maintain existing structure"""
        # Check if structure needs reinforcement
        field_strength = environment.get_structural_field(site)
        
        if field_strength < 0.5 * self.dam_strength:
            environment.add_structural_field(site, 0.1 * self.dam_strength, 2.0)
            self.energy -= 0.05
    
    def communicate(self, other_agent: Agent) -> Optional['Message']:
        """Beavers broadcast structural stability information"""
        if isinstance(other_agent, (AntAgent, BeeAgent)):
            return Message(
                sender_id=self.id,
                receiver_id=other_agent.id,
                message_type="structural_update",
                content={
                    "position": self.position,
                    "field_strength": self.structural_field,
                    "stability": self._compute_local_stability()
                }
            )
        return None
    
    def _compute_local_stability(self) -> float:
        """
        Compute stability metric at current location.

        Stability depends on:
        - Structural field strength relative to dam strength
        - Local spacetime curvature (high curvature = less stable)

        Returns value in [0, 1] where 1 = perfectly stable.
        """
        # Base stability from structural field
        field_stability = min(1.0, self.structural_field / self.dam_strength)

        # Curvature penalty: high curvature reduces stability
        # Use sigmoid to smoothly reduce stability as curvature increases
        curvature_penalty = 1.0 / (1.0 + self.local_curvature)

        # Combined stability metric
        stability = field_stability * curvature_penalty

        return max(0.0, min(1.0, stability))

# agents/ants.py

@dataclass
class AntAgent(Agent):
    """
    Ant agent: Builds and maintains semantic graph structures
    
    Capabilities:
    - Explore information space
    - Deposit pheromones on semantic edges
    - Construct consensus graph from distributed observations
    """
    
    # Ant-specific attributes
    pheromone_strength: float = 1.0
    current_vertex: Optional[int] = None
    graph_view: Optional[nx.DiGraph] = None
    memory: List[int] = field(default_factory=list)  # Path history
    
    def __post_init__(self):
        if self.graph_view is None:
            self.graph_view = nx.DiGraph()
    
    def update(self, dt: float, environment: 'Environment'):
        """Update ant behavior"""
        # If not at a vertex, search for information
        if self.current_vertex is None:
            self._explore(environment)
        else:
            # Follow pheromone trails to high-salience vertices
            self._follow_pheromones(environment)
        
        # Deposit pheromones on traversed edges
        if len(self.memory) >= 2:
            edge = (self.memory[-2], self.memory[-1])
            self._deposit_pheromone(edge, environment)
        
        # Update position
        self.position += dt * self.velocity
        
        # Energy decay
        self.energy -= dt * 0.005
        
        if self.energy <= 0:
            self.state = AgentState.DEAD
    
    def _explore(self, environment: 'Environment'):
        """Explore for new semantic nodes"""
        # Sample local information field
        info = environment.sample_information(self.position)
        
        # If significant information found, create vertex
        if info['significance'] > 0.5:
            vertex_id = environment.semantic_graph.add_vertex(
                position=self.position,
                content=info['content']
            )
            self.current_vertex = vertex_id
            self.memory.append(vertex_id)
    
    def _follow_pheromones(self, environment: 'Environment'):
        """Follow pheromone gradient to next vertex"""
        # Get neighbors of current vertex
        neighbors = list(environment.semantic_graph.graph.neighbors(self.current_vertex))
        
        if not neighbors:
            # No neighbors, explore
            self.current_vertex = None
            return
        
        # Choose neighbor probabilistically based on pheromone strength
        pheromones = [
            environment.semantic_graph.get_pheromone((self.current_vertex, n))
            for n in neighbors
        ]
        
        # Softmax probabilities
        pheromones = np.array(pheromones)
        probs = np.exp(pheromones) / np.sum(np.exp(pheromones))
        
        # Sample next vertex
        next_vertex = np.random.choice(neighbors, p=probs)
        
        # Move to next vertex
        self.current_vertex = next_vertex
        self.memory.append(next_vertex)
        
        # Update position to vertex location
        vertex_data = environment.semantic_graph.graph.nodes[next_vertex]
        self.position = vertex_data['position']
    
    def _deposit_pheromone(self, edge: Tuple[int, int], environment: 'Environment'):
        """Deposit pheromone on edge"""
        environment.semantic_graph.add_pheromone(edge, self.pheromone_strength * dt)
        self.energy -= 0.01
    
    def communicate(self, other_agent: Agent) -> Optional['Message']:
        """Ants share graph structure information"""
        if isinstance(other_agent, AntAgent):
            # Share local graph view
            return Message(
                sender_id=self.id,
                receiver_id=other_agent.id,
                message_type="graph_update",
                content={
                    "vertices": list(self.graph_view.nodes()),
                    "edges": list(self.graph_view.edges()),
                    "current_vertex": self.current_vertex
                }
            )
        elif isinstance(other_agent, BeeAgent):
            # Provide routing information
            if self.current_vertex is not None:
                return Message(
                    sender_id=self.id,
                    receiver_id=other_agent.id,
                    message_type="routing_info",
                    content={
                        "vertex": self.current_vertex,
                        "salience": environment.semantic_graph.get_salience(self.current_vertex)
                    }
                )
        return None

# agents/bees.py

@dataclass
class BeeAgent(Agent):
    """
    Bee agent: Transport packets through wormhole
    
    Capabilities:
    - Scout for wormhole routes
    - Transport data packets
    - Perform waggle dance to recruit other bees
    - Maintain temporal coherence across time dilation
    """
    
    # Bee-specific attributes
    role: str = "forager"  # scout, forager, guard
    packet: Optional['Packet'] = None
    waggle_intensity: float = 0.0
    target_vertex: Optional[int] = None
    clock: float = 0.0  # Proper time
    
    def update(self, dt: float, environment: 'Environment'):
        """Update bee behavior"""
        # Update proper time (accounting for time dilation)
        time_dilation = environment.get_time_dilation(self.position)
        self.clock += dt / time_dilation
        
        if self.role == "scout":
            self._scout_behavior(dt, environment)
        elif self.role == "forager":
            self._forager_behavior(dt, environment)
        elif self.role == "guard":
            self._guard_behavior(dt, environment)
        
        # Update position
        self.position += dt * self.velocity
        
        # Energy decay
        self.energy -= dt * 0.01 * (1 + bool(self.packet))  # Higher cost when carrying packet
        
        if self.energy <= 0:
            self.state = AgentState.DEAD
    
    def _scout_behavior(self, dt: float, environment: 'Environment'):
        """Scout explores for high-value information"""
        # Query ants for high-salience vertices
        nearby_ants = environment.get_nearby_agents(self.position, radius=5.0, agent_type=AntAgent)
        
        if nearby_ants:
            # Get routing information
            salience_map = {}
            for ant in nearby_ants:
                if ant.current_vertex is not None:
                    salience = environment.semantic_graph.get_salience(ant.current_vertex)
                    salience_map[ant.current_vertex] = salience
            
            # Choose highest salience vertex
            if salience_map:
                self.target_vertex = max(salience_map, key=salience_map.get)
                
                # Perform waggle dance to recruit foragers
                self.waggle_intensity = salience_map[self.target_vertex]
                
                # Switch to forager
                self.role = "forager"
    
    def _forager_behavior(self, dt: float, environment: 'Environment'):
        """Forager collects and transports packets"""
        if self.packet is None:
            # Try to collect packet at target vertex
            if self.target_vertex is not None:
                packet = environment.semantic_graph.extract_packet(self.target_vertex)
                if packet is not None:
                    self.packet = packet
                    # Head to wormhole
                    self._set_velocity_to_wormhole(environment)
        else:
            # Transport packet through wormhole
            if environment.is_at_wormhole(self.position):
                # Deliver packet
                environment.wormhole_queue.append(self.packet)
                self.packet = None
                self.target_vertex = None
                
                # Switch back to scout
                self.role = "scout"
    
    def _guard_behavior(self, dt: float, environment: 'Environment'):
        """Guard maintains wormhole entrance"""
        # Stay near wormhole mouth
        wormhole_position = environment.get_wormhole_position()
        direction = wormhole_position - self.position
        self.velocity[1:] = 0.1 * direction[1:] / (np.linalg.norm(direction[1:]) + 1e-6)
    
    def _set_velocity_to_wormhole(self, environment: 'Environment'):
        """Set velocity toward wormhole mouth"""
        wormhole_position = environment.get_wormhole_position()
        direction = wormhole_position - self.position
        self.velocity[1:] = 0.5 * direction[1:] / (np.linalg.norm(direction[1:]) + 1e-6)
    
    def communicate(self, other_agent: Agent, environment: 'Environment' = None) -> Optional['Message']:
        """Bees perform waggle dance to communicate"""
        if isinstance(other_agent, BeeAgent) and self.role == "scout" and self.waggle_intensity > 0:
            return Message(
                sender_id=self.id,
                receiver_id=other_agent.id,
                message_type="waggle_dance",
                content={
                    "target_vertex": self.target_vertex,
                    "intensity": self.waggle_intensity,
                    "direction": self._encode_direction(environment),
                    "distance": self._encode_distance(environment)
                }
            )
        return None
    
    def _encode_direction(self, environment: 'Environment' = None) -> float:
        """
        Encode direction to target as angle relative to local frame.

        Uses spherical coordinates (theta, phi) to compute direction angle.
        Returns angle in radians [0, 2π] relative to current heading.
        """
        if self.target_vertex is None or environment is None:
            return 0.0

        # Get target vertex position from semantic graph
        if not hasattr(environment, 'semantic_graph'):
            return 0.0

        target_pos = environment.semantic_graph.get_vertex_position(self.target_vertex)
        if target_pos is None:
            return 0.0

        # Compute direction vector in spatial coordinates (r, theta, phi)
        # Position format: [t, r, theta, phi]
        direction = target_pos[1:] - self.position[1:]

        # Compute angle in local tangent space
        # Use atan2 for proper quadrant handling
        if len(direction) >= 2:
            # Project onto theta-phi plane for direction encoding
            angle = np.arctan2(direction[2] if len(direction) > 2 else 0.0,
                              direction[1] if len(direction) > 1 else 1.0)
            # Normalize to [0, 2π]
            return (angle + 2 * np.pi) % (2 * np.pi)

        return 0.0

    def _encode_distance(self, environment: 'Environment' = None) -> float:
        """
        Encode distance to target as waggle duration.

        Computes proper distance accounting for curved spacetime.
        Returns duration in time units proportional to distance.
        """
        if self.target_vertex is None or environment is None:
            return 0.0

        # Get target vertex position from semantic graph
        if not hasattr(environment, 'semantic_graph'):
            return 0.0

        target_pos = environment.semantic_graph.get_vertex_position(self.target_vertex)
        if target_pos is None:
            return 0.0

        # Compute coordinate distance
        coord_distance = np.linalg.norm(target_pos[1:] - self.position[1:])

        # Compute proper distance using metric if available
        if hasattr(environment, 'metric') and environment.metric is not None:
            metric = environment.metric
            r = self.position[1] if len(self.position) > 1 else 1.0
            theta = self.position[2] if len(self.position) > 2 else np.pi / 2

            # Get metric tensor at current position
            g = metric.metric_tensor(max(r, metric.r_s * 1.01), theta)

            # Approximate proper distance using g_rr component
            # ds² = g_rr dr² for radial motion
            proper_distance = coord_distance * np.sqrt(abs(g[1, 1]))
        else:
            proper_distance = coord_distance

        # Waggle duration proportional to distance (scaling factor for dance)
        waggle_scale = 0.1  # Time units per distance unit
        return proper_distance * waggle_scale

# This is just the beginning - there's much more to implement!
# Next sections would include:
# - Environment class
# - SemanticGraph class
# - Packet definitions
# - Message passing system
# - Full simulation loop
# - Visualization
# - Analysis tools

# Let me know if you want me to continue with the remaining sections!
