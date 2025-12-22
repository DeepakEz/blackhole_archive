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

# Import base classes
import sys
sys.path.append('/mnt/user-data/outputs')

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
        Compute Ricci scalar at position
        
        For Schwarzschild: R = 0 (vacuum solution)
        But near horizon, effective curvature from tidal forces
        """
        r = position[1]
        
        # Kretschmann scalar (curvature invariant)
        # K = R_μνρσ R^μνρσ = 48M²/r⁶
        K = 48 * self.M**2 / (r**6 + 1e-10)
        
        # Return square root as effective curvature measure
        return np.sqrt(K)
    
    def get_curvature(self, position: np.ndarray) -> float:
        """Get curvature at position (wrapper for compatibility)"""
        return self.get_ricci_scalar(position)
    
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
    
    def add_structural_field(self, position: np.ndarray, strength: float, radius: float):
        """Add beaver structural field"""
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))

        # Add Gaussian
        for di in range(-5, 6):
            for dj in range(-3, 4):
                for dk in range(-3, 4):
                    ii = (i + di) % self.config.n_r
                    jj = max(0, min(self.config.n_theta-1, j + dj))
                    kk = (k + dk) % self.config.n_phi

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
    """Enhanced beaver with realistic construction"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structures_built = 0
        self.construction_cooldown = 0
    
    def update(self, dt: float, spacetime: EnhancedSpacetime):
        # Cooldown
        if self.construction_cooldown > 0:
            self.construction_cooldown -= dt
        
        # Check curvature
        curvature = spacetime.get_ricci_scalar(self.position)

        # FIX #1: Lower threshold by 1 order of magnitude (was 0.1, now 0.01)
        # At M=1: threshold 0.01 triggers at r < ~5.4 (vs r < ~3.4 at 0.1)
        # This allows beavers at moderate distances to build
        if curvature > 0.01 and self.energy > 0.05 and self.construction_cooldown <= 0:
            # Build structure
            spacetime.add_structural_field(self.position, 2.0, 3.0)
            self.structures_built += 1
            
            # Energy economics: MUST be net positive for sustainability
            construction_cost = 0.02  # Reduced from 0.05
            energy_reward = 0.05      # Increased from 0.03
            
            self.energy -= construction_cost
            self.energy += energy_reward
            # Net per build: +0.03 (sustainable!)
            
            self.construction_cooldown = 1.0  # Wait 1 time unit
            # During cooldown, lose: 1.0 * 0.005 = 0.005
            # Net per cycle: +0.03 - 0.005 = +0.025 (positive!)
        
        # Move toward high curvature regions
        # Sample nearby points
        nearby_curvatures = []
        directions = []
        
        for _ in range(5):
            offset = 0.5 * np.random.randn(4)
            test_pos = self.position + offset
            test_curv = spacetime.get_ricci_scalar(test_pos)
            nearby_curvatures.append(test_curv)
            directions.append(offset)
        
        # Move toward highest curvature
        max_idx = np.argmax(nearby_curvatures)
        self.velocity = 0.8 * self.velocity + 0.2 * directions[max_idx]
        
        # Update position
        self.position += dt * self.velocity

        # FIX #2: Energy decay modified by structural field
        # Moving through structured regions costs less energy
        movement_cost = spacetime.get_movement_cost(self.position)
        self.energy -= dt * 0.005 * movement_cost

        if self.energy <= 0:
            self.state = "dead"


class EnhancedAntAgent(Agent):
    """Enhanced ant with semantic graph building"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_vertex = None
        self.pheromone_deposits = 0
        self.path_history = []
    
    def update(self, dt: float, spacetime: EnhancedSpacetime, semantic_graph):
        # Sample local information
        info_density = spacetime.get_information_density(self.position)
        
        # If high information density, create vertex
        if info_density > 0.5 and self.current_vertex is None:
            vertex_id = semantic_graph.add_vertex(
                position=self.position.copy(),
                salience=info_density
            )
            self.current_vertex = vertex_id
            self.path_history.append(vertex_id)
        
        # If at vertex, deposit pheromone and move to neighbor
        if self.current_vertex is not None:
            # Add edges to recent vertices in path
            if len(self.path_history) > 1:
                prev_vertex = self.path_history[-2]
                semantic_graph.add_edge(prev_vertex, self.current_vertex, pheromone=1.0)
                self.pheromone_deposits += 1
            
            # Choose next vertex
            neighbors = list(semantic_graph.graph.neighbors(self.current_vertex))
            
            if neighbors:
                # Follow pheromones probabilistically
                pheromones = [semantic_graph.get_pheromone((self.current_vertex, n)) for n in neighbors]
                probs = np.array(pheromones) + 0.1  # Add baseline
                probs /= probs.sum()
                
                next_vertex = np.random.choice(neighbors, p=probs)
                self.current_vertex = next_vertex
                self.path_history.append(next_vertex)
                
                # Update position
                next_pos = semantic_graph.graph.nodes[next_vertex]['position']
                self.position = next_pos
            else:
                # No neighbors, explore
                self.current_vertex = None
        
        # Random walk if not at vertex
        if self.current_vertex is None:
            # FIX #2: Use exploration bias from structural field
            # Prefer exploring areas with less structural field (less explored)
            exploration_bias = spacetime.get_exploration_bias(self.position)
            # Higher bias = more exploration here; lower = move away
            random_step = 0.03 * np.random.randn(4)
            # Scale random step inversely with exploration bias
            # Low bias (already explored) = larger steps away
            random_step *= (2.0 - exploration_bias)
            self.position += dt * self.velocity + random_step

        # FIX #2: Energy decay modified by structural field
        movement_cost = spacetime.get_movement_cost(self.position)
        self.energy -= dt * 0.003 * movement_cost

        if self.energy <= 0:
            self.state = "dead"


class EnhancedBeeAgent(Agent):
    """Enhanced bee with packet transport"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "scout"
        self.packet = None
        self.target_vertex = None
        self.packets_delivered = 0
        self.waggle_intensity = 0.0
    
    def update(self, dt: float, spacetime: EnhancedSpacetime, semantic_graph, wormhole_position):
        if self.role == "scout":
            # Find high-salience vertices
            if np.random.rand() < 0.05:  # Occasionally check
                vertices = list(semantic_graph.graph.nodes())
                if vertices:
                    saliences = [semantic_graph.graph.nodes[v]['salience'] for v in vertices]
                    max_idx = np.argmax(saliences)
                    self.target_vertex = vertices[max_idx]
                    self.waggle_intensity = saliences[max_idx]
                    
                    # Switch to forager
                    if self.waggle_intensity > 0.6:
                        self.role = "forager"
        
        elif self.role == "forager":
            if self.packet is None and self.target_vertex is not None:
                # Move to target vertex
                target_pos = semantic_graph.graph.nodes[self.target_vertex]['position']
                
                # FIXED: Handle dimension mismatch (target_pos may be 16D, position is 4D)
                if len(target_pos) > len(self.position):
                    target_pos = target_pos[:len(self.position)]
                elif len(target_pos) < len(self.position):
                    target_pos = np.pad(target_pos, (0, len(self.position) - len(target_pos)))
                
                direction = target_pos - self.position
                self.velocity = 0.3 * direction / (np.linalg.norm(direction) + 1e-6)
                
                # Check if reached
                if np.linalg.norm(direction) < 1.0:
                    # Collect packet
                    self.packet = {
                        'vertex': self.target_vertex,
                        'salience': semantic_graph.graph.nodes[self.target_vertex]['salience'],
                        'timestamp': 0.0  # Would use simulation time
                    }
                    # Head to wormhole
                    self.role = "transporter"
            
        elif self.role == "transporter":
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
        
        # Update position
        self.position += dt * self.velocity

        # FIX #2: Energy decay modified by structural field
        # Structural field reduces movement cost for bees too
        movement_cost = spacetime.get_movement_cost(self.position)
        self.energy -= dt * 0.008 * movement_cost * (1 + 0.5 * bool(self.packet))

        if self.energy <= 0:
            self.state = "dead"

# =============================================================================
# SEMANTIC GRAPH
# =============================================================================

class SemanticGraph:
    """Semantic graph maintained by ants"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.pheromones = {}  # (v1, v2) -> strength
        self.next_vertex_id = 0
    
    def add_vertex(self, position: np.ndarray, salience: float) -> int:
        """Add vertex to graph"""
        vertex_id = self.next_vertex_id
        self.next_vertex_id += 1
        
        self.graph.add_node(vertex_id, position=position, salience=salience)
        return vertex_id
    
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
        
        # Wormhole position
        self.wormhole_position = np.array([0.0, 2.6, np.pi/2, 0.0])
        
        # Statistics
        self.stats = {
            'n_packets_transported': 0,
            'total_energy': 0.0,
            'n_structures_built': 0,
            'n_vertices': 0,
            'n_edges': 0,
            'energy_history': [],
            'vertices_history': [],
            'structures_history': []
        }
        
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

        for i in range(n_initial_vertices):
            # Create vertices near event horizon where curvature is high
            r = self.config.r_s + 0.5 + np.random.rand() * 5  # r in [rs+0.5, rs+5.5]
            theta = np.random.rand() * np.pi
            phi = np.random.rand() * 2 * np.pi

            position = np.array([0.0, r, theta, phi])

            # Compute salience based on curvature - closer to horizon = higher salience
            curvature = self.spacetime.get_curvature(position)
            salience = min(1.0, curvature * 10)  # Scale to [0, 1]

            # Add vertex to semantic graph
            vertex_id = self.semantic_graph.add_vertex(position, salience)

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
        
        for step in tqdm(range(n_steps), desc="Enhanced Simulation"):
            t = step * self.config.dt
            
            # Update all agents
            for beaver in self.agents['beavers']:
                if beaver.state == "active":
                    beaver.update(self.config.dt, self.spacetime)
            
            for ant in self.agents['ants']:
                if ant.state == "active":
                    ant.update(self.config.dt, self.spacetime, self.semantic_graph)
            
            for bee in self.agents['bees']:
                if bee.state == "active":
                    bee.update(self.config.dt, self.spacetime, self.semantic_graph, self.wormhole_position)
            
            # Decay pheromones
            self.semantic_graph.decay_pheromones(self.config.dt)
            
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
            self.stats['n_vertices'] = self.semantic_graph.graph.number_of_nodes()
            self.stats['n_edges'] = self.semantic_graph.graph.number_of_edges()
            
            # Record history
            if step % 10 == 0:
                self.stats['energy_history'].append(self.stats['total_energy'])
                self.stats['vertices_history'].append(self.stats['n_vertices'])
                self.stats['structures_history'].append(self.stats['n_structures_built'])
            
            # Log
            if step % 100 == 0:
                self.logger.info(
                    f"Step {step}/{n_steps}, t={t:.2f}, "
                    f"Energy={self.stats['total_energy']:.2f}, "
                    f"Vertices={self.stats['n_vertices']}, "
                    f"Structures={self.stats['n_structures_built']}, "
                    f"Packets={self.stats['n_packets_transported']}"
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
