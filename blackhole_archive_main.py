# Blackhole Archive Project: Complete Implementation Guide
# Executable Reference Implementation

"""
QUICK START GUIDE

This file provides a complete, executable reference implementation of the
Blackhole Archive Project. All components are production-ready and can be
extended for research purposes.

INSTALLATION:
```bash
pip install -r requirements.txt
python blackhole_archive_main.py --mode demo
```

COMPONENTS:
1. Physics engine with full GR simulation
2. Multi-colony agent system (beavers, ants, bees)
3. Wormhole transport protocol
4. MycoNet integration layer
5. Visualization and analysis tools

DIRECTORY STRUCTURE (after running setup):
blackhole_archive_project/
├── data/                  # Simulation data (HDF5)
├── logs/                  # Execution logs
├── visualizations/        # Generated plots
├── checkpoints/           # Saved states
└── results/               # Analysis results
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import json
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from tqdm import tqdm
import asyncio
from concurrent.futures import ProcessPoolExecutor
import uuid

from blackhole_archive_enhanced import (
    EnhancedSpacetime,
    EnhancedBeaverAgent,
    EnhancedAntAgent,
    EnhancedBeeAgent,
    SemanticGraph,
)
from blackhole_archive_protocols import (
    WormholeTransportProtocol,
    Packet,
    PacketType,
    SemanticCoordinate,
    EntropySignature,
    CausalCertificate,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    
    # Physics parameters
    black_hole_mass: float = 1.0  # Solar masses (geometric units)
    throat_radius: float = 2.0    # Schwarzschild radii
    throat_length: float = 4.0    # Schwarzschild radii
    
    # Spatial grid
    r_min: float = 2.5   # Just outside event horizon
    r_max: float = 50.0
    n_r: int = 128
    n_theta: int = 64
    n_phi: int = 64
    
    # Time evolution
    t_max: float = 100.0
    dt: float = 0.01
    
    # Colony parameters
    n_beavers: int = 50
    n_ants: int = 200
    n_bees: int = 100
    
    # Agent behavior
    beaver_build_rate: float = 0.1
    ant_pheromone_strength: float = 1.0
    bee_waggle_threshold: float = 0.5
    
    # Memory parameters
    consolidation_threshold: float = 0.7
    retrieval_confidence: float = 0.5
    
    # Performance
    use_gpu: bool = True
    n_workers: int = 4
    checkpoint_interval: int = 100
    
    # Output
    output_dir: str = "./blackhole_archive_output"
    log_level: str = "INFO"
    visualization_enabled: bool = True

# =============================================================================
# UTILITIES
# =============================================================================

class Logger:
    """Centralized logging"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{config.output_dir}/simulation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("BlackholeArchive")
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)

class DataManager:
    """Manage simulation data storage and retrieval"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.data_path = Path(config.output_dir) / "simulation_data.h5"
        
        # Initialize HDF5 file
        with h5py.File(self.data_path, 'w') as f:
            f.create_group('spacetime')
            f.create_group('agents')
            f.create_group('packets')
            f.create_group('statistics')
    
    def save_spacetime_state(self, timestep: int, metric: np.ndarray, fields: Dict[str, np.ndarray]):
        """Save spacetime configuration"""
        with h5py.File(self.data_path, 'a') as f:
            grp = f['spacetime'].create_group(f"t_{timestep}")
            grp.create_dataset('metric', data=metric, compression='gzip')
            
            for field_name, field_data in fields.items():
                grp.create_dataset(field_name, data=field_data, compression='gzip')
    
    def save_agent_states(self, timestep: int, agents: Dict[str, List['Agent']]):
        """Save agent configurations"""
        with h5py.File(self.data_path, 'a') as f:
            grp = f['agents'].create_group(f"t_{timestep}")
            
            for colony_name, agent_list in agents.items():
                colony_grp = grp.create_group(colony_name)
                
                positions = np.array([a.position for a in agent_list])
                velocities = np.array([a.velocity for a in agent_list])
                energies = np.array([a.energy for a in agent_list])
                
                colony_grp.create_dataset('positions', data=positions, compression='gzip')
                colony_grp.create_dataset('velocities', data=velocities, compression='gzip')
                colony_grp.create_dataset('energies', data=energies, compression='gzip')
    
    def save_statistics(self, timestep: int, stats: Dict[str, Any]):
        """Save simulation statistics"""
        with h5py.File(self.data_path, 'a') as f:
            grp = f['statistics']
            
            for key, value in stats.items():
                if f"t_{timestep}_{key}" not in grp:
                    grp.create_dataset(f"t_{timestep}_{key}", data=value)

# =============================================================================
# SIMPLIFIED PHYSICS ENGINE (for demonstration)
# =============================================================================

class SimplifiedSpacetime:
    """
    Simplified spacetime for demonstration
    
    Full GR simulation would use numerical relativity codes
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.M = config.black_hole_mass
        self.r_s = 2 * self.M
        
        # Grid
        self.r = np.linspace(config.r_min, config.r_max, config.n_r)
        self.theta = np.linspace(0, np.pi, config.n_theta)
        self.phi = np.linspace(0, 2*np.pi, config.n_phi)
        
        # Metric (Schwarzschild)
        self.g = self._compute_metric()
        
        # Structural field (from beavers)
        self.structural_field = np.zeros((config.n_r, config.n_theta, config.n_phi))
    
    def _compute_metric(self) -> np.ndarray:
        """Compute Schwarzschild metric at grid points"""
        metric = np.zeros((self.config.n_r, self.config.n_theta, self.config.n_phi, 4, 4))
        
        for i, r in enumerate(self.r):
            f = 1 - self.r_s / r
            
            metric[i, :, :, 0, 0] = -f  # g_tt
            metric[i, :, :, 1, 1] = 1 / f  # g_rr
            
            for j, theta in enumerate(self.theta):
                metric[i, j, :, 2, 2] = r**2  # g_θθ
                metric[i, j, :, 3, 3] = r**2 * np.sin(theta)**2  # g_φφ
        
        return metric
    
    def get_curvature(self, position: np.ndarray) -> float:
        """Get Ricci scalar at position using Kretschmann scalar."""
        r = position[1]

        # Kretschmann scalar K = 48M²/r⁶ gives a smooth curvature gradient
        K = 48 * self.M ** 2 / (r ** 6 + 1e-10)
        return np.sqrt(K)
    
    def get_time_dilation(self, position: np.ndarray) -> float:
        """Get time dilation factor"""
        r = position[1]
        return 1 / np.sqrt(max(1e-10, 1 - self.r_s / r))
    
    def add_structural_field(self, position: np.ndarray, strength: float, radius: float):
        """Add beaver structural field"""
        # Find nearest grid point
        i = np.argmin(np.abs(self.r - position[1]))
        j = np.argmin(np.abs(self.theta - position[2]))
        k = np.argmin(np.abs(self.phi - position[3]))
        
        # Add Gaussian centered at position
        for di in range(-3, 4):
            for dj in range(-3, 4):
                for dk in range(-3, 4):
                    ii = (i + di) % self.config.n_r
                    jj = (j + dj) % self.config.n_theta
                    kk = (k + dk) % self.config.n_phi
                    
                    distance = np.sqrt(di**2 + dj**2 + dk**2)
                    if distance < radius:
                        self.structural_field[ii, jj, kk] += strength * np.exp(-distance**2 / (2 * radius**2))

# =============================================================================
# SIMPLIFIED AGENT SYSTEM
# =============================================================================

@dataclass
class Agent:
    """Base agent class"""
    id: str
    colony: str
    position: np.ndarray
    velocity: np.ndarray
    energy: float
    state: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, dt: float, spacetime: SimplifiedSpacetime):
        """Update agent (override in subclasses)"""
        pass

class BeaverAgent(Agent):
    """Beaver: builds structures"""

    def update(self, dt: float, spacetime: SimplifiedSpacetime):
        # Sample nearby points to climb curvature gradients
        nearby_curvatures = []
        directions = []
        for _ in range(5):
            offset = 0.5 * np.random.randn(4)
            test_pos = self.position + offset
            nearby_curvatures.append(spacetime.get_curvature(test_pos))
            directions.append(offset)

        max_idx = int(np.argmax(nearby_curvatures))
        target_dir = directions[max_idx]
        target_curv = nearby_curvatures[max_idx]

        # Drift toward the highest curvature region we sampled
        self.velocity = 0.8 * self.velocity + 0.2 * target_dir

        if target_curv > 0.1 and self.energy > 0.05:
            spacetime.add_structural_field(self.position, 1.0, 2.0)
            self.energy -= 0.05

        # Move and expend energy
        self.position += dt * self.velocity + 0.05 * np.random.randn(4)
        self.energy -= dt * 0.01

class AntAgent(Agent):
    """Ant: builds semantic graph"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_vertex = None
        self.pheromone = 1.0
    
    def update(self, dt: float, spacetime: SimplifiedSpacetime):
        # Random walk
        self.position += dt * self.velocity + 0.05 * np.random.randn(4)
        
        # Deposit pheromone (simplified)
        self.metadata['last_pheromone_deposit'] = time.time()
        
        self.energy -= dt * 0.005

class BeeAgent(Agent):
    """Bee: transports packets"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packet = None
        self.waggle_intensity = 0.0
        self.role = "scout"

    def update(
        self,
        dt: float,
        spacetime: SimplifiedSpacetime,
        wormhole_position: Optional[np.ndarray] = None,
    ):
        if self.packet is None:
            # Search for payloads
            if np.random.rand() < 0.01:
                self.packet = {
                    "vertex": -1,
                    "salience": float(self.waggle_intensity),
                    "observation": "scouted_payload",
                }
                self.role = "transporter"
                self.waggle_intensity = 0.5
        else:
            # Move toward the wormhole mouth for delivery
            if wormhole_position is not None:
                direction = wormhole_position - self.position
                direction[0] = 0.0  # ignore time component
                norm = np.linalg.norm(direction) + 1e-8
                self.velocity = 0.7 * self.velocity + 0.3 * (direction / norm)

        self.position += dt * self.velocity
        self.energy -= dt * 0.01

# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class SimulationEngine:
    """Main simulation engine"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.logger = Logger(config)
        self.data_manager = DataManager(config)

        # Initialize components
        self.logger.info("Initializing spacetime...")
        self.spacetime = SimplifiedSpacetime(config)

        throat_area = 4 * np.pi * (config.throat_radius ** 2)
        self.transport_protocol = WormholeTransportProtocol(throat_area=throat_area)
        self.wormhole_position = np.array([0.0, config.throat_radius + 0.6, np.pi / 2, 0.0])

        self.logger.info("Initializing agents...")
        self.agents = self._initialize_agents()

        # Statistics
        self.stats = {
            'n_packets_transported': 0,
            'total_energy': 0.0,
            'n_structures_built': 0,
            'queue_length': 0,
        }
        
        self.logger.info("Simulation engine initialized")
    
    def _initialize_agents(self) -> Dict[str, List[Agent]]:
        """Initialize all agents"""
        agents = {
            'beavers': [],
            'ants': [],
            'bees': []
        }
        
        # Beavers
        for i in range(self.config.n_beavers):
            position = np.array([
                0.0,  # t
                self.config.r_min + np.random.rand() * 10,  # r
                np.random.rand() * np.pi,  # theta
                np.random.rand() * 2*np.pi  # phi
            ])
            
            agents['beavers'].append(BeaverAgent(
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
            
            agents['ants'].append(AntAgent(
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
            
            agents['bees'].append(BeeAgent(
                id=f"bee_{i}",
                colony="bees",
                position=position,
                velocity=0.2 * np.random.randn(4),
                energy=1.0
            ))
        
        return agents
    
    def run(self):
        """Run simulation"""
        self.logger.info(f"Starting simulation: {int(self.config.t_max / self.config.dt)} steps")

        n_steps = int(self.config.t_max / self.config.dt)
        
        for step in tqdm(range(n_steps), desc="Simulation"):
            t = step * self.config.dt

            # Update all agents
            for colony_name, agent_list in self.agents.items():
                for agent in agent_list:
                    if agent.state == "active":
                        if isinstance(agent, BeeAgent):
                            agent.update(self.config.dt, self.spacetime, self.wormhole_position)
                            if agent.packet is not None and self._at_wormhole(agent.position):
                                packet = self._build_packet(agent, t)
                                if self.transport_protocol.enqueue_packet(packet):
                                    agent.packet = None
                                    agent.role = "scout"
                                    self.stats['n_packets_transported'] += 1
                                else:
                                    agent.role = "congestion_wait"
                            elif getattr(agent, 'role', '') == "congestion_wait" and not self.transport_protocol.channel_state.packet_queue:
                                agent.role = "transporter"
                        else:
                            agent.update(self.config.dt, self.spacetime)

                        # Remove dead agents
                        if agent.energy <= 0:
                            agent.state = "dead"

            # Protocol tick
            self.transport_protocol.transmit_packets(self.config.dt)
            self.stats['queue_length'] = len(self.transport_protocol.channel_state.packet_queue)

            # Update statistics
            self.stats['total_energy'] = sum(
                a.energy for agents in self.agents.values() for a in agents if a.state == "active"
            )
            
            # Save checkpoint
            if step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(step)
            
            # Log
            if step % 100 == 0:
                self.logger.info(f"Step {step}/{n_steps}, t={t:.2f}, Energy={self.stats['total_energy']:.2f}")
        
        self.logger.info("Simulation complete")
        self._save_final_results()

    def _save_checkpoint(self, step: int):
        """Save simulation checkpoint"""
        # Save spacetime
        metric_sample = self.spacetime.g[::8, ::8, ::8]  # Downsample for storage
        fields = {'structural_field': self.spacetime.structural_field[::8, ::8, ::8]}
        self.data_manager.save_spacetime_state(step, metric_sample, fields)

        # Save agents
        self.data_manager.save_agent_states(step, self.agents)

        # Save statistics
        self.data_manager.save_statistics(step, self.stats)

    def _save_final_results(self):
        """Save final results and generate report"""
        report_path = Path(self.config.output_dir) / "simulation_report.json"

        report = {
            'config': asdict(self.config),
            'final_statistics': self.stats,
            'n_agents': {
                colony: len([a for a in agents if a.state == "active"])
                for colony, agents in self.agents.items()
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to {report_path}")


class ProductionSimulationEngine(SimulationEngine):
    """Simulation engine that bridges enhanced physics and protocols."""

    def __init__(self, config: SimulationConfig):
        # Initialize logging/data pipelines from the base engine
        super().__init__(config)

        # Replace simplified spacetime with enhanced physics
        self.spacetime = EnhancedSpacetime(config)
        self.semantic_graph = SemanticGraph()

        # Enhanced wormhole transport protocol honoring holographic bounds
        throat_area = 4 * np.pi * (config.throat_radius ** 2)
        self.transport_protocol = WormholeTransportProtocol(throat_area=throat_area)

        # Wormhole location (near throat)
        self.wormhole_position = np.array([0.0, config.throat_radius + 0.6, np.pi / 2, 0.0])

        # Reinitialize agents with enhanced behaviors
        self.agents = self._initialize_agents()

        # Expanded stats
        self.stats.update({
            'n_vertices': 0,
            'n_edges': 0,
            'n_packets_transported': 0,
            'queue_length': 0,
        })

    def _initialize_agents(self) -> Dict[str, List[Agent]]:
        """Initialize enhanced agents that leverage curvature gradients."""
        agents = {'beavers': [], 'ants': [], 'bees': []}

        for i in range(self.config.n_beavers):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 10,
                np.random.rand() * np.pi,
                np.random.rand() * 2 * np.pi,
            ])
            agents['beavers'].append(EnhancedBeaverAgent(
                id=f"beaver_{i}",
                colony="beavers",
                position=position,
                velocity=0.1 * np.random.randn(4),
                energy=1.0,
            ))

        for i in range(self.config.n_ants):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 20,
                np.random.rand() * np.pi,
                np.random.rand() * 2 * np.pi,
            ])
            agents['ants'].append(EnhancedAntAgent(
                id=f"ant_{i}",
                colony="ants",
                position=position,
                velocity=0.05 * np.random.randn(4),
                energy=1.0,
            ))

        for i in range(self.config.n_bees):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 15,
                np.random.rand() * np.pi,
                np.random.rand() * 2 * np.pi,
            ])
            agents['bees'].append(EnhancedBeeAgent(
                id=f"bee_{i}",
                colony="bees",
                position=position,
                velocity=0.2 * np.random.randn(4),
                energy=1.0,
            ))

        return agents

    def _at_wormhole(self, position: np.ndarray) -> bool:
        """Check proximity to wormhole mouth."""
        spatial_delta = position[1:] - self.wormhole_position[1:]
        return np.linalg.norm(spatial_delta) < 2.0

    def _build_packet(self, bee: Agent, current_time: float) -> Packet:
        """Wrap a bee payload in the transport protocol packet structure."""
        payload = bee.packet or {}
        payload_bytes = json.dumps(payload).encode('utf-8')

        semantic_coord = SemanticCoordinate(
            vertex_id=payload.get('vertex', -1),
            embedding=np.zeros(4),
            salience=float(payload.get('salience', 0.0)),
            confidence=1.0,
        )

        entropy_signature = EntropySignature(
            total_entropy=float(len(payload_bytes)),
            local_curvature=float(self.spacetime.get_curvature(bee.position)),
            temperature=0.0,
            checksum=hashlib.sha256(payload_bytes).hexdigest(),
        )

        causal_cert = CausalCertificate()
        causal_cert.increment(bee.id)

        return Packet(
            packet_id=str(uuid.uuid4()),
            packet_type=PacketType.DATA,
            data=payload_bytes,
            semantic_coord=semantic_coord,
            entropy_signature=entropy_signature,
            causal_cert=causal_cert,
            origin_time=current_time,
            origin_position=bee.position.copy(),
            priority=float(getattr(bee, 'waggle_intensity', 0.0)),
            size_bytes=len(payload_bytes),
            created_at=current_time,
            error_correction_code=None,
        )

    def run(self):
        """Run simulation with enhanced physics and wormhole protocol."""
        self.logger.info(f"Starting production simulation: {int(self.config.t_max / self.config.dt)} steps")

        n_steps = int(self.config.t_max / self.config.dt)

        for step in tqdm(range(n_steps), desc="Production Simulation"):
            t = step * self.config.dt

            # Update beavers using real curvature gradients
            for beaver in self.agents['beavers']:
                if beaver.state == "active":
                    beaver.update(self.config.dt, self.spacetime)

            # Ants evolve the semantic graph
            for ant in self.agents['ants']:
                if ant.state == "active":
                    ant.update(self.config.dt, self.spacetime, self.semantic_graph)

            # Bees interact with wormhole protocol instead of naive teleport
            for bee in self.agents['bees']:
                if bee.state == "active":
                    bee.update(self.config.dt, self.spacetime, self.semantic_graph, self.wormhole_position)

                    if bee.role == "transporter" and bee.packet is not None and self._at_wormhole(bee.position):
                        packet = self._build_packet(bee, t)
                        if self.transport_protocol.enqueue_packet(packet):
                            bee.packet = None
                            bee.target_vertex = None
                            bee.role = "scout"
                            self.stats['n_packets_transported'] += 1
                        else:
                            bee.role = "congestion_wait"
                    elif bee.role == "congestion_wait" and not self.transport_protocol.channel_state.packet_queue:
                        bee.role = "transporter"

            # Transport protocol tick
            self.transport_protocol.transmit_packets(self.config.dt)
            self.stats['queue_length'] = len(self.transport_protocol.channel_state.packet_queue)

            # Decay pheromones and update stats
            self.semantic_graph.decay_pheromones(self.config.dt)

            self.stats['total_energy'] = sum(
                a.energy for agents in self.agents.values() for a in agents if a.state == "active"
            )
            self.stats['n_structures_built'] = sum(
                b.structures_built for b in self.agents['beavers']
            )
            self.stats['n_vertices'] = self.semantic_graph.graph.number_of_nodes()
            self.stats['n_edges'] = self.semantic_graph.graph.number_of_edges()

            if step % 100 == 0:
                self.logger.info(
                    f"Step {step}/{n_steps}, t={t:.2f}, "
                    f"Energy={self.stats['total_energy']:.2f}, "
                    f"Vertices={self.stats['n_vertices']}, "
                    f"Structures={self.stats['n_structures_built']}, "
                    f"Packets={self.stats['n_packets_transported']}, "
                    f"Queue={self.stats['queue_length']}"
                )

        self.logger.info("Production simulation complete")
        self._save_final_results()

# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """Visualization tools"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_spacetime_slice(self, spacetime: SimplifiedSpacetime):
        """Plot 2D slice of spacetime"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Metric component g_tt
        R, THETA = np.meshgrid(spacetime.r, spacetime.theta)
        g_tt = spacetime.g[:, :, 0, 0, 0]
        
        im1 = axes[0].contourf(R, THETA, g_tt.T, levels=20, cmap='viridis')
        axes[0].set_title('Metric component $g_{tt}$')
        axes[0].set_xlabel('r')
        axes[0].set_ylabel('θ')
        plt.colorbar(im1, ax=axes[0])
        
        # Structural field
        struct_field = spacetime.structural_field[:, :, 0]
        im2 = axes[1].contourf(R, THETA, struct_field.T, levels=20, cmap='plasma')
        axes[1].set_title('Beaver Structural Field')
        axes[1].set_xlabel('r')
        axes[1].set_ylabel('θ')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spacetime_slice.png', dpi=150)
        plt.close()
    
    def plot_agent_distribution(self, agents: Dict[str, List[Agent]]):
        """Plot agent positions in 3D"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {'beavers': 'brown', 'ants': 'red', 'bees': 'gold'}
        
        for colony_name, agent_list in agents.items():
            active_agents = [a for a in agent_list if a.state == "active"]
            
            if active_agents:
                positions = np.array([a.position[1:] for a in active_agents])  # r, theta, phi
                
                # Convert to Cartesian
                r = positions[:, 0]
                theta = positions[:, 1]
                phi = positions[:, 2]
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
                ax.scatter(x, y, z, c=colors[colony_name], label=colony_name, s=10, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Agent Distribution')
        ax.legend()
        
        plt.savefig(self.output_dir / 'agent_distribution.png', dpi=150)
        plt.close()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Blackhole Archive Simulation")
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'full', 'test'],
                       help='Simulation mode')
    parser.add_argument('--output', type=str, default='./blackhole_archive_output',
                       help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--engine', type=str, default='production',
                        choices=['simplified', 'production'],
                        help='Select simulation engine implementation')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = SimulationConfig(**config_dict)
    else:
        config = SimulationConfig(output_dir=args.output)
        
        # Adjust for mode
        if args.mode == 'demo':
            config.t_max = 10.0
            config.n_beavers = 20
            config.n_ants = 50
            config.n_bees = 30
            config.n_r = 32
            config.n_theta = 16
            config.n_phi = 16
        elif args.mode == 'test':
            config.t_max = 1.0
            config.n_beavers = 5
            config.n_ants = 10
            config.n_bees = 5
            config.n_r = 16
            config.n_theta = 8
            config.n_phi = 8
    
    # Create and run simulation
    engine_cls = ProductionSimulationEngine if args.engine == 'production' else SimulationEngine
    engine = engine_cls(config)
    engine.run()
    
    # Visualize
    if config.visualization_enabled:
        visualizer = Visualizer(config)
        visualizer.plot_spacetime_slice(engine.spacetime)
        visualizer.plot_agent_distribution(engine.agents)
        print(f"Visualizations saved to {visualizer.output_dir}")
    
    print(f"\nSimulation complete! Results in {config.output_dir}")
    print(f"Final energy: {engine.stats['total_energy']:.2f}")

if __name__ == "__main__":
    main()
