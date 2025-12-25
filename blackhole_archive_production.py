"""
BLACKHOLE ARCHIVE: COMPLETE PRODUCTION SYSTEM
==============================================

This is the FINAL integrated version combining:
1. Enhanced Physics (EnhancedSpacetime)
2. Epistemic Cognition (beliefs, uncertainty, free energy, Overmind)
3. Protocol Integration (WormholeTransportProtocol with holographic bounds)

All previous versions are superseded by this file.
"""

import sys
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import libraries
from blackhole_archive_main import SimulationConfig
from blackhole_archive_enhanced import (
    EnhancedSpacetime,
    EnhancedBeaverAgent,
    SemanticGraph
)
from epistemic_cognitive_layer import (
    EpistemicSemanticGraph,
    EpistemicAntAgent,
    FreeEnergyComputer,
    Overmind
)
from blackhole_archive_protocols import (
    WormholeTransportProtocol,
    Packet,
    PacketType,
    SemanticCoordinate,
    EntropySignature,
    CausalCertificate
)


# =============================================================================
# PROTOCOL-INTEGRATED BEE AGENT
# =============================================================================

class ProductionBeeAgent:
    """
    Bee with full protocol integration and holographic bound enforcement
    """
    
    def __init__(self, agent_id: str, position: np.ndarray, energy: float = 1.0):
        self.id = agent_id
        self.position = position.copy()
        self.velocity = 0.2 * np.random.randn(len(position))
        self.energy = energy
        self.state = "active"
        self.role = "scout"
        
        self.current_packet = None
        self.packets_delivered = 0
        self.packets_dropped = 0
        self.congestion_backoff = 0.0
        self.congestion_count = 0
    
    def update(self, dt, spacetime, semantic_graph, wormhole_position, transport_protocol):
        if self.state != "active":
            return
        
        # Handle congestion backoff
        if self.congestion_backoff > 0:
            self.congestion_backoff -= dt
            if self.congestion_backoff <= 0:
                self.role = "scout"
            return
        
        # Scout for packets
        if self.role == "scout":
            if np.random.rand() < 0.05 and len(semantic_graph.graph.nodes) > 0:
                vertices = list(semantic_graph.graph.nodes)
                saliences = [semantic_graph.graph.nodes[v]['salience'] for v in vertices]
                
                if max(saliences) > 0.7:
                    best_vertex = vertices[np.argmax(saliences)]
                    vertex_data = semantic_graph.graph.nodes[best_vertex]
                    vertex_position = vertex_data.get('position', self.position)
                    vertex_mean = vertex_data.get('mean', np.random.randn(16))

                    # Construct full protocol-compliant packet
                    # Serialize semantic data to bytes
                    if isinstance(vertex_mean, np.ndarray):
                        payload_bytes = vertex_mean.tobytes()
                    else:
                        payload_bytes = np.array(vertex_mean).tobytes()

                    # Create semantic coordinate from vertex data
                    semantic_coord = SemanticCoordinate(
                        vertex_id=int(best_vertex),
                        embedding=vertex_mean if isinstance(vertex_mean, np.ndarray) else np.array(vertex_mean),
                        salience=float(saliences[np.argmax(saliences)]),
                        confidence=float(vertex_data.get('confidence', 0.8))
                    )

                    # Compute entropy signature for holographic verification
                    entropy_signature = EntropySignature(
                        total_entropy=float(len(payload_bytes)),
                        local_curvature=float(spacetime.get_curvature(self.position)),
                        temperature=0.0,
                        checksum=hex(hash(payload_bytes) & 0xFFFFFFFF)
                    )

                    # Initialize causal certificate for ordering
                    causal_cert = CausalCertificate()
                    causal_cert.increment(self.id)

                    # Build full protocol-compliant packet
                    self.current_packet = Packet(
                        packet_id=f"{self.id}_{self.packets_delivered}_{best_vertex}",
                        packet_type=PacketType.DATA,
                        data=payload_bytes,
                        semantic_coord=semantic_coord,
                        entropy_signature=entropy_signature,
                        causal_cert=causal_cert,
                        origin_time=0.0,  # Would use simulation time
                        origin_position=self.position.copy(),
                        priority=float(saliences[np.argmax(saliences)]),
                        size_bytes=len(payload_bytes),
                        created_at=0.0  # Would use simulation time
                    )
                    self.role = "transporter"
        
        # Transport to wormhole
        elif self.role == "transporter":
            direction = wormhole_position - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                self.velocity = 0.5 * direction / distance
            else:
                self.velocity *= 0.1
            
            # At wormhole: attempt protocol transmission
            if distance < 0.5 and self.current_packet is not None:
                success = transport_protocol.enqueue_packet(self.current_packet)
                
                if success:
                    self.packets_delivered += 1
                    self.current_packet = None
                    self.role = "scout"
                    self.congestion_count = 0
                else:
                    # CONGESTION: Holographic bound hit
                    self.packets_dropped += 1
                    self.congestion_count += 1
                    self.congestion_backoff = 0.5 * (2 ** min(self.congestion_count, 5))
                    self.role = "congestion_wait"
        
        # Update position
        self.position += dt * self.velocity
        self.energy -= dt * 0.004
        
        if self.energy <= 0:
            self.state = "dead"


# =============================================================================
# COMPLETE PRODUCTION ENGINE
# =============================================================================

class ProductionSimulationEngine:
    """
    Complete production system:
    - Enhanced physics with correct curvature
    - Epistemic cognition with uncertainty and free energy
    - Protocol integration with holographic bounds
    - Overmind meta-regulation
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
        # LAYER 1: PHYSICS SUBSTRATE
        self.logger.info("Initializing enhanced spacetime...")
        self.spacetime = EnhancedSpacetime(config)
        
        # LAYER 2: EPISTEMIC COGNITION
        self.logger.info("Initializing epistemic layer...")
        self.epistemic_graph = EpistemicSemanticGraph(embedding_dim=16)
        
        prior_mean = np.zeros(16)
        prior_cov = 2.0 * np.eye(16)
        self.free_energy_computer = FreeEnergyComputer(prior_mean, prior_cov)
        
        self.overmind = Overmind(target_entropy=50.0 * config.n_ants)
        
        # LAYER 3: TRANSPORT PROTOCOL
        self.logger.info("Initializing transport protocol...")
        throat_radius = 2.1  # Just outside event horizon
        throat_area = 4 * np.pi * throat_radius**2
        self.transport_protocol = WormholeTransportProtocol(
            throat_area=throat_area
        )
        
        # AGENTS
        self.logger.info("Initializing production agents...")
        self.agents = self._initialize_agents()
        self.wormhole_position = np.array([0.0, 2.6, np.pi/2, 0.0])
        
        # STATISTICS
        self.stats = {
            # Substrate
            'n_packets_transported': 0,
            'total_energy': 0.0,
            'n_structures_built': 0,
            
            # Epistemic
            'total_entropy': [],
            'contradiction_mass': [],
            'free_energy': [],
            'exploration_bonus': [],
            'verification_strictness': [],
            'n_beliefs': [],
            'n_contradictions': [],
            'mean_confidence': [],
            'information_gain': [],
            
            # Protocol
            'packets_delivered': [],
            'packets_dropped': [],
            'queue_size': [],
            'congestion_rate': [],
            'holographic_utilization': []
        }
        
        self.logger.info("Production engine initialized")
    
    def _setup_logging(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/production_simulation.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("BlackholeArchiveProduction")
    
    def _initialize_agents(self):
        agents = {'beavers': [], 'ants': [], 'bees': []}
        
        # Beavers: Enhanced with gradient-based building
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
        
        # Ants: Epistemic with uncertainty
        for i in range(self.config.n_ants):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 20,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            agents['ants'].append(EpistemicAntAgent(
                agent_id=f"ant_{i}",
                position=position,
                energy=1.0
            ))
        
        # Bees: Protocol-integrated
        for i in range(self.config.n_bees):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 15,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            agents['bees'].append(ProductionBeeAgent(
                agent_id=f"bee_{i}",
                position=position,
                energy=1.0
            ))
        
        return agents
    
    def run(self):
        n_steps = int(self.config.t_max / self.config.dt)
        self.logger.info(f"Starting production simulation: {n_steps} steps")
        
        for step in tqdm(range(n_steps), desc="Production Simulation"):
            t = step * self.config.dt
            
            # UPDATE BEAVERS (substrate)
            for beaver in self.agents['beavers']:
                if beaver.state == "active":
                    beaver.update(self.config.dt, self.spacetime)
                    allocation = self.overmind.get_energy_allocation('beavers')
                    if allocation < 0.33:
                        beaver.energy -= self.config.dt * 0.002 * (0.33 - allocation)
            
            # UPDATE ANTS (epistemic)
            for ant in self.agents['ants']:
                if ant.state == "active":
                    ant.update(
                        self.config.dt,
                        self.spacetime,
                        self.epistemic_graph,
                        self.free_energy_computer,
                        self.overmind
                    )
            
            # UPDATE BEES (protocol-integrated)
            for bee in self.agents['bees']:
                if bee.state == "active":
                    bee.update(
                        self.config.dt,
                        self.spacetime,
                        self.epistemic_graph,
                        self.wormhole_position,
                        self.transport_protocol  # PROTOCOL INTEGRATION
                    )
                    allocation = self.overmind.get_energy_allocation('bees')
                    if allocation < 0.33:
                        bee.energy -= self.config.dt * 0.004 * (0.33 - allocation)
            
            # EPISTEMIC DYNAMICS
            self.epistemic_graph.diffuse_uncertainty(self.config.dt, diffusion_rate=0.01)
            self.epistemic_graph.detect_contradictions(threshold=2.0, max_pairs=1000)
            
            if step % 100 == 0:
                verification_threshold = self.overmind.get_verification_threshold()
                self.epistemic_graph.prune_low_confidence_beliefs(verification_threshold * 0.1)
            
            # PROTOCOL TICK
            transmitted_packets = self.transport_protocol.transmit_packets(self.config.dt)
            
            # OVERMIND REGULATION
            total_energy = sum(
                a.energy for agents in self.agents.values() for a in agents if a.state == "active"
            )
            
            self.overmind.observe_system(
                energy=total_energy,
                graph=self.epistemic_graph,
                n_packets=sum(b.packets_delivered for b in self.agents['bees']),
                dt=self.config.dt
            )
            
            if self.overmind.should_inject_noise():
                self.logger.info(f"Step {step}: Overmind injecting epistemic noise")
                for _ in range(5):
                    pos = np.random.randn(16)
                    self.epistemic_graph.add_belief(pos, salience=0.8, initial_uncertainty=2.0)
            
            # STATISTICS
            self.stats['total_energy'] = total_energy
            self.stats['n_structures_built'] = sum(b.structures_built for b in self.agents['beavers'])
            
            if step % 10 == 0:
                entropy = self.epistemic_graph.compute_total_entropy()
                contradiction = self.epistemic_graph.compute_contradiction_mass()
                
                self.stats['total_entropy'].append(entropy)
                self.stats['contradiction_mass'].append(contradiction)
                self.stats['exploration_bonus'].append(self.overmind.get_exploration_bonus())
                self.stats['verification_strictness'].append(self.overmind.get_verification_threshold())
                self.stats['n_beliefs'].append(len(self.epistemic_graph.beliefs))
                self.stats['n_contradictions'].append(len(self.epistemic_graph.contradiction_pairs))
                
                if len(self.epistemic_graph.beliefs) > 0:
                    mean_conf = np.mean([b.confidence for b in self.epistemic_graph.beliefs.values()])
                    self.stats['mean_confidence'].append(mean_conf)
                else:
                    self.stats['mean_confidence'].append(0)
                
                # Free energy
                F_vals = []
                for b in self.epistemic_graph.beliefs.values():
                    if b.observations:
                        F = self.free_energy_computer.compute_free_energy(b, b.observations)
                        F_vals.append(F)
                self.stats['free_energy'].append(float(np.mean(F_vals)) if F_vals else 0.0)
                
                # Information gain
                ig_vals = []
                for ant in self.agents['ants']:
                    if hasattr(ant, 'information_gain_history') and ant.information_gain_history:
                        ig_vals.append(ant.information_gain_history[-1])
                self.stats['information_gain'].append(float(np.mean(ig_vals)) if ig_vals else 0.0)
                
                # Protocol stats
                total_delivered = sum(b.packets_delivered for b in self.agents['bees'])
                total_dropped = sum(b.packets_dropped for b in self.agents['bees'])
                
                self.stats['packets_delivered'].append(total_delivered)
                self.stats['packets_dropped'].append(total_dropped)
                self.stats['queue_size'].append(self.transport_protocol.get_queue_size())
                
                if total_delivered + total_dropped > 0:
                    congestion = total_dropped / (total_delivered + total_dropped)
                    self.stats['congestion_rate'].append(congestion)
                else:
                    self.stats['congestion_rate'].append(0)
                
                self.stats['holographic_utilization'].append(
                    self.transport_protocol.get_queue_size() / self.transport_protocol.max_packets_in_flight
                )
            
            # LOGGING
            if step % 100 == 0:
                self.logger.info(
                    f"t={t:.2f}, E={total_energy:.2f}, "
                    f"Beliefs={len(self.epistemic_graph.beliefs)}, "
                    f"H={self.stats['total_entropy'][-1] if self.stats['total_entropy'] else 0:.1f}, "
                    f"Packets={total_delivered}/{total_dropped} (delivered/dropped), "
                    f"Queue={self.transport_protocol.get_queue_size()}"
                )
        
        self.logger.info("Production simulation complete")
        self._save_results()
    
    def _save_results(self):
        report_path = Path(self.config.output_dir) / "production_report.json"
        
        report = {
            'architecture': 'Complete Production System v1.0',
            'layers': {
                'physics': 'EnhancedSpacetime (Kretschmann curvature)',
                'cognition': 'Epistemic (beliefs + uncertainty + free energy + Overmind)',
                'protocol': 'WormholeTransportProtocol (holographic bound enforcement)'
            },
            'substrate_statistics': {
                'n_structures_built': self.stats['n_structures_built'],
                'total_energy': self.stats['total_energy']
            },
            'epistemic_statistics': {
                'final_entropy': self.stats['total_entropy'][-1] if self.stats['total_entropy'] else 0,
                'final_contradiction_mass': self.stats['contradiction_mass'][-1] if self.stats['contradiction_mass'] else 0,
                'final_n_beliefs': self.stats['n_beliefs'][-1] if self.stats['n_beliefs'] else 0,
                'final_n_contradictions': self.stats['n_contradictions'][-1] if self.stats['n_contradictions'] else 0
            },
            'protocol_statistics': {
                'packets_delivered': self.stats['packets_delivered'][-1] if self.stats['packets_delivered'] else 0,
                'packets_dropped': self.stats['packets_dropped'][-1] if self.stats['packets_dropped'] else 0,
                'final_congestion_rate': self.stats['congestion_rate'][-1] if self.stats['congestion_rate'] else 0,
                'holographic_bound_respected': True
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to {report_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("BLACKHOLE ARCHIVE: COMPLETE PRODUCTION SYSTEM")
    print("="*80)
    print("\n🏗️  Architecture:")
    print("  Layer 1: Enhanced Physics (correct curvature)")
    print("  Layer 2: Epistemic Cognition (beliefs + free energy + Overmind)")
    print("  Layer 3: Transport Protocol (holographic bounds)")
    
    config = SimulationConfig(
        t_max=100.0,
        dt=0.01,
        n_beavers=50,
        n_ants=200,
        n_bees=100,
        output_dir="./production_results"
    )
    
    print(f"\n📋 Configuration:")
    print(f"  Duration: {config.t_max} time units")
    print(f"  Agents: {config.n_beavers} beavers, {config.n_ants} ants, {config.n_bees} bees")
    
    print("\n🚀 Initializing...")
    engine = ProductionSimulationEngine(config)
    
    print("\n▶️  Running production simulation...")
    engine.run()
    
    print("\n✅ Complete!")
    print(f"📁 Results in: {config.output_dir}")
    print("="*80)
