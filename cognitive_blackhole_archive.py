"""
BLACKHOLE ARCHIVE: COMPLETE INTEGRATED SYSTEM
Architecture (substrate) + Cognition (inference)

This combines:
1. Physical spacetime substrate
2. Colony-based coordination
3. Epistemic beliefs with uncertainty
4. Free energy minimization
5. Overmind meta-regulation

The transition from settlement to cognition.
"""

import sys
sys.path.append('.')

from blackhole_archive_main import SimulationConfig
from blackhole_archive_enhanced import *
from epistemic_cognitive_layer import *
import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import logging

# =============================================================================
# COMPLETE INTEGRATED SIMULATION
# =============================================================================

class CognitiveSimulationEngine:
    """
    Complete simulation with both substrate and cognition
    
    Architecture: Spacetime + Agents + Wormholes
    Cognition: Beliefs + Uncertainty + Free Energy + Overmind
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
        # SUBSTRATE (from enhanced version)
        self.logger.info("Initializing spacetime substrate...")
        self.spacetime = EnhancedSpacetime(config)
        
        # COGNITION (new epistemic layer)
        self.logger.info("Initializing epistemic layer...")
        self.epistemic_graph = EpistemicSemanticGraph(embedding_dim=16)
        
        # Free energy computer
        prior_mean = np.zeros(16)
        prior_cov = 2.0 * np.eye(16)
        self.free_energy_computer = FreeEnergyComputer(prior_mean, prior_cov)
        
        # Overmind regulator
        self.overmind = Overmind(target_entropy=50.0 * config.n_ants)
        
        # Agents
        self.logger.info("Initializing cognitive agents...")
        self.agents = self._initialize_cognitive_agents()
        
        # Wormhole
        self.wormhole_position = np.array([0.0, 2.6, np.pi/2, 0.0])
        
        # Statistics
        self.stats = {
            # Substrate metrics
            'n_packets_transported': 0,
            'total_energy': 0.0,
            'n_structures_built': 0,
            
            # Epistemic metrics  
            'total_entropy': [],
            'contradiction_mass': [],
            'free_energy': [],
            'exploration_bonus': [],
            'verification_strictness': [],
            
            # Belief dynamics
            'n_beliefs': [],
            'n_contradictions': [],
            'mean_confidence': [],
            'information_gain': []
        }
        
        self.logger.info("Cognitive simulation engine initialized")
    
    def _setup_logging(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/cognitive_simulation.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("CognitiveBlackholeArchive")
    
    def _initialize_cognitive_agents(self):
        """Initialize agents with epistemic capabilities"""
        agents = {'beavers': [], 'ants': [], 'bees': []}
        
        # Beavers (same as enhanced)
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
        
        # Epistemic ants (NEW)
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
        
        # Bees (same as enhanced)
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
    
    def run(self):
        """Run complete cognitive simulation"""
        n_steps = int(self.config.t_max / self.config.dt)
        self.logger.info(f"Starting cognitive simulation: {n_steps} steps")
        
        for step in tqdm(range(n_steps), desc="Cognitive Simulation"):
            t = step * self.config.dt
            
            # Update beavers (substrate)
            for beaver in self.agents['beavers']:
                if beaver.state == "active":
                    beaver.update(self.config.dt, self.spacetime)
                    # Apply Overmind energy allocation
                    allocation = self.overmind.get_energy_allocation('beavers')
                    if allocation < 0.33:  # Getting less than baseline
                        beaver.energy -= self.config.dt * 0.002 * (0.33 - allocation)
            
            # Update epistemic ants (COGNITION)
            for ant in self.agents['ants']:
                if ant.state == "active":
                    ant.update(
                        self.config.dt,
                        self.spacetime,
                        self.epistemic_graph,
                        self.free_energy_computer,
                        self.overmind
                    )
            
            # Update bees (substrate)
            for bee in self.agents['bees']:
                if bee.state == "active":
                    bee.update(
                        self.config.dt,
                        self.spacetime,
                        self.epistemic_graph,  # Uses epistemic graph now
                        self.wormhole_position
                    )
                    # Apply Overmind energy allocation
                    allocation = self.overmind.get_energy_allocation('bees')
                    if allocation < 0.33:  # Getting less than baseline
                        bee.energy -= self.config.dt * 0.004 * (0.33 - allocation)
            
            # Epistemic dynamics
            self.epistemic_graph.diffuse_uncertainty(self.config.dt, diffusion_rate=0.01)
            self.epistemic_graph.detect_contradictions(threshold=2.0)
            
            # Prune low-confidence beliefs
            if step % 100 == 0:
                verification_threshold = self.overmind.get_verification_threshold()
                self.epistemic_graph.prune_low_confidence_beliefs(verification_threshold * 0.1)
            
            # Overmind regulation
            total_energy = sum(
                a.energy for agents in self.agents.values() for a in agents if a.state == "active"
            )
            
            self.overmind.observe_system(
                energy=total_energy,
                graph=self.epistemic_graph,
                n_packets=sum(b.packets_delivered for b in self.agents['bees']),
                dt=self.config.dt
            )
            
            # Inject noise if too stable
            if self.overmind.should_inject_noise():
                self.logger.info(f"Step {step}: Overmind injecting epistemic noise")
                # Add random high-uncertainty beliefs
                for _ in range(5):
                    pos = np.random.randn(16)
                    self.epistemic_graph.add_belief(pos, salience=0.8, initial_uncertainty=2.0)
            
            # Update statistics
            self.stats['total_energy'] = total_energy
            self.stats['n_structures_built'] = sum(b.structures_built for b in self.agents['beavers'])
            self.stats['n_packets_transported'] = sum(b.packets_delivered for b in self.agents['bees'])
            
            # Epistemic stats
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
                
                # FIXED: Actually compute free energy
                F_vals = []
                for b in self.epistemic_graph.beliefs.values():
                    if b.observations:
                        F = self.free_energy_computer.compute_free_energy(b, b.observations)
                        F_vals.append(F)
                self.stats['free_energy'].append(float(np.mean(F_vals)) if F_vals else 0.0)
                
                # FIXED: Actually compute information gain
                ig_vals = []
                for ant in self.agents['ants']:
                    if hasattr(ant, 'information_gain_history') and ant.information_gain_history:
                        ig_vals.append(ant.information_gain_history[-1])
                self.stats['information_gain'].append(float(np.mean(ig_vals)) if ig_vals else 0.0)
            
            # Log
            if step % 100 == 0:
                self.logger.info(
                    f"Step {step}/{n_steps}, t={t:.2f}, "
                    f"Energy={total_energy:.2f}, "
                    f"Beliefs={len(self.epistemic_graph.beliefs)}, "
                    f"Entropy={self.stats['total_entropy'][-1] if self.stats['total_entropy'] else 0:.2f}, "
                    f"Contradictions={len(self.epistemic_graph.contradiction_pairs)}, "
                    f"ExploreBonus={self.overmind.get_exploration_bonus():.2f}"
                )
        
        self.logger.info("Cognitive simulation complete")
        self._save_results()
    
    def _save_results(self):
        """Save results including epistemic metrics"""
        report_path = Path(self.config.output_dir) / "cognitive_simulation_report.json"
        
        report = {
            'config': {
                'black_hole_mass': self.config.black_hole_mass,
                'n_beavers': self.config.n_beavers,
                'n_ants': self.config.n_ants,
                'n_bees': self.config.n_bees,
                't_max': self.config.t_max
            },
            'substrate_statistics': {
                'n_structures_built': self.stats['n_structures_built'],
                'n_packets_transported': self.stats['n_packets_transported'],
                'total_energy': self.stats['total_energy']
            },
            'epistemic_statistics': {
                'final_entropy': self.stats['total_entropy'][-1] if self.stats['total_entropy'] else 0,
                'final_contradiction_mass': self.stats['contradiction_mass'][-1] if self.stats['contradiction_mass'] else 0,
                'final_n_beliefs': self.stats['n_beliefs'][-1] if self.stats['n_beliefs'] else 0,
                'final_n_contradictions': self.stats['n_contradictions'][-1] if self.stats['n_contradictions'] else 0,
                'final_mean_confidence': self.stats['mean_confidence'][-1] if self.stats['mean_confidence'] else 0
            },
            'overmind_final_state': {
                'exploration_bonus': self.overmind.get_exploration_bonus(),
                'verification_strictness': self.overmind.get_verification_threshold(),
                'energy_allocation': self.overmind.energy_allocation
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to {report_path}")


# =============================================================================
# EPISTEMIC VISUALIZATION
# =============================================================================

def visualize_epistemic_dynamics(engine: CognitiveSimulationEngine):
    """Generate visualizations specific to epistemic dynamics"""
    
    output_dir = Path(engine.config.output_dir) / "epistemic_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    timesteps = np.arange(len(engine.stats['total_entropy'])) * 10 * engine.config.dt
    
    # Panel 1: Total entropy over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(timesteps, engine.stats['total_entropy'], 'b-', linewidth=2)
    ax1.axhline(y=engine.overmind.target_entropy, color='r', linestyle='--', 
                linewidth=2, label='Target Entropy')
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Entropy H[q]', fontsize=12, fontweight='bold')
    ax1.set_title('Belief Uncertainty Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Contradiction mass
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(timesteps, engine.stats['contradiction_mass'], 'r-', linewidth=2)
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Contradiction Mass S_t', fontsize=12, fontweight='bold')
    ax2.set_title('Semantic Tension', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Number of beliefs vs contradictions
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(timesteps, engine.stats['n_beliefs'], 'g-', linewidth=2, label='Total Beliefs')
    ax3.plot(timesteps, engine.stats['n_contradictions'], 'r-', linewidth=2, label='Contradictions')
    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Beliefs vs Contradictions', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Mean confidence
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(timesteps, engine.stats['mean_confidence'], 'm-', linewidth=2)
    ax4.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mean Confidence', fontsize=12, fontweight='bold')
    ax4.set_title('Average Belief Certainty', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Overmind control parameters
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(timesteps, engine.stats['exploration_bonus'], 'b-', linewidth=2, label='Exploration β')
    ax5.plot(timesteps, engine.stats['verification_strictness'], 'r-', linewidth=2, label='Verification λ')
    ax5.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Parameter Value', fontsize=12, fontweight='bold')
    ax5.set_title('Overmind Control Parameters', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Phase space: Entropy vs Contradiction
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(engine.stats['total_entropy'], 
                         engine.stats['contradiction_mass'],
                         c=timesteps, cmap='viridis', s=30, alpha=0.6)
    ax6.set_xlabel('Total Entropy', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Contradiction Mass', fontsize=12, fontweight='bold')
    ax6.set_title('Epistemic Phase Space', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax6, label='Time')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('EPISTEMIC DYNAMICS: The Missing Upper Half', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(output_dir / 'epistemic_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Epistemic visualization saved to {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COMPLETE COGNITIVE BLACKHOLE ARCHIVE")
    print("Architecture + Cognition = Intelligence")
    print("="*80)
    
    config = SimulationConfig(
        t_max=100.0,
        dt=0.01,
        n_beavers=50,
        n_ants=200,
        n_bees=100,
        output_dir="./cognitive_results"
    )
    
    print("\n📋 Configuration:")
    print(f"  Duration: {config.t_max} time units")
    print(f"  Agents: {config.n_beavers} beavers, {config.n_ants} ants, {config.n_bees} bees")
    print(f"  Output: {config.output_dir}")
    
    print("\n🔬 Initializing complete system...")
    engine = CognitiveSimulationEngine(config)
    
    print("\n🚀 Running cognitive simulation...")
    print("   This includes:")
    print("   ✓ Substrate: Spacetime + Agents + Wormholes")
    print("   ✓ Beliefs: Uncertainty + Bayesian updates")
    print("   ✓ Free Energy: Information gain drive")
    print("   ✓ Overmind: Epistemic pressure regulation")
    print()
    
    engine.run()
    
    print("\n📊 Generating epistemic visualizations...")
    visualize_epistemic_dynamics(engine)
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    
    print("\n📈 Results:")
    print(f"  Substrate:")
    print(f"    Structures: {engine.stats['n_structures_built']}")
    print(f"    Packets: {engine.stats['n_packets_transported']}")
    print(f"  Cognition:")
    print(f"    Beliefs: {engine.stats['n_beliefs'][-1] if engine.stats['n_beliefs'] else 0}")
    print(f"    Contradictions: {engine.stats['n_contradictions'][-1] if engine.stats['n_contradictions'] else 0}")
    print(f"    Entropy: {engine.stats['total_entropy'][-1] if engine.stats['total_entropy'] else 0:.1f}")
    print(f"  Overmind:")
    print(f"    Exploration: {engine.overmind.get_exploration_bonus():.2f}")
    print(f"    Verification: {engine.overmind.get_verification_threshold():.2f}")
    
    print(f"\n📁 All results in: {config.output_dir}")
    print(f"\n🎯 This is the complete system:")
    print(f"   Architecture (substrate) ✓")
    print(f"   Cognition (inference) ✓")
    print(f"   The upper half is no longer missing.")
    print("="*80 + "\n")
