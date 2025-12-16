# Blackhole Archive Advanced Visualizations
# Comprehensive visualization suite for simulation analysis

"""
VISUALIZATION SUITE:

1. Semantic Graph Visualization
   - Graph structure with vertex salience
   - Pheromone trail intensity on edges
   - Temporal evolution

2. Spacetime Analysis
   - 3D metric visualization
   - Structural field evolution
   - Curvature maps

3. Colony Dynamics
   - Agent trajectories over time
   - Energy levels per colony
   - Survival rates

4. Information Flow
   - Packet transport visualization
   - Wormhole throughput
   - Latency distributions

5. System Metrics Dashboard
   - Real-time performance metrics
   - Efficiency measures
   - Comparative analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import json

sns.set_style("darkgrid")

class AdvancedVisualizer:
    """Advanced visualization suite"""
    
    def __init__(self, engine):
        self.engine = engine
        self.output_dir = Path(engine.config.output_dir) / "advanced_visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'beavers': '#8B4513',
            'ants': '#DC143C',
            'bees': '#FFD700',
            'wormhole': '#9370DB',
            'vertices': '#4169E1',
            'pheromone': '#FF6347'
        }
    
    def visualize_all(self):
        """Generate all visualizations"""
        print("Generating advanced visualizations...")
        
        self.plot_semantic_graph()
        self.plot_system_metrics()
        self.plot_colony_dynamics()
        self.plot_information_density()
        self.plot_structural_field_3d()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def plot_semantic_graph(self):
        """Visualize semantic graph with pheromone trails"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create 2x2 grid
        gs = GridSpec(2, 2, figure=fig)
        
        # Graph structure
        ax1 = fig.add_subplot(gs[0, :])
        
        if self.engine.semantic_graph.graph.number_of_nodes() > 0:
            # Get positions from graph
            pos = {}
            for node in self.engine.semantic_graph.graph.nodes():
                node_pos = self.engine.semantic_graph.graph.nodes[node]['position']
                # Project to 2D (r, theta)
                r = node_pos[1]
                theta = node_pos[2]
                pos[node] = (r * np.cos(theta), r * np.sin(theta))
            
            # Get saliences for node colors
            saliences = [
                self.engine.semantic_graph.graph.nodes[node]['salience']
                for node in self.engine.semantic_graph.graph.nodes()
            ]
            
            # Get pheromone strengths for edge colors
            edge_colors = []
            edge_widths = []
            for edge in self.engine.semantic_graph.graph.edges():
                pheromone = self.engine.semantic_graph.get_pheromone(edge)
                edge_colors.append(pheromone)
                edge_widths.append(1 + 3 * pheromone)
            
            # Draw graph
            nx.draw_networkx_nodes(
                self.engine.semantic_graph.graph,
                pos,
                node_color=saliences,
                node_size=300,
                cmap='YlOrRd',
                vmin=0, vmax=1,
                ax=ax1
            )
            
            if edge_colors:
                nx.draw_networkx_edges(
                    self.engine.semantic_graph.graph,
                    pos,
                    edge_color=edge_colors,
                    width=edge_widths,
                    edge_cmap='Reds',
                    edge_vmin=0, edge_vmax=max(edge_colors) if edge_colors else 1,
                    arrows=True,
                    arrowsize=10,
                    ax=ax1
                )
            
            # Add labels
            labels = {node: str(node) for node in list(self.engine.semantic_graph.graph.nodes())[:10]}
            nx.draw_networkx_labels(
                self.engine.semantic_graph.graph,
                pos,
                labels,
                font_size=8,
                ax=ax1
            )
            
            ax1.set_title(f'Semantic Graph: {self.engine.semantic_graph.graph.number_of_nodes()} vertices, '
                         f'{self.engine.semantic_graph.graph.number_of_edges()} edges',
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Radial projection')
            ax1.set_ylabel('Angular projection')
        else:
            ax1.text(0.5, 0.5, 'No semantic graph yet', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=16)
            ax1.set_title('Semantic Graph', fontsize=14, fontweight='bold')
        
        # Graph metrics over time
        ax2 = fig.add_subplot(gs[1, 0])
        if self.engine.stats['vertices_history']:
            timesteps = np.arange(len(self.engine.stats['vertices_history'])) * 10 * self.engine.config.dt
            ax2.plot(timesteps, self.engine.stats['vertices_history'], 
                    color=self.colors['vertices'], linewidth=2, label='Vertices')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Number of Vertices')
            ax2.set_title('Semantic Graph Growth', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Degree distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if self.engine.semantic_graph.graph.number_of_nodes() > 0:
            degrees = [deg for node, deg in self.engine.semantic_graph.graph.degree()]
            ax3.hist(degrees, bins=20, color=self.colors['pheromone'], alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Degree')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Degree Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'semantic_graph_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_system_metrics(self):
        """Plot system-wide metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        timesteps = np.arange(len(self.engine.stats['energy_history'])) * 10 * self.engine.config.dt
        
        # Energy evolution
        axes[0, 0].plot(timesteps, self.engine.stats['energy_history'], 
                       color='#2E86AB', linewidth=2)
        axes[0, 0].set_xlabel('Time', fontsize=12)
        axes[0, 0].set_ylabel('Total Energy', fontsize=12)
        axes[0, 0].set_title('System Energy Evolution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Structures built
        axes[0, 1].plot(timesteps, self.engine.stats['structures_history'], 
                       color=self.colors['beavers'], linewidth=2)
        axes[0, 1].set_xlabel('Time', fontsize=12)
        axes[0, 1].set_ylabel('Structures Built', fontsize=12)
        axes[0, 1].set_title('Beaver Construction Activity', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Agent survival rates
        survival_data = {
            'Beavers': len([a for a in self.engine.agents['beavers'] if a.state == 'active']) / self.engine.config.n_beavers,
            'Ants': len([a for a in self.engine.agents['ants'] if a.state == 'active']) / self.engine.config.n_ants,
            'Bees': len([a for a in self.engine.agents['bees'] if a.state == 'active']) / self.engine.config.n_bees
        }
        
        colors_list = [self.colors['beavers'], self.colors['ants'], self.colors['bees']]
        axes[1, 0].bar(survival_data.keys(), [v * 100 for v in survival_data.values()], 
                      color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
        axes[1, 0].set_ylabel('Survival Rate (%)', fontsize=12)
        axes[1, 0].set_title('Colony Survival Rates', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylim([0, 105])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Information metrics
        info_metrics = {
            'Vertices': self.engine.stats['n_vertices'],
            'Edges': self.engine.stats['n_edges'],
            'Structures': self.engine.stats['n_structures_built'],
            'Packets': self.engine.stats['n_packets_transported']
        }
        
        axes[1, 1].bar(info_metrics.keys(), info_metrics.values(), 
                      color='#A23B72', alpha=0.7, edgecolor='black', linewidth=2)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Information Processing Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'system_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_colony_dynamics(self):
        """Plot colony-specific dynamics"""
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Beaver energy distribution
        ax1 = fig.add_subplot(gs[0, 0])
        beaver_energies = [b.energy for b in self.engine.agents['beavers'] if b.state == 'active']
        if beaver_energies:
            ax1.hist(beaver_energies, bins=20, color=self.colors['beavers'], alpha=0.7, edgecolor='black')
            ax1.axvline(np.mean(beaver_energies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(beaver_energies):.3f}')
            ax1.set_xlabel('Energy', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.set_title('Beaver Energy Distribution', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Ant energy distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ant_energies = [a.energy for a in self.engine.agents['ants'] if a.state == 'active']
        if ant_energies:
            ax2.hist(ant_energies, bins=20, color=self.colors['ants'], alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(ant_energies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ant_energies):.3f}')
            ax2.set_xlabel('Energy', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Ant Energy Distribution', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Bee energy distribution
        ax3 = fig.add_subplot(gs[0, 2])
        bee_energies = [b.energy for b in self.engine.agents['bees'] if b.state == 'active']
        if bee_energies:
            ax3.hist(bee_energies, bins=20, color=self.colors['bees'], alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(bee_energies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(bee_energies):.3f}')
            ax3.set_xlabel('Energy', fontsize=11)
            ax3.set_ylabel('Frequency', fontsize=11)
            ax3.set_title('Bee Energy Distribution', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Beaver productivity
        ax4 = fig.add_subplot(gs[1, 0])
        beaver_structures = [b.structures_built for b in self.engine.agents['beavers']]
        if beaver_structures and max(beaver_structures) > 0:
            ax4.hist(beaver_structures, bins=range(max(beaver_structures) + 2), 
                    color=self.colors['beavers'], alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Structures Built', fontsize=11)
            ax4.set_ylabel('Number of Beavers', fontsize=11)
            ax4.set_title('Beaver Productivity', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # Ant pheromone deposits
        ax5 = fig.add_subplot(gs[1, 1])
        ant_deposits = [a.pheromone_deposits for a in self.engine.agents['ants']]
        if ant_deposits and max(ant_deposits) > 0:
            ax5.hist(ant_deposits, bins=range(max(ant_deposits) + 2), 
                    color=self.colors['ants'], alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Pheromone Deposits', fontsize=11)
            ax5.set_ylabel('Number of Ants', fontsize=11)
            ax5.set_title('Ant Activity', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # Bee deliveries
        ax6 = fig.add_subplot(gs[1, 2])
        bee_deliveries = [b.packets_delivered for b in self.engine.agents['bees']]
        if bee_deliveries and max(bee_deliveries) > 0:
            ax6.hist(bee_deliveries, bins=range(max(bee_deliveries) + 2), 
                    color=self.colors['bees'], alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Packets Delivered', fontsize=11)
            ax6.set_ylabel('Number of Bees', fontsize=11)
            ax6.set_title('Bee Transport Activity', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'colony_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_information_density(self):
        """Plot information density field"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get 2D slice (theta = pi/2)
        theta_idx = self.engine.spacetime.config.n_theta // 2
        info_slice = self.engine.spacetime.information_density[:, theta_idx, :]
        
        # Create meshgrid
        R, PHI = np.meshgrid(self.engine.spacetime.r, 
                            self.engine.spacetime.phi)
        
        # Plot
        im1 = axes[0].contourf(R, PHI, info_slice.T, levels=20, cmap='viridis')
        axes[0].set_xlabel('Radius (r)', fontsize=12)
        axes[0].set_ylabel('Azimuth (φ)', fontsize=12)
        axes[0].set_title('Information Density Field', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=axes[0], label='Density')
        
        # Overlay ant positions
        active_ants = [a for a in self.engine.agents['ants'] if a.state == 'active']
        if active_ants:
            ant_positions = np.array([a.position for a in active_ants])
            axes[0].scatter(ant_positions[:, 1], ant_positions[:, 3], 
                          c='red', s=10, alpha=0.5, label='Ants')
            axes[0].legend()
        
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(R, PHI, info_slice.T, cmap='viridis', alpha=0.8)
        ax2.set_xlabel('r', fontsize=10)
        ax2.set_ylabel('φ', fontsize=10)
        ax2.set_zlabel('Density', fontsize=10)
        ax2.set_title('Information Density (3D)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'information_density.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_structural_field_3d(self):
        """Plot 3D structural field from beavers"""
        fig = plt.figure(figsize=(16, 6))
        
        # Get 2D slice
        theta_idx = self.engine.spacetime.config.n_theta // 2
        struct_slice = self.engine.spacetime.structural_field[:, theta_idx, :]
        
        # Meshgrid
        R, PHI = np.meshgrid(self.engine.spacetime.r, 
                            self.engine.spacetime.phi)
        
        # 2D contour
        ax1 = fig.add_subplot(121)
        im = ax1.contourf(R, PHI, struct_slice.T, levels=20, cmap='plasma')
        ax1.set_xlabel('Radius (r)', fontsize=12)
        ax1.set_ylabel('Azimuth (φ)', fontsize=12)
        ax1.set_title('Structural Field (Beaver Scaffolds)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Field Strength')
        
        # Overlay beaver positions
        active_beavers = [b for b in self.engine.agents['beavers'] if b.state == 'active']
        if active_beavers:
            beaver_positions = np.array([b.position for b in active_beavers])
            ax1.scatter(beaver_positions[:, 1], beaver_positions[:, 3], 
                       c='brown', s=30, alpha=0.7, edgecolors='black', linewidth=0.5, label='Beavers')
            ax1.legend()
        
        # 3D surface
        ax2 = fig.add_subplot(122, projection='3d')
        if np.max(struct_slice) > 0:
            surf = ax2.plot_surface(R, PHI, struct_slice.T, cmap='plasma', alpha=0.8)
            ax2.set_xlabel('r', fontsize=10)
            ax2.set_ylabel('φ', fontsize=10)
            ax2.set_zlabel('Field Strength', fontsize=10)
            ax2.set_title('Structural Field (3D)', fontsize=12, fontweight='bold')
        else:
            ax2.text2D(0.5, 0.5, 'No structures built yet', 
                      transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'structural_field_3d.png', dpi=300, bbox_inches='tight')
        plt.close()

# Usage
if __name__ == "__main__":
    # This would be called after running the enhanced simulation
    print("Import this module and use after running EnhancedSimulationEngine")
