#!/usr/bin/env python3
"""
BLACKHOLE ARCHIVE: COMPLETE ADVANCED ANALYSIS & VISUALIZATION
Comprehensive analysis suite with publication-quality visualizations
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats, signal
from scipy.optimize import curve_fit
import warnings
import re
warnings.filterwarnings('ignore')

sys.path.append('.')

sns.set_style("darkgrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

def find_results_directory(specified_path=None):
    """
    Find the results directory to analyze.

    Priority:
    1. Command-line specified path
    2. Most recent timestamped subdirectory in results/
    3. Legacy directory names (enhanced_results, blackhole_archive_output, etc.)
    """
    # If user specified a path, use it
    if specified_path:
        p = Path(specified_path)
        if p.exists():
            return p
        else:
            print(f"❌ Specified path does not exist: {specified_path}")
            sys.exit(1)

    # Check for timestamped subdirectories in results/
    results_root = Path("results")
    if results_root.exists():
        subdirs = sorted(
            [d for d in results_root.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True  # Most recent first
        )
        for subdir in subdirs:
            if (subdir / "simulation_report.json").exists():
                return subdir
            if (subdir / "enhanced_simulation_report.json").exists():
                return subdir

    # Legacy: check older directory names
    legacy_dirs = [
        Path("enhanced_results"),
        Path("results"),
        Path("/tmp/simulation_test"),
        Path("blackhole_archive_output"),
    ]

    for d in legacy_dirs:
        if d.exists():
            if (d / "enhanced_simulation_report.json").exists():
                return d
            elif (d / "simulation_report.json").exists():
                return d

    return None

# Parse command-line argument for results directory
import argparse
parser = argparse.ArgumentParser(description="Analyze Blackhole Archive simulation results")
parser.add_argument('--results', '-r', type=str, default=None,
                    help='Path to results directory (default: auto-detect most recent)')
args, _ = parser.parse_known_args()

RESULTS_DIR = find_results_directory(args.results)

if RESULTS_DIR is None:
    print("❌ No results directory found.")
    print("\nLooked in:")
    print("  - results/<timestamp>_<engine>/ subdirectories")
    print("  - enhanced_results/")
    print("  - blackhole_archive_output/")
    print("\nRun the simulation first:")
    print("  python blackhole_archive_main.py --engine production --mode demo")
    print("\nOr specify a path:")
    print("  python complete_analysis_viz.py --results path/to/results")
    sys.exit(1)

VIZ_DIR = RESULTS_DIR / "advanced_visualizations"
VIZ_DIR.mkdir(exist_ok=True)

COLORS = {
    'beavers': '#8B4513',
    'ants': '#DC143C',
    'bees': '#FFD700',
    'energy': '#2E86AB',
    'structures': '#A23B72',
    'vertices': '#4169E1',
    'packets': '#FF6347'
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("🔬 BLACKHOLE ARCHIVE: COMPLETE ANALYSIS & VISUALIZATION")
print("="*80)

print(f"\n📂 Loading simulation data from {RESULTS_DIR}...")

# Try different report file names
report_file = None
for name in ["enhanced_simulation_report.json", "simulation_report.json"]:
    if (RESULTS_DIR / name).exists():
        report_file = RESULTS_DIR / name
        break

if report_file is None:
    print("❌ No report file found")
    sys.exit(1)

with open(report_file, 'r') as f:
    report = json.load(f)

# Try different log file names
log_file = None
for name in ["enhanced_simulation.log", "simulation.log"]:
    if (RESULTS_DIR / name).exists():
        log_file = RESULTS_DIR / name
        break

log_data = []
if log_file:
    with open(log_file, 'r') as f:
        for line in f:
            if "Step" in line and "Energy" in line:
                # Parse flexibly using regex
                data = {}

                # Extract step
                step_match = re.search(r'Step\s+(\d+)', line)
                if step_match:
                    data['step'] = int(step_match.group(1))

                # Extract t=
                time_match = re.search(r't=([0-9.]+)', line)
                if time_match:
                    data['time'] = float(time_match.group(1))

                # Extract Energy=
                energy_match = re.search(r'Energy=([0-9.]+)', line)
                if energy_match:
                    data['energy'] = float(energy_match.group(1))

                # Extract Vertices=
                vertices_match = re.search(r'Vertices=(\d+)', line)
                if vertices_match:
                    data['vertices'] = int(vertices_match.group(1))

                # Extract Structures=
                structures_match = re.search(r'Structures=(\d+)', line)
                if structures_match:
                    data['structures'] = int(structures_match.group(1))

                # Extract Packets=
                packets_match = re.search(r'Packets=(\d+)', line)
                if packets_match:
                    data['packets'] = int(packets_match.group(1))

                # Extract Queue= (new field)
                queue_match = re.search(r'Queue=(\d+)', line)
                if queue_match:
                    data['queue'] = int(queue_match.group(1))

                # Extract Materials= (new field)
                materials_match = re.search(r'Materials=(\d+)', line)
                if materials_match:
                    data['materials'] = int(materials_match.group(1))

                # Only add if we have the essential fields
                if 'step' in data and 'energy' in data:
                    log_data.append(data)

if log_data:
    df = pd.DataFrame(log_data)
else:
    # Create minimal dataframe from report
    print("⚠️ No log data found, using report statistics only")
    df = pd.DataFrame([{
        'step': 0, 'time': 0,
        'energy': report.get('final_statistics', {}).get('total_energy', 30),
        'vertices': report.get('final_statistics', {}).get('n_vertices', 0),
        'structures': report.get('final_statistics', {}).get('n_structures_built', 0),
        'packets': report.get('final_statistics', {}).get('n_packets_transported', 0)
    }])

config = report.get('config', {})
stats = report.get('final_statistics', {})
n_agents = report.get('n_agents_alive', {'beavers': 0, 'ants': 0, 'bees': 0})

print(f"✅ Loaded {len(df)} timesteps")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("\n📊 STATISTICAL ANALYSIS")
print("-" * 80)

# Energy analysis
initial_energy = df['energy'].iloc[0]
final_energy = df['energy'].iloc[-1]
total_decay = initial_energy - final_energy
decay_rate = total_decay / df['time'].iloc[-1]

print(f"\n⚡ Energy Dynamics:")
print(f"  Initial Energy: {initial_energy:.2f}")
print(f"  Final Energy: {final_energy:.2f}")
print(f"  Total Decay: {total_decay:.2f} ({total_decay/initial_energy*100:.1f}%)")
print(f"  Mean Decay Rate: {decay_rate:.4f} energy/time")

# Fit models
times = df['time'].values
energies = df['energy'].values

# Linear model
linear_coeffs = np.polyfit(times, energies, 1)
linear_pred = np.polyval(linear_coeffs, times)
linear_r2 = 1 - np.sum((energies - linear_pred)**2) / np.sum((energies - np.mean(energies))**2)

# Exponential model
def exp_decay(t, E0, k):
    return E0 * np.exp(-k * t)

try:
    exp_params, _ = curve_fit(exp_decay, times, energies, p0=[initial_energy, 0.001])
    exp_pred = exp_decay(times, *exp_params)
    exp_r2 = 1 - np.sum((energies - exp_pred)**2) / np.sum((energies - np.mean(energies))**2)
except (RuntimeError, ValueError, TypeError) as e:
    # Curve fit failed (convergence issues or invalid data)
    exp_params = [initial_energy, 0.001]
    exp_r2 = 0

# Power law model
def power_law(t, a, b, c):
    return a * (t + 1)**b + c

try:
    power_params, _ = curve_fit(power_law, times, energies, p0=[initial_energy, -0.5, 0])
    power_pred = power_law(times, *power_params)
    power_r2 = 1 - np.sum((energies - power_pred)**2) / np.sum((energies - np.mean(energies))**2)
except (RuntimeError, ValueError, TypeError) as e:
    # Curve fit failed (convergence issues or invalid data)
    power_params = [initial_energy, -0.5, 0]
    power_r2 = 0

print(f"\n  Decay Models:")
print(f"    Linear: E(t) = {linear_coeffs[1]:.2f} + {linear_coeffs[0]:.4f}*t (R²={linear_r2:.4f})")
print(f"    Exponential: E(t) = {exp_params[0]:.2f} * exp(-{exp_params[1]:.4f}*t) (R²={exp_r2:.4f})")
print(f"    Power Law: E(t) = {power_params[0]:.2f}*(t+1)^{power_params[1]:.3f} + {power_params[2]:.2f} (R²={power_r2:.4f})")
print(f"    Best Fit: {'Linear' if linear_r2 > max(exp_r2, power_r2) else 'Exponential' if exp_r2 > power_r2 else 'Power Law'}")

# Construction analysis
print(f"\n🏗️  Beaver Construction Analysis:")
print(f"  Total Structures: {stats['n_structures_built']:,}")
print(f"  Structures per Beaver: {stats['n_structures_built']/config['n_beavers']:.1f}")
print(f"  Construction Rate: {stats['n_structures_built']/config['t_max']:.2f} structures/time")
print(f"  Energy per Structure: {total_decay/stats['n_structures_built']:.3f}")

structures_ts = df['structures'].values
structure_growth = np.gradient(structures_ts, df['time'].values)
peak_construction_idx = np.argmax(structure_growth)
peak_construction_time = df.loc[peak_construction_idx, 'time']

print(f"  Mean Construction Rate: {np.mean(structure_growth):.2f} structures/time")
print(f"  Peak Construction: {structure_growth[peak_construction_idx]:.2f} structures/time at t={peak_construction_time:.1f}")

# Semantic graph analysis
print(f"\n🕸️  Semantic Graph Analysis:")
print(f"  Total Vertices: {stats['n_vertices']:,}")
print(f"  Total Edges: {stats['n_edges']:,}")
print(f"  Vertices per Ant: {stats['n_vertices']/config['n_ants']:.1f}")

if stats['n_vertices'] > 1:
    edge_density = stats['n_edges'] / (stats['n_vertices'] * (stats['n_vertices'] - 1))
    avg_degree = 2 * stats['n_edges'] / stats['n_vertices']
    print(f"  Edge Density: {edge_density:.6f}")
    print(f"  Average Degree: {avg_degree:.2f}")

vertices_ts = df['vertices'].values
vertex_growth = np.gradient(vertices_ts, df['time'].values)
peak_growth_idx = np.argmax(vertex_growth)
peak_growth_time = df.loc[peak_growth_idx, 'time']

print(f"  Mean Growth Rate: {np.mean(vertex_growth):.2f} vertices/time")
print(f"  Peak Growth: {vertex_growth[peak_growth_idx]:.0f} vertices/time at t={peak_growth_time:.1f}")

# Detect saturation
saturation_threshold = 0.1 * np.max(vertex_growth)
saturation_indices = np.where(vertex_growth < saturation_threshold)[0]
if len(saturation_indices) > 0 and saturation_indices[-1] > len(vertices_ts) * 0.8:
    saturation_time = df.loc[saturation_indices[0], 'time']
    print(f"  Saturation Detected: t={saturation_time:.1f} ({saturation_time/config['t_max']*100:.0f}% through simulation)")
else:
    print(f"  Saturation: Not detected (still growing)")

# Transport analysis
print(f"\n🐝 Packet Transport Analysis:")
print(f"  Total Packets: {stats['n_packets_transported']}")
print(f"  Packets per Bee: {stats['n_packets_transported']/config['n_bees']:.2f}")
print(f"  Transport Rate: {stats['n_packets_transported']/config['t_max']:.2f} packets/time")

first_packet_df = df[df['packets'] > 0]
if len(first_packet_df) > 0:
    first_packet_time = first_packet_df.iloc[0]['time']
    first_packet_vertices = first_packet_df.iloc[0]['vertices']
    print(f"  First Transport: t={first_packet_time:.1f} (step {first_packet_df.iloc[0]['step']})")
    print(f"  Vertices at First Transport: {first_packet_vertices}")
    print(f"  Graph Maturity Required: {first_packet_vertices} vertices before transport begins")

# Survival analysis
print(f"\n🏥 Colony Survival Analysis:")
for colony in ['beavers', 'ants', 'bees']:
    initial = config[f'n_{colony}']
    alive = n_agents[colony]
    died = initial - alive
    survival_rate = alive / initial * 100
    mortality_rate = died / initial * 100
    
    print(f"  {colony.capitalize()}:")
    print(f"    Alive: {alive}/{initial} ({survival_rate:.1f}%)")
    print(f"    Died: {died} ({mortality_rate:.1f}%)")
    print(f"    Energy per Agent: {final_energy/alive:.3f}")

# Work output metrics (NOT efficiency - that would require a baseline)
print(f"\n📈 System Work Output:")
total_work = stats['n_structures_built'] + stats['n_packets_transported']
energy_per_work = total_decay / max(total_work, 1)
work_per_energy = total_work / max(total_decay, 0.01)

# Theoretical costs from energy model: BUILD_COST=0.06, net cost ~0.04 per structure
theoretical_build_cost = 0.04  # Net cost per structure
theoretical_max_structures = total_decay / theoretical_build_cost
build_utilization = stats['n_structures_built'] / max(theoretical_max_structures, 1)

print(f"  Total Work Units: {total_work:,} (structures + packets)")
print(f"  Energy Spent: {total_decay:.1f} units")
print(f"  Work per Energy: {work_per_energy:.2f} units/energy")
print(f"  Energy per Work: {energy_per_work:.3f} energy/unit")
print(f"  Build Utilization: {build_utilization*100:.1f}% of theoretical max structures")

# Information processing rate
info_vertices = stats['n_vertices']
info_edges = stats['n_edges']
total_info = info_vertices + info_edges
info_rate = total_info / config['t_max']

print(f"\n  Information Processing:")
print(f"    Total Info Elements: {total_info:,} (vertices + edges)")
print(f"    Processing Rate: {info_rate:.1f} elements/time")
print(f"    Energy per Info: {total_decay/total_info:.4f}")

# ============================================================================
# VISUALIZATION 1: SYSTEM DYNAMICS (6 panels)
# ============================================================================

print(f"\n🎨 Generating Visualization 1: System Dynamics...")

fig1 = plt.figure(figsize=(24, 16))
gs1 = GridSpec(3, 2, figure=fig1, hspace=0.3, wspace=0.25)

# Panel 1: Energy with model fits
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.plot(df['time'], df['energy'], 'o-', color=COLORS['energy'], linewidth=2, 
         markersize=3, label='Actual', alpha=0.7)
ax1.plot(times, linear_pred, '--', color='red', linewidth=2, 
         label=f'Linear (R²={linear_r2:.3f})')
ax1.plot(times, exp_pred, '--', color='green', linewidth=2, 
         label=f'Exp (R²={exp_r2:.3f})')
if power_r2 > 0:
    ax1.plot(times, power_pred, '--', color='orange', linewidth=2, 
             label=f'Power (R²={power_r2:.3f})')
ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Energy', fontsize=12, fontweight='bold')
ax1.set_title('Energy Evolution with Decay Models', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Construction activity
ax2 = fig1.add_subplot(gs1[0, 1])
ax2.plot(df['time'], df['structures'], 'o-', color=COLORS['structures'], 
         linewidth=2, markersize=3, alpha=0.7, label='Cumulative')
ax2_twin = ax2.twinx()
ax2_twin.plot(df['time'], structure_growth, '-', color='darkred', 
              linewidth=1.5, alpha=0.5, label='Rate')
ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Structures', fontsize=12, fontweight='bold', color=COLORS['structures'])
ax2_twin.set_ylabel('Construction Rate', fontsize=12, fontweight='bold', color='darkred')
ax2.set_title('Beaver Construction Activity', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=10)
ax2_twin.legend(loc='upper right', fontsize=10)

# Panel 3: Semantic graph growth
ax3 = fig1.add_subplot(gs1[1, 0])
ax3.plot(df['time'], df['vertices'], 'o-', color=COLORS['vertices'], 
         linewidth=2, markersize=3, alpha=0.7, label='Vertices')
ax3_twin = ax3.twinx()
ax3_twin.plot(df['time'], vertex_growth, '-', color='darkblue', 
              linewidth=1.5, alpha=0.5, label='Growth Rate')
ax3.axhline(y=first_packet_vertices if len(first_packet_df) > 0 else 0, 
            color='red', linestyle='--', linewidth=2, 
            label=f'Transport Threshold ({first_packet_vertices})' if len(first_packet_df) > 0 else '')
ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
ax3.set_ylabel('Semantic Vertices', fontsize=12, fontweight='bold', color=COLORS['vertices'])
ax3_twin.set_ylabel('Vertex Growth Rate', fontsize=12, fontweight='bold', color='darkblue')
ax3.set_title('Semantic Graph Emergence', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=10)
ax3_twin.legend(loc='upper right', fontsize=10)

# Panel 4: Packet transport
ax4 = fig1.add_subplot(gs1[1, 1])
ax4.plot(df['time'], df['packets'], 'o-', color=COLORS['packets'], 
         linewidth=2, markersize=3, alpha=0.7)
if len(first_packet_df) > 0:
    ax4.axvline(x=first_packet_time, color='green', linestyle='--', linewidth=2, 
                label=f'First Transport (t={first_packet_time:.1f})')
ax4.set_xlabel('Time', fontsize=12, fontweight='bold')
ax4.set_ylabel('Cumulative Packets Transported', fontsize=12, fontweight='bold')
ax4.set_title('Bee Packet Transport', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
if len(first_packet_df) > 0:
    ax4.legend(loc='best', fontsize=10)

# Panel 5: Multi-metric comparison (normalized)
ax5 = fig1.add_subplot(gs1[2, 0])
ax5.plot(df['time'], df['energy']/initial_energy, '-', linewidth=2, 
         label='Energy (normalized)', color=COLORS['energy'])
ax5.plot(df['time'], df['structures']/stats['n_structures_built'], '-', linewidth=2, 
         label='Structures (normalized)', color=COLORS['structures'])
ax5.plot(df['time'], df['vertices']/stats['n_vertices'], '-', linewidth=2, 
         label='Vertices (normalized)', color=COLORS['vertices'])
if stats['n_packets_transported'] > 0:
    ax5.plot(df['time'], df['packets']/stats['n_packets_transported'], '-', linewidth=2, 
             label='Packets (normalized)', color=COLORS['packets'])
ax5.set_xlabel('Time', fontsize=12, fontweight='bold')
ax5.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
ax5.set_title('Normalized System Metrics', fontsize=14, fontweight='bold')
ax5.legend(loc='best', fontsize=10)
ax5.grid(True, alpha=0.3)

# Panel 6: Phase space (Energy vs Work)
ax6 = fig1.add_subplot(gs1[2, 1])
work_ts = df['structures'] + df['packets']
scatter = ax6.scatter(work_ts, df['energy'], c=df['time'], cmap='viridis', 
                      s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
ax6.set_xlabel('Total Work (Structures + Packets)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Energy', fontsize=12, fontweight='bold')
ax6.set_title('Phase Space: Energy vs Work', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Time', fontsize=10, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.suptitle('BLACKHOLE ARCHIVE: System Dynamics Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig(VIZ_DIR / 'system_dynamics.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✅ Saved: system_dynamics.png")

# ============================================================================
# VISUALIZATION 2: COLONY ANALYSIS (4 panels)
# ============================================================================

print(f"\n🎨 Generating Visualization 2: Colony Analysis...")

fig2 = plt.figure(figsize=(20, 12))
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)

# Panel 1: Survival rates
ax1 = fig2.add_subplot(gs2[0, 0])
colonies = ['Beavers', 'Ants', 'Bees']
survival_rates = [
    n_agents['beavers'] / config['n_beavers'] * 100,
    n_agents['ants'] / config['n_ants'] * 100,
    n_agents['bees'] / config['n_bees'] * 100
]
colors_list = [COLORS['beavers'], COLORS['ants'], COLORS['bees']]
bars = ax1.bar(colonies, survival_rates, color=colors_list, alpha=0.7, 
               edgecolor='black', linewidth=2)
ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% Threshold')
ax1.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Colony Survival Rates', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 105])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, rate in zip(bars, survival_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 2: Productivity metrics
ax2 = fig2.add_subplot(gs2[0, 1])
metrics = {
    'Structures\nBuilt': stats['n_structures_built'],
    'Semantic\nVertices': stats['n_vertices'],
    'Semantic\nEdges': stats['n_edges'],
    'Packets\nTransported': stats['n_packets_transported']
}
ax2.bar(metrics.keys(), metrics.values(), color='#A23B72', alpha=0.7, 
        edgecolor='black', linewidth=2)
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('System Productivity Metrics', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_yscale('log')

# Panel 3: Per-agent productivity
ax3 = fig2.add_subplot(gs2[1, 0])
per_agent = {
    'Structures\nper Beaver': stats['n_structures_built'] / config['n_beavers'],
    'Vertices\nper Ant': stats['n_vertices'] / config['n_ants'],
    'Packets\nper Bee': stats['n_packets_transported'] / config['n_bees']
}
colors_per_agent = [COLORS['beavers'], COLORS['ants'], COLORS['bees']]
ax3.bar(per_agent.keys(), per_agent.values(), color=colors_per_agent, 
        alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Output per Agent', fontsize=12, fontweight='bold')
ax3.set_title('Per-Agent Productivity', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Energy distribution
ax4 = fig2.add_subplot(gs2[1, 1])
energy_dist = {
    'Beavers': final_energy * (n_agents['beavers'] / (n_agents['beavers'] + n_agents['ants'] + n_agents['bees'])),
    'Ants': final_energy * (n_agents['ants'] / (n_agents['beavers'] + n_agents['ants'] + n_agents['bees'])),
    'Bees': final_energy * (n_agents['bees'] / (n_agents['beavers'] + n_agents['ants'] + n_agents['bees']))
}
ax4.pie(energy_dist.values(), labels=energy_dist.keys(), autopct='%1.1f%%',
        colors=colors_list, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax4.set_title('Final Energy Distribution by Colony', fontsize=14, fontweight='bold')

plt.suptitle('BLACKHOLE ARCHIVE: Colony Performance Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig(VIZ_DIR / 'colony_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✅ Saved: colony_analysis.png")

# ============================================================================
# VISUALIZATION 3: GROWTH DYNAMICS (6 panels)
# ============================================================================

print(f"\n🎨 Generating Visualization 3: Growth Dynamics...")

fig3 = plt.figure(figsize=(24, 16))
gs3 = GridSpec(3, 2, figure=fig3, hspace=0.3, wspace=0.25)

# Panel 1: Energy decay rate
ax1 = fig3.add_subplot(gs3[0, 0])
energy_decay_rate = -np.gradient(df['energy'].values, df['time'].values)
ax1.plot(df['time'], energy_decay_rate, '-', color=COLORS['energy'], linewidth=2)
ax1.axhline(y=np.mean(energy_decay_rate), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(energy_decay_rate):.3f}')
ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
ax1.set_ylabel('Energy Decay Rate', fontsize=12, fontweight='bold')
ax1.set_title('Instantaneous Energy Decay Rate', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Structure construction rate
ax2 = fig3.add_subplot(gs3[0, 1])
ax2.plot(df['time'], structure_growth, '-', color=COLORS['structures'], linewidth=2)
ax2.axhline(y=np.mean(structure_growth), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(structure_growth):.2f}')
ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
ax2.set_ylabel('Construction Rate', fontsize=12, fontweight='bold')
ax2.set_title('Beaver Construction Rate', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Panel 3: Vertex growth rate
ax3 = fig3.add_subplot(gs3[1, 0])
ax3.plot(df['time'], vertex_growth, '-', color=COLORS['vertices'], linewidth=2)
ax3.axhline(y=np.mean(vertex_growth), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(vertex_growth):.2f}')
ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
ax3.set_ylabel('Vertex Growth Rate', fontsize=12, fontweight='bold')
ax3.set_title('Semantic Graph Growth Rate', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Panel 4: Packet transport rate
ax4 = fig3.add_subplot(gs3[1, 1])
packet_growth = np.gradient(df['packets'].values, df['time'].values)
ax4.plot(df['time'], packet_growth, '-', color=COLORS['packets'], linewidth=2)
ax4.axhline(y=np.mean(packet_growth), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(packet_growth):.3f}')
ax4.set_xlabel('Time', fontsize=12, fontweight='bold')
ax4.set_ylabel('Transport Rate', fontsize=12, fontweight='bold')
ax4.set_title('Bee Packet Transport Rate', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Panel 5: Cumulative work vs time
ax5 = fig3.add_subplot(gs3[2, 0])
cumulative_work = df['structures'] + df['packets']
ax5.fill_between(df['time'], 0, df['structures'], alpha=0.5, 
                 color=COLORS['structures'], label='Structures')
ax5.fill_between(df['time'], df['structures'], cumulative_work, alpha=0.5, 
                 color=COLORS['packets'], label='Packets')
ax5.plot(df['time'], cumulative_work, 'k-', linewidth=2, label='Total Work')
ax5.set_xlabel('Time', fontsize=12, fontweight='bold')
ax5.set_ylabel('Cumulative Work', fontsize=12, fontweight='bold')
ax5.set_title('Cumulative Work Decomposition', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Panel 6: Efficiency over time
ax6 = fig3.add_subplot(gs3[2, 1])
work_rate = np.gradient(cumulative_work.values, df['time'].values)
efficiency = work_rate / (energy_decay_rate + 1e-10)
# Smooth efficiency
from scipy.ndimage import gaussian_filter1d
efficiency_smooth = gaussian_filter1d(efficiency, sigma=10)
ax6.plot(df['time'], efficiency_smooth, '-', color='purple', linewidth=2)
ax6.axhline(y=np.mean(efficiency_smooth), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(efficiency_smooth):.3f}')
ax6.set_xlabel('Time', fontsize=12, fontweight='bold')
ax6.set_ylabel('Work Efficiency (work/energy)', fontsize=12, fontweight='bold')
ax6.set_title('System Efficiency Over Time', fontsize=14, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.suptitle('BLACKHOLE ARCHIVE: Growth Rate Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig(VIZ_DIR / 'growth_dynamics.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✅ Saved: growth_dynamics.png")

# ============================================================================
# VISUALIZATION 4: CORRELATION ANALYSIS
# ============================================================================

print(f"\n🎨 Generating Visualization 4: Correlation Analysis...")

fig4 = plt.figure(figsize=(20, 16))
gs4 = GridSpec(3, 2, figure=fig4, hspace=0.3, wspace=0.3)

# Correlation matrix
metrics_df = df[['energy', 'structures', 'vertices', 'packets']].copy()
correlation_matrix = metrics_df.corr()

ax1 = fig4.add_subplot(gs4[0, :])
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            ax=ax1, vmin=-1, vmax=1)
ax1.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold')

# Scatter plots with regression
# Energy vs Structures
ax2 = fig4.add_subplot(gs4[1, 0])
ax2.scatter(df['structures'], df['energy'], c=df['time'], cmap='viridis', 
            s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
z = np.polyfit(df['structures'], df['energy'], 1)
p = np.poly1d(z)
ax2.plot(df['structures'], p(df['structures']), "r--", linewidth=2, 
         label=f'Fit: y={z[0]:.3f}x+{z[1]:.1f}')
ax2.set_xlabel('Structures Built', fontsize=12, fontweight='bold')
ax2.set_ylabel('Energy', fontsize=12, fontweight='bold')
ax2.set_title('Energy vs Structures', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Vertices vs Packets
ax3 = fig4.add_subplot(gs4[1, 1])
ax3.scatter(df['vertices'], df['packets'], c=df['time'], cmap='viridis', 
            s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
# Only fit where packets > 0
if len(first_packet_df) > 0:
    packets_data = df[df['packets'] > 0]
    z = np.polyfit(packets_data['vertices'], packets_data['packets'], 1)
    p = np.poly1d(z)
    ax3.plot(packets_data['vertices'], p(packets_data['vertices']), "r--", 
             linewidth=2, label=f'Fit: y={z[0]:.4f}x+{z[1]:.1f}')
ax3.set_xlabel('Semantic Vertices', fontsize=12, fontweight='bold')
ax3.set_ylabel('Packets Transported', fontsize=12, fontweight='bold')
ax3.set_title('Packets vs Vertices (Graph Dependency)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Work vs Energy
ax4 = fig4.add_subplot(gs4[2, 0])
work = df['structures'] + df['packets']
ax4.scatter(work, df['energy'], c=df['time'], cmap='plasma', 
            s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
z = np.polyfit(work, df['energy'], 1)
p = np.poly1d(z)
ax4.plot(work, p(work), "r--", linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.1f}')
ax4.set_xlabel('Total Work (Structures + Packets)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Energy', fontsize=12, fontweight='bold')
ax4.set_title('Energy vs Total Work', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Structures vs Vertices
ax5 = fig4.add_subplot(gs4[2, 1])
ax5.scatter(df['structures'], df['vertices'], c=df['time'], cmap='cividis', 
            s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
z = np.polyfit(df['structures'], df['vertices'], 1)
p = np.poly1d(z)
ax5.plot(df['structures'], p(df['structures']), "r--", linewidth=2, 
         label=f'Fit: y={z[0]:.3f}x+{z[1]:.1f}')
ax5.set_xlabel('Structures Built', fontsize=12, fontweight='bold')
ax5.set_ylabel('Semantic Vertices', fontsize=12, fontweight='bold')
ax5.set_title('Vertices vs Structures (Infrastructure Correlation)', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

plt.suptitle('BLACKHOLE ARCHIVE: Correlation & Regression Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig(VIZ_DIR / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✅ Saved: correlation_analysis.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print(f"\n📝 Generating Summary Report...")

report_text = f"""
{'='*80}
BLACKHOLE ARCHIVE SIMULATION: COMPREHENSIVE ANALYSIS REPORT
{'='*80}

CONFIGURATION
{'-'*80}
Black Hole Mass: {config.get('black_hole_mass', 1.0)} M☉
Event Horizon: r_s = {2*config.get('black_hole_mass', 1.0):.2f}
Simulation Domain: r ∈ [{config.get('r_min', 2.5)}, {config.get('r_max', 50.0)}]
Grid Resolution: {config.get('n_r', 128)}×{config.get('n_theta', 64)}×{config.get('n_phi', 64)} = {config.get('n_r', 128)*config.get('n_theta', 64)*config.get('n_phi', 64):,} points
Duration: {config.get('t_max', 100)} time units
Timestep: dt = {config.get('dt', 0.01)}
Total Steps: {len(df):,}

Initial Agent Populations:
  - Beavers: {config.get('n_beavers', 50)}
  - Ants: {config.get('n_ants', 200)}
  - Bees: {config.get('n_bees', 100)}
  - Total: {config.get('n_beavers', 50) + config.get('n_ants', 200) + config.get('n_bees', 100)}

ENERGY DYNAMICS
{'-'*80}
Initial Energy: {initial_energy:.2f}
Final Energy: {final_energy:.2f}
Total Energy Consumed: {total_decay:.2f} ({total_decay/initial_energy*100:.1f}%)
Mean Decay Rate: {decay_rate:.4f} energy/time

Decay Model Comparison (R² values):
  - Linear: {linear_r2:.4f} {'✓ BEST FIT' if linear_r2 > max(exp_r2, power_r2) else ''}
  - Exponential: {exp_r2:.4f} {'✓ BEST FIT' if exp_r2 > max(linear_r2, power_r2) else ''}
  - Power Law: {power_r2:.4f} {'✓ BEST FIT' if power_r2 > max(linear_r2, exp_r2) else ''}

BEAVER CONSTRUCTION
{'-'*80}
Total Structures Built: {stats['n_structures_built']:,}
Structures per Beaver: {stats['n_structures_built']/config.get('n_beavers', 50):.2f}
Construction Rate: {stats['n_structures_built']/config.get('t_max', 100):.2f} structures/time
Peak Construction Rate: {structure_growth[peak_construction_idx]:.2f} structures/time at t={peak_construction_time:.1f}
Energy per Structure: {total_decay/stats['n_structures_built']:.3f}

SEMANTIC GRAPH
{'-'*80}
Total Vertices: {stats['n_vertices']:,}
Total Edges: {stats['n_edges']:,}
Vertices per Ant: {stats['n_vertices']/config.get('n_ants', 200):.2f}
Edge Density: {(stats['n_edges']/(stats['n_vertices']*(stats['n_vertices']-1)) if stats['n_vertices'] > 1 else 0):.6f}
Average Degree: {(2*stats['n_edges']/stats['n_vertices'] if stats['n_vertices'] > 0 else 0):.2f}

Vertex Growth:
  - Mean Rate: {np.mean(vertex_growth):.2f} vertices/time
  - Peak Rate: {vertex_growth[peak_growth_idx]:.0f} vertices/time at t={peak_growth_time:.1f}

PACKET TRANSPORT
{'-'*80}
Total Packets Transported: {stats['n_packets_transported']}
Packets per Bee: {stats['n_packets_transported']/config.get('n_bees', 100):.2f}
Transport Rate: {stats['n_packets_transported']/config.get('t_max', 100):.2f} packets/time
First Transport: t={first_packet_time:.1f} ({first_packet_vertices} vertices required)

COLONY SURVIVAL
{'-'*80}
Beavers: {n_agents['beavers']}/{config.get('n_beavers', 50)} alive ({n_agents['beavers']/config.get('n_beavers', 50)*100:.1f}%), {config.get('n_beavers', 50)-n_agents['beavers']} died
Ants: {n_agents['ants']}/{config.get('n_ants', 200)} alive ({n_agents['ants']/config.get('n_ants', 200)*100:.1f}%), {config.get('n_ants', 200)-n_agents['ants']} died
Bees: {n_agents['bees']}/{config.get('n_bees', 100)} alive ({n_agents['bees']/config.get('n_bees', 100)*100:.1f}%), {config.get('n_bees', 100)-n_agents['bees']} died

WORK OUTPUT METRICS
{'-'*80}
Total Work: {total_work:,} units (structures + packets)
Energy Spent: {total_decay:.1f} units
Work per Energy: {work_per_energy:.2f} units/energy
Energy per Work: {energy_per_work:.3f} energy/unit
Build Utilization: {build_utilization*100:.1f}% of theoretical max structures

Information Elements: {total_info:,} (vertices + edges)
Information Processing Rate: {info_rate:.1f} elements/time
Energy per Information Element: {total_decay/max(total_info, 1):.4f}

KEY FINDINGS
{'-'*80}
✓ Energy decay follows {'LINEAR' if linear_r2 > max(exp_r2, power_r2) else 'EXPONENTIAL' if exp_r2 > power_r2 else 'POWER LAW'} model (R²={max(linear_r2, exp_r2, power_r2):.4f})
✓ Beaver construction: {stats['n_structures_built']:,} structures with peak rate {structure_growth[peak_construction_idx]:.1f}/time
✓ Semantic graph emerged: {stats['n_vertices']:,} vertices, {stats['n_edges']:,} edges
✓ Transport initiated at t={first_packet_time:.1f} after {first_packet_vertices} vertices formed
✓ Build utilization: {build_utilization*100:.1f}% (structures built / theoretical max)

VISUALIZATIONS GENERATED
{'-'*80}
1. system_dynamics.png - Energy, construction, graph growth, transport
2. colony_analysis.png - Survival rates, productivity, energy distribution
3. growth_dynamics.png - Instantaneous rates and efficiency metrics
4. correlation_analysis.png - Cross-metric correlations and regressions

{'='*80}
Report Generated: {pd.Timestamp.now()}
{'='*80}
"""

report_path = VIZ_DIR / 'analysis_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"  ✅ Saved: analysis_report.txt")

# ============================================================================
# COMPLETION
# ============================================================================

print(f"\n{'='*80}")
print(f"✅ COMPLETE ANALYSIS FINISHED")
print(f"{'='*80}")
print(f"\n📁 All outputs saved to: {VIZ_DIR}")
print(f"\n📊 Generated Files:")
print(f"  1. system_dynamics.png - 6-panel system overview")
print(f"  2. colony_analysis.png - Colony performance metrics")
print(f"  3. growth_dynamics.png - Growth rates and efficiency")
print(f"  4. correlation_analysis.png - Cross-metric relationships")
print(f"  5. analysis_report.txt - Complete text summary")

print(f"\n🎯 Key Results:")
print(f"  • Energy Decay: {total_decay:.1f} ({total_decay/initial_energy*100:.0f}%)")
print(f"  • Structures: {stats['n_structures_built']:,}")
print(f"  • Graph: {stats['n_vertices']:,} vertices, {stats['n_edges']:,} edges")
print(f"  • Transport: {stats['n_packets_transported']} packets")
print(f"  • Build Utilization: {build_utilization*100:.1f}%")

print(f"\n{'='*80}\n")
