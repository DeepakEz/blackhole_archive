# Blackhole Archive Simulation Analysis
# Comprehensive analysis of simulation results

"""
ANALYSIS SUITE:

This script analyzes simulation results to extract insights about:
1. System efficiency and performance
2. Colony behavior patterns
3. Information processing capabilities
4. Comparison with theoretical predictions
5. Recommendations for parameter tuning
"""

import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

class SimulationAnalyzer:
    """Comprehensive simulation analysis"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
        # Load data
        self.report = self._load_report()
        self.log_data = self._parse_log()
        
        print(f"📊 Analysis initialized for: {results_dir}")
    
    def _load_report(self) -> Dict:
        """Load simulation report"""
        report_path = self.results_dir / "simulation_report.json"
        if report_path.exists():
            with open(report_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _parse_log(self) -> List[Dict]:
        """Parse simulation log"""
        log_path = self.results_dir / "simulation.log"
        if not log_path.exists():
            return []
        
        log_data = []
        with open(log_path, 'r') as f:
            for line in f:
                if "Step" in line and "Energy" in line:
                    # Extract step, time, energy
                    parts = line.split(',')
                    
                    step_str = [p for p in parts if 'Step' in p][0]
                    step = int(step_str.split('/')[0].split()[-1])
                    
                    time_str = [p for p in parts if 't=' in p][0]
                    time = float(time_str.split('=')[1])
                    
                    energy_str = [p for p in parts if 'Energy=' in p][0]
                    energy = float(energy_str.split('=')[1])
                    
                    log_data.append({
                        'step': step,
                        'time': time,
                        'energy': energy
                    })
        
        return log_data
    
    def analyze_all(self):
        """Run complete analysis suite"""
        print("\n" + "="*80)
        print("🔬 BLACKHOLE ARCHIVE SIMULATION ANALYSIS")
        print("="*80)
        
        self.analyze_configuration()
        self.analyze_energy_dynamics()
        self.analyze_colony_survival()
        self.analyze_efficiency()
        self.identify_issues()
        self.provide_recommendations()
        
        print("\n" + "="*80)
        print("✅ Analysis Complete")
        print("="*80)
    
    def analyze_configuration(self):
        """Analyze simulation configuration"""
        print("\n📋 CONFIGURATION ANALYSIS")
        print("-" * 80)
        
        config = self.report.get('config', {})
        
        print(f"Black Hole Mass: {config.get('black_hole_mass', 'N/A')} M☉")
        print(f"Event Horizon: {2 * config.get('black_hole_mass', 1.0):.2f} (Schwarzschild radii)")
        print(f"Simulation Domain: r ∈ [{config.get('r_min', 'N/A')}, {config.get('r_max', 'N/A')}]")
        print(f"Grid Resolution: {config.get('n_r', 'N/A')} × {config.get('n_theta', 'N/A')} × {config.get('n_phi', 'N/A')}")
        print(f"\nTime Integration:")
        print(f"  Duration: {config.get('t_max', 'N/A')} time units")
        print(f"  Timestep: {config.get('dt', 'N/A')}")
        print(f"  Total Steps: {int(config.get('t_max', 0) / config.get('dt', 1))}")
        print(f"\nAgent Populations:")
        print(f"  Beavers: {config.get('n_beavers', 'N/A')}")
        print(f"  Ants: {config.get('n_ants', 'N/A')}")
        print(f"  Bees: {config.get('n_bees', 'N/A')}")
        print(f"  Total: {config.get('n_beavers', 0) + config.get('n_ants', 0) + config.get('n_bees', 0)}")
    
    def analyze_energy_dynamics(self):
        """Analyze energy evolution"""
        print("\n⚡ ENERGY DYNAMICS ANALYSIS")
        print("-" * 80)
        
        if not self.log_data:
            print("❌ No log data available")
            return
        
        df = pd.DataFrame(self.log_data)
        
        # Energy statistics
        initial_energy = df['energy'].iloc[0]
        final_energy = df['energy'].iloc[-1]
        total_decay = initial_energy - final_energy
        decay_rate = total_decay / df['time'].iloc[-1]
        
        print(f"Initial Energy: {initial_energy:.2f}")
        print(f"Final Energy: {final_energy:.2f}")
        print(f"Total Decay: {total_decay:.2f} ({total_decay/initial_energy*100:.1f}%)")
        print(f"Average Decay Rate: {decay_rate:.3f} energy/time")
        
        # Fit decay model
        # E(t) = E_0 - k*t (linear)
        # E(t) = E_0 * exp(-k*t) (exponential)
        
        times = df['time'].values
        energies = df['energy'].values
        
        # Linear fit
        k_linear = np.polyfit(times, energies, 1)[0]
        
        # Exponential fit
        log_energies = np.log(energies + 1e-10)
        k_exp = -np.polyfit(times, log_energies, 1)[0]
        
        print(f"\nDecay Model Analysis:")
        print(f"  Linear Model: E(t) = {initial_energy:.2f} - {-k_linear:.3f}*t")
        print(f"  Exponential Model: E(t) = {initial_energy:.2f} * exp(-{k_exp:.4f}*t)")
        
        # Which fits better?
        linear_residuals = np.sum((energies - (initial_energy + k_linear * times))**2)
        exp_residuals = np.sum((energies - initial_energy * np.exp(-k_exp * times))**2)
        
        if linear_residuals < exp_residuals:
            print(f"  Best Fit: Linear (R² residuals: {linear_residuals:.2f})")
        else:
            print(f"  Best Fit: Exponential (R² residuals: {exp_residuals:.2f})")
        
        # Energy per agent
        n_agents = self.report['config']['n_beavers'] + self.report['config']['n_ants'] + self.report['config']['n_bees']
        energy_per_agent = final_energy / n_agents
        
        print(f"\nPer-Agent Metrics:")
        print(f"  Final Energy/Agent: {energy_per_agent:.4f}")
        print(f"  Energy Consumed/Agent: {total_decay/n_agents:.4f}")
    
    def analyze_colony_survival(self):
        """Analyze colony survival patterns"""
        print("\n🏥 COLONY SURVIVAL ANALYSIS")
        print("-" * 80)
        
        n_agents = self.report.get('n_agents', {})
        config = self.report.get('config', {})
        
        colonies = ['beavers', 'ants', 'bees']
        
        for colony in colonies:
            initial = config.get(f'n_{colony}', 0)
            surviving = n_agents.get(colony, 0)
            mortality = initial - surviving
            survival_rate = surviving / initial if initial > 0 else 0
            
            print(f"\n{colony.capitalize()}:")
            print(f"  Initial Population: {initial}")
            print(f"  Surviving: {surviving}")
            print(f"  Died: {mortality}")
            print(f"  Survival Rate: {survival_rate*100:.1f}%")
            
            if survival_rate < 0.5:
                print(f"  ⚠️  HIGH MORTALITY - Check energy parameters")
            elif survival_rate == 1.0:
                print(f"  ✓  Perfect survival - May indicate low activity")
            else:
                print(f"  ✓  Healthy survival rate")
    
    def analyze_efficiency(self):
        """Analyze system efficiency"""
        print("\n📈 EFFICIENCY ANALYSIS")
        print("-" * 80)
        
        stats = self.report.get('final_statistics', {})
        config = self.report.get('config', {})
        
        # Structures per beaver
        n_beavers_initial = config.get('n_beavers', 1)
        structures = stats.get('n_structures_built', 0)
        structures_per_beaver = structures / n_beavers_initial
        
        print(f"Beaver Productivity:")
        print(f"  Total Structures: {structures}")
        print(f"  Structures/Beaver: {structures_per_beaver:.2f}")
        
        if structures == 0:
            print(f"  ❌ CRITICAL: No structures built!")
            print(f"     Possible causes:")
            print(f"       - Curvature threshold too high")
            print(f"       - Energy decay too fast")
            print(f"       - Beavers dying before building")
        
        # Packets transported
        packets = stats.get('n_packets_transported', 0)
        n_bees_initial = config.get('n_bees', 1)
        packets_per_bee = packets / n_bees_initial
        
        print(f"\nBee Transport:")
        print(f"  Total Packets: {packets}")
        print(f"  Packets/Bee: {packets_per_bee:.2f}")
        
        if packets == 0:
            print(f"  ℹ️  No packets transported (expected in early simulation)")
        
        # Energy efficiency
        total_energy_consumed = self.log_data[0]['energy'] - self.log_data[-1]['energy'] if self.log_data else 0
        work_done = structures + packets  # Simple metric
        
        if work_done > 0:
            energy_per_work = total_energy_consumed / work_done
            print(f"\nEnergy Efficiency:")
            print(f"  Energy/Work Unit: {energy_per_work:.2f}")
        else:
            print(f"\nEnergy Efficiency:")
            print(f"  ❌ No work performed - cannot compute efficiency")
    
    def identify_issues(self):
        """Identify potential issues"""
        print("\n🔍 ISSUE DETECTION")
        print("-" * 80)
        
        issues = []
        warnings = []
        
        # Check structures
        if self.report.get('final_statistics', {}).get('n_structures_built', 0) == 0:
            issues.append({
                'severity': 'CRITICAL',
                'component': 'Beavers',
                'issue': 'No structures built',
                'solution': 'Lower curvature threshold from 0.5 to 0.1, increase beaver energy'
            })
        
        # Check beaver survival
        n_agents = self.report.get('n_agents', {})
        config = self.report.get('config', {})
        
        beaver_survival = n_agents.get('beavers', 0) / config.get('n_beavers', 1)
        if beaver_survival < 0.5:
            issues.append({
                'severity': 'WARNING',
                'component': 'Beavers',
                'issue': f'High mortality ({(1-beaver_survival)*100:.0f}%)',
                'solution': 'Decrease energy decay rate or increase initial energy'
            })
        
        # Check ant/bee survival
        if n_agents.get('ants', 0) == config.get('n_ants', 0):
            warnings.append({
                'severity': 'INFO',
                'component': 'Ants',
                'issue': 'Perfect survival (100%)',
                'solution': 'May indicate low activity - check if ants are exploring'
            })
        
        if n_agents.get('bees', 0) == config.get('n_bees', 0):
            warnings.append({
                'severity': 'INFO',
                'component': 'Bees',
                'issue': 'Perfect survival (100%)',
                'solution': 'May indicate low activity - check if bees are transporting'
            })
        
        # Display issues
        if issues:
            print("⚠️  ISSUES FOUND:\n")
            for i, issue in enumerate(issues, 1):
                print(f"{i}. [{issue['severity']}] {issue['component']}")
                print(f"   Problem: {issue['issue']}")
                print(f"   Solution: {issue['solution']}\n")
        else:
            print("✓ No critical issues detected")
        
        if warnings:
            print("\nℹ️  WARNINGS:\n")
            for i, warning in enumerate(warnings, 1):
                print(f"{i}. [{warning['severity']}] {warning['component']}")
                print(f"   Note: {warning['issue']}")
                print(f"   Suggestion: {warning['solution']}\n")
    
    def provide_recommendations(self):
        """Provide recommendations for improvement"""
        print("\n💡 RECOMMENDATIONS")
        print("-" * 80)
        
        recommendations = []
        
        # Based on results
        if self.report.get('final_statistics', {}).get('n_structures_built', 0) == 0:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Use Enhanced Simulation',
                'reason': 'Fixes beaver construction bug with proper curvature computation',
                'command': 'python blackhole_archive_enhanced.py'
            })
        
        # General improvements
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Increase simulation duration',
            'reason': 'Allow more time for semantic graph emergence',
            'command': 'Set t_max=200.0 in config'
        })
        
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Enable advanced visualizations',
            'reason': 'Better insights into colony behavior',
            'command': 'Use AdvancedVisualizer after simulation'
        })
        
        recommendations.append({
            'priority': 'LOW',
            'action': 'Tune agent parameters',
            'reason': 'Optimize energy balance and activity levels',
            'command': 'Adjust beaver_build_rate, ant_pheromone_strength, etc.'
        })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']} PRIORITY] {rec['action']}")
            print(f"   Why: {rec['reason']}")
            print(f"   How: {rec['command']}")
    
    def generate_report(self, output_path: str = "analysis_report.txt"):
        """Generate text report"""
        import sys
        from io import StringIO
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        self.analyze_all()
        
        report_text = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Save to file
        output_file = self.results_dir / output_path
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n📄 Report saved to: {output_file}")
        
        # Also print to console
        print(report_text)

# Usage
if __name__ == "__main__":
    import sys
    
    # Get results directory from command line or use default
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    
    analyzer = SimulationAnalyzer(results_dir)
    analyzer.analyze_all()
    analyzer.generate_report()
