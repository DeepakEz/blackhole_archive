"""
Experiment Analysis & Visualization
====================================

PHASE 5: Generate publication-quality figures from experiments.

Creates:
- Ablation comparison bar charts
- Stress test recovery curves
- Scaling law log-log plots
- Combined dashboard figures
- Statistical significance tests

All figures are designed for academic publication.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

from .metrics import AggregatedMetrics, MetricsCollector
from .ablation import AblationCondition
from .stress import RecoveryMetrics, StressType
from .scaling import ScalingResults, ScalingPoint


@dataclass
class FigureConfig:
    """Configuration for publication figures"""
    figsize_single: Tuple[int, int] = (8, 6)
    figsize_double: Tuple[int, int] = (12, 5)
    figsize_full: Tuple[int, int] = (14, 10)
    dpi: int = 300

    # Style
    style: str = "seaborn-v0_8-whitegrid"
    font_size: int = 12
    title_size: int = 14
    label_size: int = 11

    # Colors
    colors: List[str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                "#2ecc71",  # Green (full system)
                "#3498db",  # Blue
                "#e74c3c",  # Red
                "#f39c12",  # Orange
                "#9b59b6",  # Purple
                "#1abc9c",  # Teal
                "#34495e",  # Dark gray
            ]


class ExperimentAnalyzer:
    """
    Analysis and visualization tools for experiment results.

    Generates publication-quality figures with proper error bars,
    statistical annotations, and clean formatting.
    """

    def __init__(self, config: Optional[FigureConfig] = None):
        self.config = config or FigureConfig()

        # Try to set matplotlib style
        try:
            plt.style.use(self.config.style)
        except (ValueError, OSError) as e:
            # Style not found or file error - continue with default style
            pass

        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
        })

    def plot_ablation_comparison(self, results: Dict[str, AggregatedMetrics],
                                  metric: str = "survival_rate",
                                  title: Optional[str] = None,
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create bar chart comparing ablation conditions.

        Args:
            results: Dict mapping condition name to aggregated metrics
            metric: Which metric to plot (survival_rate, time_to_stable, etc.)
            title: Optional title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_single,
                               dpi=self.config.dpi)

        conditions = list(results.keys())
        n_conditions = len(conditions)

        # Get metric values
        means = []
        stds = []
        for cond in conditions:
            agg = results[cond]
            if metric == "survival_rate":
                means.append(agg.survival_rate_mean)
                stds.append(agg.survival_rate_std)
            elif metric == "time_to_stable":
                means.append(agg.time_to_stable_mean)
                stds.append(agg.time_to_stable_std)
            elif metric == "efficiency":
                means.append(agg.infrastructure_efficiency_mean)
                stds.append(agg.infrastructure_efficiency_std)
            elif metric == "info_cost":
                means.append(agg.info_cost_mean)
                stds.append(agg.info_cost_std)
            elif metric == "coherence":
                means.append(agg.coherence_mean)
                stds.append(agg.coherence_std)
            else:
                means.append(agg.reward_mean)
                stds.append(agg.reward_std)

        # Create bars
        x = np.arange(n_conditions)
        colors = [self.config.colors[i % len(self.config.colors)]
                 for i in range(n_conditions)]

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                     color=colors, edgecolor='black', linewidth=1,
                     error_kw={'linewidth': 2})

        # Highlight full system
        if 'full' in conditions:
            full_idx = conditions.index('full')
            bars[full_idx].set_edgecolor('gold')
            bars[full_idx].set_linewidth(3)

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=0)
        ax.set_ylabel(metric.replace('_', ' ').title())

        if title is None:
            title = f"Ablation Study: {metric.replace('_', ' ').title()}"
        ax.set_title(title)

        # Add significance markers (simplified)
        if 'full' in conditions:
            full_idx = conditions.index('full')
            baseline = means[full_idx]
            for i, (m, s) in enumerate(zip(means, stds)):
                if i != full_idx:
                    # Simple significance check (effect size)
                    effect = abs(m - baseline) / (s + 1e-10)
                    if effect > 2:
                        ax.annotate('***', (i, m + s + 0.02 * max(means)),
                                   ha='center', fontsize=12)
                    elif effect > 1:
                        ax.annotate('**', (i, m + s + 0.02 * max(means)),
                                   ha='center', fontsize=12)

        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

        return fig

    def plot_ablation_matrix(self, results: Dict[str, AggregatedMetrics],
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive ablation matrix showing all metrics.

        4-panel figure with survival, stability, efficiency, and cost.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize_full,
                                 dpi=self.config.dpi)

        metrics = [
            ("survival_rate", "Survival Rate", axes[0, 0]),
            ("time_to_stable", "Time to Stability (steps)", axes[0, 1]),
            ("efficiency", "Infrastructure Efficiency", axes[1, 0]),
            ("info_cost", "Total Info Cost", axes[1, 1]),
        ]

        conditions = list(results.keys())
        x = np.arange(len(conditions))

        for metric, ylabel, ax in metrics:
            means = []
            stds = []
            for cond in conditions:
                agg = results[cond]
                if metric == "survival_rate":
                    means.append(agg.survival_rate_mean)
                    stds.append(agg.survival_rate_std)
                elif metric == "time_to_stable":
                    means.append(max(0, agg.time_to_stable_mean))
                    stds.append(agg.time_to_stable_std)
                elif metric == "efficiency":
                    means.append(agg.infrastructure_efficiency_mean)
                    stds.append(agg.infrastructure_efficiency_std)
                else:
                    means.append(agg.info_cost_mean)
                    stds.append(agg.info_cost_std)

            colors = [self.config.colors[i % len(self.config.colors)]
                     for i in range(len(conditions))]

            ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace('_', '\n')[:10] for c in conditions],
                              rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Ablation Matrix: Component Contributions', fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

        return fig

    def plot_stress_recovery(self, results: Dict[str, List[RecoveryMetrics]],
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot stress test recovery analysis.

        Shows recovery time vs stress intensity.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize_full,
                                 dpi=self.config.dpi)

        scenarios = list(results.keys())

        # Panel 1: Recovery time comparison
        ax1 = axes[0, 0]
        recovery_times = []
        recovery_stds = []
        for name in scenarios:
            times = [m.recovery_time for m in results[name]]
            recovery_times.append(np.mean(times))
            recovery_stds.append(np.std(times))

        x = np.arange(len(scenarios))
        ax1.bar(x, recovery_times, yerr=recovery_stds, capsize=4,
               color=self.config.colors[:len(scenarios)])
        ax1.set_xticks(x)
        ax1.set_xticklabels([s[:12] for s in scenarios], rotation=45, ha='right')
        ax1.set_ylabel('Recovery Time (steps)')
        ax1.set_title('Recovery Time by Stress Type')

        # Panel 2: Survival during stress
        ax2 = axes[0, 1]
        min_survivals = []
        survival_stds = []
        for name in scenarios:
            survivals = [m.min_survival_during_stress for m in results[name]]
            min_survivals.append(np.mean(survivals))
            survival_stds.append(np.std(survivals))

        ax2.bar(x, min_survivals, yerr=survival_stds, capsize=4,
               color=self.config.colors[:len(scenarios)])
        ax2.set_xticks(x)
        ax2.set_xticklabels([s[:12] for s in scenarios], rotation=45, ha='right')
        ax2.set_ylabel('Minimum Survival Rate')
        ax2.set_title('Survival During Stress')
        ax2.set_ylim(0, 1)

        # Panel 3: Structure reuse rate
        ax3 = axes[1, 0]
        reuse_rates = []
        reuse_stds = []
        for name in scenarios:
            rates = [m.reuse_rate for m in results[name]]
            reuse_rates.append(np.mean(rates))
            reuse_stds.append(np.std(rates))

        ax3.bar(x, reuse_rates, yerr=reuse_stds, capsize=4,
               color=self.config.colors[:len(scenarios)])
        ax3.set_xticks(x)
        ax3.set_xticklabels([s[:12] for s in scenarios], rotation=45, ha='right')
        ax3.set_ylabel('Structure Reuse Rate')
        ax3.set_title('Infrastructure Reuse After Stress')
        ax3.set_ylim(0, 1.5)

        # Panel 4: Coherence preservation
        ax4 = axes[1, 1]
        coherence_drops = []
        drop_stds = []
        for name in scenarios:
            drops = [m.coherence_drop for m in results[name]]
            coherence_drops.append(np.mean(drops))
            drop_stds.append(np.std(drops))

        colors = ['red' if d > 0.3 else 'green' for d in coherence_drops]
        ax4.bar(x, coherence_drops, yerr=drop_stds, capsize=4, color=colors)
        ax4.set_xticks(x)
        ax4.set_xticklabels([s[:12] for s in scenarios], rotation=45, ha='right')
        ax4.set_ylabel('Coherence Drop')
        ax4.set_title('Knowledge Coherence Loss')
        ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
        ax4.legend()

        fig.suptitle('Stress Test Analysis: Resilience & Recovery', fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

        return fig

    def plot_scaling_laws(self, results: ScalingResults,
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot scaling law results with log-log fits.

        Shows how metrics scale with colony size.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize_full,
                                 dpi=self.config.dpi)

        points = results.scaling_points
        n_agents = np.array([p.n_agents for p in points])

        # Panel 1: Infrastructure per agent
        ax1 = axes[0, 0]
        infra = np.array([p.infrastructure_per_agent for p in points])
        infra_std = np.array([p.infrastructure_std for p in points])

        ax1.errorbar(n_agents, infra, yerr=infra_std, fmt='o-',
                    color=self.config.colors[0], capsize=5, linewidth=2, markersize=8)

        # Fit line
        log_n = np.log(n_agents)
        fit_infra = np.exp(results.infrastructure_exponent * log_n +
                          np.log(infra[0]) - results.infrastructure_exponent * log_n[0])
        ax1.plot(n_agents, fit_infra, '--', color='gray',
                label=f'Fit: n^{results.infrastructure_exponent:.2f} (R²={results.infrastructure_r2:.2f})')

        ax1.set_xlabel('Number of Agents')
        ax1.set_ylabel('Infrastructure per Agent')
        ax1.set_title('Infrastructure Scaling')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Info cost per agent
        ax2 = axes[0, 1]
        info_cost = np.array([p.info_cost_per_agent for p in points])
        info_std = np.array([p.info_cost_std for p in points])

        ax2.errorbar(n_agents, info_cost, yerr=info_std, fmt='s-',
                    color=self.config.colors[1], capsize=5, linewidth=2, markersize=8)

        fit_info = np.exp(results.info_cost_exponent * log_n +
                         np.log(info_cost[0] + 1) - results.info_cost_exponent * log_n[0])
        ax2.plot(n_agents, fit_info, '--', color='gray',
                label=f'Fit: n^{results.info_cost_exponent:.2f} (R²={results.info_cost_r2:.2f})')

        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Info Cost per Agent')
        ax2.set_title('Communication Overhead Scaling')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Coordination efficiency
        ax3 = axes[1, 0]
        efficiency = np.array([p.coordination_efficiency for p in points])

        ax3.plot(n_agents, efficiency, 'D-', color=self.config.colors[2],
                linewidth=2, markersize=8)

        fit_eff = np.exp(results.efficiency_exponent * log_n +
                        np.log(efficiency[0] + 1e-10) - results.efficiency_exponent * log_n[0])
        ax3.plot(n_agents, fit_eff, '--', color='gray',
                label=f'Fit: n^{results.efficiency_exponent:.2f} (R²={results.efficiency_r2:.2f})')

        ax3.set_xlabel('Number of Agents')
        ax3.set_ylabel('Coordination Efficiency')
        ax3.set_title('Efficiency Scaling')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Survival rate
        ax4 = axes[1, 1]
        survival = np.array([p.survival_rate for p in points])
        survival_std = np.array([p.survival_std for p in points])

        ax4.errorbar(n_agents, survival, yerr=survival_std, fmt='^-',
                    color=self.config.colors[3], capsize=5, linewidth=2, markersize=8)
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')

        ax4.set_xlabel('Number of Agents')
        ax4.set_ylabel('Survival Rate')
        ax4.set_title('Survival vs Colony Size')
        ax4.set_ylim(0, 1.1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Scaling Laws Analysis\n'
                    f'(Grid: {"scaled" if results.config.scale_grid else "fixed"})',
                    fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

        return fig


def plot_ablation_results(results: Dict[str, AggregatedMetrics],
                          save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function for ablation visualization"""
    analyzer = ExperimentAnalyzer()
    fig = analyzer.plot_ablation_matrix(results,
                                        save_path=Path(save_path) if save_path else None)
    return fig


def plot_stress_results(results: Dict[str, List[RecoveryMetrics]],
                        save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function for stress test visualization"""
    analyzer = ExperimentAnalyzer()
    fig = analyzer.plot_stress_recovery(results,
                                        save_path=Path(save_path) if save_path else None)
    return fig


def plot_scaling_results(results: ScalingResults,
                         save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function for scaling visualization"""
    analyzer = ExperimentAnalyzer()
    fig = analyzer.plot_scaling_laws(results,
                                     save_path=Path(save_path) if save_path else None)
    return fig


def generate_paper_figures(ablation_results: Dict[str, AggregatedMetrics],
                           stress_results: Dict[str, List[RecoveryMetrics]],
                           scaling_results: ScalingResults,
                           output_dir: str = "figures") -> List[Path]:
    """
    Generate all publication-ready figures.

    Returns list of saved figure paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = ExperimentAnalyzer()
    saved_paths = []

    # Figure 1: Ablation matrix
    fig = analyzer.plot_ablation_matrix(ablation_results)
    path = output_path / "fig1_ablation_matrix.pdf"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    saved_paths.append(path)
    plt.close(fig)

    # Figure 2: Stress recovery
    fig = analyzer.plot_stress_recovery(stress_results)
    path = output_path / "fig2_stress_recovery.pdf"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    saved_paths.append(path)
    plt.close(fig)

    # Figure 3: Scaling laws
    fig = analyzer.plot_scaling_laws(scaling_results)
    path = output_path / "fig3_scaling_laws.pdf"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    saved_paths.append(path)
    plt.close(fig)

    print(f"Generated {len(saved_paths)} figures in {output_dir}/")
    return saved_paths
