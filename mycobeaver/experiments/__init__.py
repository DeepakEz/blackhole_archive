"""
MycoBeaver Experiments Module
=============================

PHASE 5: Results That Actually Mean Something

Stop thinking "reward curve". Think emergent capability curves.

This module provides a rigorous experimental framework for:
1. Ablation studies - Understanding component contributions
2. Stress tests - Measuring resilience and recovery
3. Scaling laws - Characterizing system behavior at scale

All experiments produce publishable-quality metrics and visualizations.

Modules:
- ablation: Systematic ablation matrix experiments
- stress: Environmental and system stress tests
- scaling: Agent count and grid size scaling experiments
- metrics: Comprehensive metrics collection
- analysis: Statistical analysis and visualization
- runner: Experiment execution and parallelization
"""

from .metrics import (
    MetricsCollector,
    ExperimentMetrics,
    AggregatedMetrics,
)

from .ablation import (
    AblationConfig,
    AblationExperiment,
    run_ablation_matrix,
)

from .stress import (
    StressConfig,
    StressScenario,
    StressExperiment,
    run_stress_tests,
)

from .scaling import (
    ScalingConfig,
    ScalingExperiment,
    run_scaling_experiment,
)

from .analysis import (
    ExperimentAnalyzer,
    plot_ablation_results,
    plot_stress_results,
    plot_scaling_results,
    generate_paper_figures,
)

from .runner import (
    ExperimentRunner,
    ExperimentConfig,
    run_full_experiment_suite,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "ExperimentMetrics",
    "AggregatedMetrics",
    # Ablation
    "AblationConfig",
    "AblationExperiment",
    "run_ablation_matrix",
    # Stress
    "StressConfig",
    "StressScenario",
    "StressExperiment",
    "run_stress_tests",
    # Scaling
    "ScalingConfig",
    "ScalingExperiment",
    "run_scaling_experiment",
    # Analysis
    "ExperimentAnalyzer",
    "plot_ablation_results",
    "plot_stress_results",
    "plot_scaling_results",
    "generate_paper_figures",
    # Runner
    "ExperimentRunner",
    "ExperimentConfig",
    "run_full_experiment_suite",
]
