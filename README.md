# The Blackhole Archive Project

**Multi-Agent Emergent Cognition: Swarm Intelligence for Collective Knowledge Construction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

The Blackhole Archive is a **multi-agent simulation framework** exploring how swarm agents collectively build and maintain knowledge structures under adversarial pressure.

The black hole serves as a **conceptual metaphor**: an environment where information is difficult to preserve, resources are scarce, and coordination is challenging. Agents must cooperate to prevent knowledge from being "lost to the void."

### Three Colony Types

- **🦫 Beavers** (Structural Layer): Build infrastructure, reduce traversal costs, manage shared resources
- **🐜 Ants** (Semantic Layer): Construct knowledge graphs via stigmergy, create vertices and edges through exploration
- **🐝 Bees** (Transport Layer): Move information packets between graph vertices, scout for high-value targets

### What This Project Actually Does

| Component | Implementation |
|-----------|---------------|
| **Environment** | Schwarzschild spacetime with geodesic motion (curved space navigation) |
| **Knowledge Graph** | NetworkX directed graph with creation/pruning/merging lifecycle |
| **Adversarial Pressure** | 16 threat types (energy drought, noise injection, packet loss, etc.) |
| **Learning** | Threat memory, social learning, strategy adaptation |
| **Energy Economics** | Bounded resources, no perpetual motion, realistic constraints |

This is **swarm intelligence research** with physics-inspired environment dynamics.

## Research Goals

This project explores **emergent intelligence** through self-organizing multi-agent systems:

### Core Research Questions

1. **Self-Organizing Belief Networks**: Can agents collectively build and maintain complex knowledge structures (graph vertices) without central control?

2. **Multi-Agent Cooperation**: How do heterogeneous colonies (Ants, Bees, Beavers) achieve collective cognition through local interactions?

3. **Resilience Under Perturbation**: Can the system maintain coherence despite adversarial conditions (APL triggers: noise injection, fatigue waves, congestion)?

4. **Graph Lifecycle Dynamics**: What is the optimal balance between merging (consolidation) and pruning (forgetting) to maintain healthy knowledge graphs?

### Success Criteria

A successful simulation demonstrates:
- **Surviving colonies** through 10,000+ steps
- **500+ stable vertices** (beliefs/concepts)
- **Passing GraphLedger validation** (created - pruned - merged = alive)
- **Healthy V/Ant ratio** (cognitive density per agent)

This proves a **self-sustaining cognitive architecture** where agents collectively build knowledge that doesn't collapse or explode under perturbation.

## Recent Fixes (December 2024)

### Graph Collapse Fix ("Black Hole" Bug)

**Problem**: The merge function was too aggressive, consuming vertices faster than they could be created. This caused the knowledge graph to collapse to minimal size.

**Diagnosis via A/B Testing**:
| Configuration | Vertices at Step 100 |
|--------------|---------------------|
| `DISABLE_MERGE=1` | 721 |
| `threshold=0.3` | 115 (606 merges!) |
| `threshold=0.05` + salience check | 204 ✓ |

**Solution** (commit `3429fb1`):
1. Reduced merge distance threshold: 0.3 → 0.05
2. Added salience similarity requirement (within 0.3)
3. Only true duplicates now merge, preserving semantic diversity

**Results**:
- 160% improvement in vertex retention
- Healthy graph growth throughout simulation
- GraphLedger validation passes

## Quick Start

### Installation

```bash
# Clone or download the project files
cd blackhole_archive_project

# Install dependencies
pip install numpy scipy torch networkx fastapi h5py pandas pydantic matplotlib tqdm

# Run demo simulation
python blackhole_archive_main.py --mode demo

# Run full simulation (warning: computationally intensive)
python blackhole_archive_main.py --mode full --output ./results

# Run enhanced simulation with all fixes (recommended)
python blackhole_archive_enhanced.py

# Quick test mode (1000 steps, faster iterations)
SIM_FAST=1 python blackhole_archive_enhanced.py

# Live visualization mode - watch agents in real-time
VIS=1 python blackhole_archive_enhanced.py                    # Linux/Mac
set VIS=1 && python blackhole_archive_enhanced.py             # Windows CMD

# Combine: fast test with visualization
VIS=1 SIM_FAST=1 python blackhole_archive_enhanced.py         # Linux/Mac
set VIS=1 && set SIM_FAST=1 && python blackhole_archive_enhanced.py  # Windows
```

### Enhanced Simulation

The enhanced simulation (`blackhole_archive_enhanced.py`) includes critical fixes and improvements:

| Feature | Description |
|---------|-------------|
| **GraphLedger** | Tracks vertex lifecycle (created/pruned/merged) for validation |
| **Conservative Merge** | Threshold 0.05 + salience similarity check prevents graph collapse |
| **APL System** | Adaptive Perturbation Layer introduces realistic challenges |
| **Energy Balance** | Proper energy conservation across all colony types |
| **Vectorized Geodesics** | numpy einsum acceleration (3-5x faster) |
| **Live Visualization** | Real-time agent positions, graph, and metrics |
| **Checkpointing** | Auto-saves every 500 steps, crash recovery support |

**Runtime Estimates:**
- Quick test (`SIM_FAST=1`, 1000 steps): ~20 minutes
- Full simulation (10,000 steps): ~12-14 hours
- With visualization: ~30% slower (rendering overhead)

### Live Visualization

Run with `VIS=1` to see real-time:
- **Agent positions** in r-θ projection (Beavers=brown, Ants=red, Bees=yellow)
- **Knowledge graph** with salience-colored vertices
- **Energy & vertex count** over time
- **Live metrics** panel with colony status

### Crash Recovery

The simulation automatically saves checkpoints every 500 steps. If the simulation crashes:

1. Checkpoints are saved to `./enhanced_results/checkpoints/`
2. An emergency checkpoint is saved on crash
3. Resume with:

```python
from blackhole_archive_enhanced import EnhancedSimulationEngine

# Load from checkpoint and resume
engine = EnhancedSimulationEngine.load_checkpoint("enhanced_results/checkpoints/checkpoint_step_002500.json")
engine.run()  # Continues from step 2500
```

### Expected Output

```
[INFO] Initializing spacetime...
[INFO] Initializing agents...
[INFO] Simulation engine initialized
[INFO] Starting simulation: 1000 steps
Simulation: 100%|████████████████████| 1000/1000 [00:45<00:00, 22.13it/s]
[INFO] Step 100/1000, t=1.00, Energy=245.50
[INFO] Step 200/1000, t=2.00, Energy=238.20
...
[INFO] Simulation complete
[INFO] Report saved to ./blackhole_archive_output/simulation_report.json

Visualizations saved to ./blackhole_archive_output/visualizations
Final energy: 215.30
```

## Architecture

### 1. Physics Engine (`blackhole_archive_simulation.py`)

Handles curved spacetime dynamics:
- **Schwarzschild metric**: Background black hole geometry
- **Morris-Thorne wormhole**: Traversable connection
- **Geodesic integration**: Particle trajectories
- **Field evolution**: Scalar/vector fields on curved manifold

**Key Classes:**
- `SchwarzschildMetric`: Black hole spacetime
- `MorrisThrorneWormhole`: Wormhole geometry
- `GeodesicIntegrator`: Relativistic particle motion
- `FieldEvolver`: Field dynamics with covariant derivatives

### 2. Colony Agents (`blackhole_archive_simulation.py`)

Three specialized colonies with distinct roles:

#### Beavers (Structural Layer)
- **Purpose**: Spacetime engineering
- **Capabilities**:
  - Build scaffolds at unstable regions (high curvature)
  - Maintain wormhole throat with exotic matter field
  - Regulate information flow via topology modification
- **Algorithm**: Seek curvature gradients, deposit structural field
- **Coupling**: σ_B field contributes to stress-energy tensor

#### Ants (Semantic Layer)
- **Purpose**: Information organization
- **Capabilities**:
  - Explore information space, create semantic vertices
  - Deposit pheromones on edges (stigmergy)
  - Achieve consensus through graph Laplacian dynamics
- **Algorithm**: Random walk + pheromone following + decay
- **Output**: Distributed semantic graph with confidence weights

#### Bees (Transport Layer)
- **Purpose**: Packet delivery
- **Capabilities**:
  - Scout for high-value information (query ants)
  - Transport packets through wormhole
  - Perform waggle dance for recruitment
  - Maintain temporal coherence (clock synchronization)
- **Algorithm**: Scout → waggle → forager → deliver
- **Protocol**: Priority-based, bandwidth-limited, error-corrected

### 3. Packet System

Bees transport packets between graph vertices:
- Packets have TTL (time-to-live) and expire if not delivered
- FIFO queues at each vertex (max 10 packets)
- Delivery/drop statistics tracked

### 4. Adversarial Pressure Layer

The APL system introduces 16 threat types across categories:
- **Resource**: Energy drought, material scarcity
- **Structural**: Erosion, structural decay
- **Information**: Noise injection, concept drift
- **Communication**: Congestion, packet loss

Pressure budget adapts based on system health.

## Technical Details

### Environment Physics

The simulation uses **Schwarzschild spacetime** as the environment:
- Metric: `ds² = -(1 - 2M/r)dt² + (1 - 2M/r)⁻¹dr² + r²dΩ²`
- Geodesic integration via RK4 or symplectic leapfrog
- Curvature measure: Kretschmann scalar `K = 48M²/r⁶`

This provides curved-space navigation, not full GR dynamics.

### Graph Lifecycle Invariant

The system maintains and validates:
```
vertices_created - vertices_pruned - vertices_merged = vertices_alive
```

This is tracked via GraphLedger and validated each maintenance cycle.

### Energy Conservation

All agent actions have energy costs. The system enforces:
- No energy creation (bounded total)
- Realistic resource constraints
- Material budgets for construction

## File Structure

```
blackhole_archive_project/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── blackhole_archive_mathematics.tex      # Complete mathematical formalization
├── blackhole_archive_simulation.py        # Physics engine + agents
├── blackhole_archive_protocols.py         # Communication protocols
├── blackhole_archive_integration.py       # MycoNet integration
├── blackhole_archive_main.py             # Executable simulation
├── examples/
│   ├── demo_physics.py                   # Physics engine examples
│   ├── demo_colonies.py                  # Agent behavior examples
│   ├── demo_protocols.py                 # Protocol usage examples
│   └── demo_integration.py               # Integration examples
├── tests/
│   ├── test_physics.py                   # Physics engine tests
│   ├── test_agents.py                    # Agent tests
│   ├── test_protocols.py                 # Protocol tests
│   └── test_integration.py               # Integration tests
└── docs/
    ├── architecture.md                   # Detailed architecture
    ├── api_reference.md                  # API documentation
    └── theory.md                         # Theoretical background
```

## Usage Examples

### Example 1: Basic Simulation

```python
from blackhole_archive_main import SimulationEngine, SimulationConfig

# Configure
config = SimulationConfig(
    black_hole_mass=1.0,
    n_beavers=50,
    n_ants=200,
    n_bees=100,
    t_max=100.0
)

# Run
engine = SimulationEngine(config)
engine.run()

# Analyze
print(f"Final energy: {engine.stats['total_energy']}")
print(f"Structures built: {engine.stats['n_structures_built']}")
```

### Example 2: Query Archive

```python
from blackhole_archive_integration import ArchiveInterface, MemoryQuery
import numpy as np

# Connect to archive
interface = ArchiveInterface(archive_connection, myconet_system)

# Create query
query = MemoryQuery(
    query_id="search_1",
    requesting_agent_id="myconet_agent_42",
    semantic_content=np.random.rand(128),  # Embedding
    temporal_context=(0.0, 100.0),
    importance=0.8
)

# Query asynchronously
response = await interface.query_archive(query)

# Convert to MycoNet format
field_data = response.to_myconet_format()
```

### Example 3: Custom Agent

```python
from blackhole_archive_simulation import Agent

class CustomAgent(Agent):
    """Custom agent with specific behavior"""
    
    def update(self, dt: float, environment):
        # Check local conditions
        curvature = environment.get_curvature(self.position)
        
        # Decision logic
        if curvature > self.threshold:
            self.perform_action()
        
        # Update state
        self.position += dt * self.velocity
        self.energy -= dt * self.cost

# Add to simulation
engine.agents['custom'].append(CustomAgent(...))
```

## Advanced Features

### GPU Acceleration

The simulation supports GPU acceleration for field evolution:

```python
config = SimulationConfig(use_gpu=True)
```

Requires: PyTorch with CUDA support

### Parallel Processing

Multi-agent updates can be parallelized:

```python
config = SimulationConfig(n_workers=8)
```

Uses ProcessPoolExecutor for agent colonies.

### Checkpointing

Automatic checkpointing saves simulation state:

```python
config = SimulationConfig(checkpoint_interval=100)
```

Load from checkpoint:

```python
engine.load_checkpoint('checkpoint_1000.h5')
```

### Visualization

Real-time visualization using Plotly/Mayavi:

```python
from blackhole_archive_visualizations import AdvancedVisualizer

viz = AdvancedVisualizer(engine)
viz.render_spacetime_3d()  # Renders 3D spacetime view
viz.plot_energy_evolution()  # Plot agent energy over time
```

## Performance

**Computational Complexity:**
- Physics engine: O(N³) per step (grid-based)
- Agent updates: O(N_agents)
- Graph operations: O(N_vertices + N_edges)
- Total: O(N³ + N_agents) per timestep

**Memory Requirements:**
- Grid: ~8 GB for 128³ resolution
- Agents: ~1 MB per 1000 agents
- Packets: ~1 KB per packet
- Total: ~10 GB for typical simulation

**Recommended Hardware:**
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA with 8+ GB VRAM (optional)
- Storage: 100+ GB for long simulations

## Validation

### Environment Validation

1. **Geodesic motion**:
   - Verify RK4/leapfrog integration accuracy
   - Check energy conservation along trajectories

2. **Curvature computation**:
   - Kretschmann scalar K = 48M²/r⁶
   - Correct behavior near horizon

### Agent Validation

1. **Graph lifecycle**:
   - GraphLedger invariant: created - pruned - merged = alive
   - No vertex count collapse or explosion

2. **Energy balance**:
   - Total system energy bounded
   - No perpetual motion exploits

3. **Packet transport**:
   - Packets picked up from real queues (not fabricated)
   - TTL enforcement, delivery/drop tracking

## Applications

### 1. Swarm Intelligence Research

Multi-agent coordination mechanisms:
- Stigmergy (pheromone-based communication)
- Division of labor across colony types
- Emergent collective behavior

### 2. Knowledge Graph Dynamics

Graph lifecycle management:
- Creation vs pruning vs merging balance
- Stability under adversarial pressure
- Optimal forgetting policies

### 3. Resilient Distributed Systems

Coordination under adversity:
- Recovery from perturbations
- Resource-bounded operation
- Graceful degradation

### 4. Embodied/Bounded AI

Energy-constrained cognition:
- No free computation
- Resource allocation strategies
- Survival-driven behavior

## Future Work

### Post-Run Analysis

After a successful 10,000 step simulation:
- [ ] Pattern mining - analyze belief clusters and vertex connectivity
- [ ] Temporal dynamics - identify when stability emerges and phase transitions
- [ ] Colony mapping - understand specialization and territory formation
- [ ] APL impact analysis - measure recovery times from perturbations

### Cognitive Architecture Enhancements

- [ ] **Hierarchical vertices**: Beliefs containing sub-beliefs (abstraction layers)
- [ ] **Temporal decay**: Unused vertices fade over time (forgetting mechanism)
- [ ] **Associative recall**: Query the graph for semantically related concepts
- [ ] **Contradiction detection**: Identify and resolve conflicting beliefs
- [ ] **Goal-directed behavior**: Agents pursue objectives beyond survival

### Scaling Experiments

| Scale | Colonies | Agents | Steps | Goal |
|-------|----------|--------|-------|------|
| Current | 3 | ~50 | 10K | Proof of concept |
| Medium | 10 | 500 | 100K | Emergent societies |
| Large | 50 | 5000 | 1M | Division of labor |

### Technical Improvements

- [ ] GPU-accelerated field evolution
- [ ] Real-time 3D visualization
- [ ] Machine learning for agent policies
- [ ] Checkpointing for long simulations
- [ ] Parameter sweep automation

### Research Directions

- [ ] **Language grounding**: Map vertices to natural language concepts
- [ ] **External interaction**: Feed real-world data as environmental stimuli
- [ ] **Comparative studies**: Benchmark against other multi-agent frameworks
- [ ] **Information integration**: Measure emergence metrics (entropy, complexity)
- [ ] **Perturbation studies**: Targeted attacks, resource scarcity, inter-colony competition

### Long-Term Vision

- [ ] Neuromorphic hardware implementation
- [ ] Application to real swarm robotics
- [ ] Integration with LLM-based agents
- [ ] Practical distributed knowledge systems

## Contributing

Contributions welcome! Areas of interest:
- Physics simulation improvements
- Agent behavior algorithms
- Protocol optimizations
- Integration with other systems
- Documentation and examples

## Citation

If you use this work, please cite:

```bibtex
@software{blackhole_archive_2025,
  title={The Blackhole Archive Project: Information Extraction from Causally Hostile Domains},
  author={[Authors]},
  year={2025},
  url={https://github.com/[username]/blackhole-archive}
}
```

## License

MIT License - see LICENSE file

## Acknowledgments

This project synthesizes ideas from:
- General Relativity (Einstein, Morris-Thorne)
- Information Theory (Bekenstein, Hawking, Susskind)
- Distributed Systems (Lamport, Fischer-Lynch-Paterson)
- Swarm Intelligence (Bonabeau, Dorigo)
- Theoretical Computer Science (Kolmogorov, Shannon)

## Contact

For questions, discussions, or collaborations:
- Open an issue on GitHub
- Email: [contact email]
- Twitter: [@blackhole_archive]

---

**"Exploring emergent cognition through multi-agent swarm simulations."**
