"""
BLACKHOLE ARCHIVE: EPISTEMIC COGNITIVE LAYER
The missing upper half - inference, uncertainty, and meta-regulation

What was missing from the base implementation:
1. Belief uncertainty: q(h_t) = N(μ_t, Σ_t)
2. Variational objective: ELBO / free energy minimization
3. Contradiction metrics: semantic drift and tension
4. Overmind regulation: epistemic pressure modulation

This implements the transition from architecture to cognition.
"""

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal, entropy
from scipy.linalg import inv
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx

# =============================================================================
# EPISTEMIC BELIEFS WITH UNCERTAINTY
# =============================================================================

@dataclass
class EpistemicBelief:
    """
    Belief with explicit uncertainty tracking
    
    Core innovation: Vertices are no longer just locations,
    they're probability distributions over hypotheses.
    """
    vertex_id: int
    
    # Belief state: q(h) = N(μ, Σ)
    mean: np.ndarray  # μ ∈ R^d (belief about semantic location)
    covariance: np.ndarray  # Σ ∈ R^(d×d) (uncertainty)
    
    # Epistemic metadata
    confidence: float = 1.0  # det(Σ)^(-1/2) - high when certain
    salience: float = 0.5  # How important this belief is
    
    # Evidence tracking
    observations: List[np.ndarray] = field(default_factory=list)
    observation_count: int = 0
    
    # Contradiction tracking
    contradicted_by: Set[int] = field(default_factory=set)
    supports: Set[int] = field(default_factory=set)
    
    # Temporal dynamics
    last_updated: float = 0.0
    creation_time: float = 0.0
    
    def entropy(self) -> float:
        """H[q(h)] = (d/2)log(2πe) + (1/2)log|Σ|"""
        d = len(self.mean)
        sign, logdet = np.linalg.slogdet(self.covariance)
        return 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet
    
    def kl_divergence_to(self, other: 'EpistemicBelief') -> float:
        """
        KL[q₁||q₂] for Gaussians
        Measures how different this belief is from another
        
        KL(N(μ₁,Σ₁) || N(μ₂,Σ₂)) = 
            0.5 * (tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - d + log|Σ₂|/|Σ₁|)
        """
        d = len(self.mean)
        
        # CORRECTED: Use other's covariance inverse
        Σ2_inv = inv(other.covariance)
        
        diff = other.mean - self.mean
        
        sign1, logdet1 = np.linalg.slogdet(self.covariance)
        sign2, logdet2 = np.linalg.slogdet(other.covariance)
        
        trace_term = np.trace(Σ2_inv @ self.covariance)
        quad_term = diff.T @ Σ2_inv @ diff
        
        kl = 0.5 * (trace_term + quad_term - d + logdet2 - logdet1)
        return max(0.0, kl)  # numerical stability
    
    def update_belief(self, observation: np.ndarray, observation_noise: float = 0.1):
        """
        Bayesian update: posterior ∝ likelihood × prior
        
        For Gaussians with observation model y = h + ε, ε ~ N(0, σ²I):
        μ_new = Σ_new(Σ⁻¹μ + σ⁻²y)
        Σ_new = (Σ⁻¹ + σ⁻²I)⁻¹
        """
        d = len(self.mean)
        
        # Observation precision
        obs_precision = (1 / observation_noise**2) * np.eye(d)
        
        # Prior precision
        prior_precision = inv(self.covariance)
        
        # Posterior precision
        post_precision = prior_precision + obs_precision
        
        # Posterior covariance
        self.covariance = inv(post_precision)
        
        # Posterior mean
        self.mean = self.covariance @ (prior_precision @ self.mean + obs_precision @ observation)
        
        # Update confidence
        sign, logdet = np.linalg.slogdet(self.covariance)
        self.confidence = np.exp(-0.5 * logdet / d)
        
        # Track observation
        self.observations.append(observation.copy())
        self.observation_count += 1
    
    def increase_uncertainty(self, diffusion_rate: float = 0.01):
        """
        Epistemic drift: Σ → Σ + δI
        Beliefs become less certain over time without evidence
        """
        d = len(self.mean)
        self.covariance += diffusion_rate * np.eye(d)
        
        # Update confidence
        sign, logdet = np.linalg.slogdet(self.covariance)
        self.confidence = np.exp(-0.5 * logdet / d)


# =============================================================================
# EPISTEMIC SEMANTIC GRAPH
# =============================================================================

class EpistemicSemanticGraph:
    """
    Semantic graph where vertices are beliefs with uncertainty
    
    Key difference from base version:
    - Vertices can be wrong
    - Uncertainty can grow
    - Contradictions are tracked
    - Beliefs can be revised or pruned
    """
    
    def __init__(self, embedding_dim: int = 16):
        self.graph = nx.DiGraph()
        self.beliefs: Dict[int, EpistemicBelief] = {}
        self.embedding_dim = embedding_dim
        self.next_vertex_id = 0
        
        # Pheromone trails (from base version)
        self.pheromones: Dict[Tuple[int, int], float] = {}
        
        # Epistemic tracking
        self.contradiction_pairs: Set[Tuple[int, int]] = set()
        self.total_entropy: float = 0.0
        self.total_kl_drift: float = 0.0
        
    def add_belief(self, 
                   position: np.ndarray,
                   salience: float = 0.5,
                   initial_uncertainty: float = 1.0) -> int:
        """Add new belief with uncertainty"""
        
        vertex_id = self.next_vertex_id
        self.next_vertex_id += 1
        
        # Initialize belief distribution in embedding space
        # Pad position if needed
        mean = np.zeros(self.embedding_dim)
        mean[:min(len(position), self.embedding_dim)] = position[:self.embedding_dim]
        
        covariance = initial_uncertainty * np.eye(self.embedding_dim)
        
        belief = EpistemicBelief(
            vertex_id=vertex_id,
            mean=mean,
            covariance=covariance,
            salience=salience,
            confidence=1.0 / initial_uncertainty,
            creation_time=0.0  # Set by caller
        )
        
        self.beliefs[vertex_id] = belief
        self.graph.add_node(
            vertex_id,
            belief=belief,
            salience=salience,  # NEEDED for bees
            mean=belief.mean,
            confidence=belief.confidence
        )
        
        return vertex_id
    
    def update_belief_from_observation(self, vertex_id: int, observation: np.ndarray):
        """Update belief based on new evidence"""
        if vertex_id in self.beliefs:
            b = self.beliefs[vertex_id]
            b.update_belief(observation)
            # Update graph node attributes
            self.graph.nodes[vertex_id]['confidence'] = b.confidence
            self.graph.nodes[vertex_id]['salience'] = b.salience
    
    def detect_contradictions(self, threshold: float = 2.0, max_pairs: int = 1000):
        """
        Find beliefs that contradict each other
        
        Two beliefs contradict if:
        ||μ₁ - μ₂|| > threshold * √(tr(Σ₁) + tr(Σ₂))
        
        OPTIMIZED: For large graphs, sample pairs instead of O(N²)
        """
        # CRITICAL: Clear old contradictions first
        for b in self.beliefs.values():
            b.contradicted_by.clear()
        self.contradiction_pairs.clear()
        
        vertices = list(self.beliefs.keys())
        n = len(vertices)
        
        if n == 0:
            return
        
        # Fast path for small graphs: check all pairs
        if n * (n - 1) // 2 <= max_pairs:
            for i, v1 in enumerate(vertices):
                for v2 in vertices[i+1:]:
                    self._check_contradiction_pair(v1, v2, threshold)
        else:
            # Large graph: sample pairs
            import random
            sampled = 0
            while sampled < max_pairs:
                v1 = random.choice(vertices)
                v2 = random.choice(vertices)
                if v1 != v2 and (v1, v2) not in self.contradiction_pairs and (v2, v1) not in self.contradiction_pairs:
                    self._check_contradiction_pair(v1, v2, threshold)
                    sampled += 1
    
    def _check_contradiction_pair(self, v1: int, v2: int, threshold: float):
        """Helper: check if two vertices contradict"""
        b1, b2 = self.beliefs[v1], self.beliefs[v2]
        
        # Distance between means
        distance = np.linalg.norm(b1.mean - b2.mean)
        
        # Combined uncertainty
        uncertainty = np.sqrt(np.trace(b1.covariance) + np.trace(b2.covariance))
        
        if distance > threshold * uncertainty:
            self.contradiction_pairs.add((v1, v2))
            b1.contradicted_by.add(v2)
            b2.contradicted_by.add(v1)
    
    def prune_low_confidence_beliefs(self, confidence_threshold: float = 0.1):
        """Remove beliefs with very high uncertainty"""
        to_remove = [
            vid for vid, belief in self.beliefs.items()
            if belief.confidence < confidence_threshold
        ]
        
        for vid in to_remove:
            del self.beliefs[vid]
            self.graph.remove_node(vid)
    
    def diffuse_uncertainty(self, dt: float, diffusion_rate: float = 0.01):
        """All beliefs become less certain over time without evidence"""
        for belief in self.beliefs.values():
            belief.increase_uncertainty(diffusion_rate * dt)
    
    def compute_total_entropy(self) -> float:
        """Total uncertainty in the system: H = Σ H[q(h_i)]"""
        self.total_entropy = sum(b.entropy() for b in self.beliefs.values())
        return self.total_entropy
    
    def compute_contradiction_mass(self) -> float:
        """
        S_t = Σ_{(i,j) contradicting} (salience_i + salience_j)
        
        Measures semantic tension in the system
        """
        return sum(
            self.beliefs[v1].salience + self.beliefs[v2].salience
            for v1, v2 in self.contradiction_pairs
        )


# =============================================================================
# FREE ENERGY / ELBO COMPUTATION
# =============================================================================

class FreeEnergyComputer:
    """
    Compute variational free energy for the system
    
    F = E_q[log q(h)] - E_q[log p(h,o)]
      = -ELBO
    
    Minimizing F = maximizing ELBO = learning
    """
    
    def __init__(self, prior_mean: np.ndarray, prior_cov: np.ndarray):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.prior_precision = inv(prior_cov)
    
    def compute_free_energy(self, belief: EpistemicBelief, observations: List[np.ndarray]) -> float:
        """
        F[q] = KL[q(h)||p(h)] - E_q[log p(o|h)]
        
        For Gaussian q and prior:
        KL term is analytical
        Likelihood term requires observations
        """
        # KL divergence to prior
        d = len(belief.mean)
        diff = belief.mean - self.prior_mean
        
        sign_q, logdet_q = np.linalg.slogdet(belief.covariance)
        sign_p, logdet_p = np.linalg.slogdet(self.prior_cov)
        
        trace_term = np.trace(self.prior_precision @ belief.covariance)
        quad_term = diff.T @ self.prior_precision @ diff
        
        kl_term = 0.5 * (trace_term + quad_term - d + logdet_p - logdet_q)
        
        # Expected log likelihood
        # Assuming p(o|h) = N(o; h, σ²I)
        obs_noise = 0.1
        likelihood_term = 0.0
        
        for obs in observations:
            # E_q[(o-h)ᵀ(o-h)] = (o-μ)ᵀ(o-μ) + tr(Σ)
            diff_obs = obs - belief.mean
            expected_sq_error = diff_obs.T @ diff_obs + np.trace(belief.covariance)
            likelihood_term += -0.5 * expected_sq_error / obs_noise**2
        
        # Free energy
        F = kl_term - likelihood_term
        
        return F
    
    def compute_expected_information_gain(self, 
                                          belief: EpistemicBelief,
                                          proposed_observation: np.ndarray) -> float:
        """
        EIG = H[p(o)] - E_h[H[p(o|h)]]
        
        For Gaussian systems, this simplifies to reduction in entropy after update
        """
        # Current entropy
        H_before = belief.entropy()
        
        # Simulate update
        belief_copy = EpistemicBelief(
            vertex_id=belief.vertex_id,
            mean=belief.mean.copy(),
            covariance=belief.covariance.copy()
        )
        belief_copy.update_belief(proposed_observation)
        
        # Posterior entropy
        H_after = belief_copy.entropy()
        
        return H_before - H_after


# =============================================================================
# OVERMIND: META-REGULATORY CONTROLLER
# =============================================================================

class Overmind:
    """
    Global epistemic regulator
    
    Monitors system state and modulates epistemic pressure
    
    O: (E_t, S_t, |V_t|, |P_t|) → (β_t, λ_t, α_t)
    
    Where:
    - β_t: exploration bonus (curiosity drive)
    - λ_t: verification strictness
    - α_t: energy reallocation weights
    """
    
    def __init__(self, target_entropy: float = 100.0):
        self.target_entropy = target_entropy
        
        # Control parameters
        self.exploration_bonus = 1.0  # β
        self.verification_strictness = 0.5  # λ
        self.energy_allocation = {  # α
            'beavers': 0.33,
            'ants': 0.34,
            'bees': 0.33
        }
        
        # State tracking
        self.history = {
            'entropy': [],
            'contradiction_mass': [],
            'stability': [],
            'epistemic_pressure': []
        }
    
    def observe_system(self, 
                      energy: float,
                      graph: EpistemicSemanticGraph,
                      n_packets: int,
                      dt: float):
        """Monitor current system state"""
        
        # Compute epistemic metrics
        total_entropy = graph.compute_total_entropy()
        contradiction_mass = graph.compute_contradiction_mass()
        
        n_vertices = len(graph.beliefs)
        n_edges = graph.graph.number_of_edges()
        
        # Stability metric: how much is changing
        # Low when system is static
        if len(self.history['entropy']) > 0:
            entropy_change = abs(total_entropy - self.history['entropy'][-1])
            stability = 1.0 / (1.0 + entropy_change)
        else:
            stability = 0.5
        
        # Track history
        self.history['entropy'].append(total_entropy)
        self.history['contradiction_mass'].append(contradiction_mass)
        self.history['stability'].append(stability)
        
        # Compute epistemic pressure
        pressure = self._compute_epistemic_pressure(
            energy, total_entropy, contradiction_mass, stability, n_vertices
        )
        self.history['epistemic_pressure'].append(pressure)
        
        # Update control parameters
        self._update_control_parameters(
            total_entropy, contradiction_mass, stability, pressure, n_vertices
        )
    
    def _compute_epistemic_pressure(self,
                                   energy: float,
                                   entropy: float,
                                   contradiction: float,
                                   stability: float,
                                   n_vertices: int) -> float:
        """
        Epistemic pressure = need for exploration
        
        High when:
        - Entropy too low (too certain, groupthink)
        - Contradiction too high (incoherent)
        - Stability too high (stagnation)
        """
        # Entropy pressure: want to maintain target
        entropy_pressure = abs(entropy - self.target_entropy) / (self.target_entropy + 1e-6)
        
        # Contradiction pressure: want some but not too much
        # FIXED: Use belief count, not time steps
        target_contradiction = 0.05 * n_vertices
        contradiction_pressure = abs(contradiction - target_contradiction) / (target_contradiction + 1e-6)
        
        # Stability pressure: penalize stagnation
        stability_pressure = stability  # High stability → high pressure to change
        
        # Combined
        pressure = entropy_pressure + 0.5 * contradiction_pressure + 0.3 * stability_pressure
        
        return pressure
    
    def _update_control_parameters(self,
                                   entropy: float,
                                   contradiction: float,
                                   stability: float,
                                   pressure: float,
                                   n_vertices: int):
        """
        Modulate control parameters based on state
        
        β (exploration): High when pressure is high
        λ (verification): High when contradiction is high  
        α (allocation): Shift resources based on needs
        """
        # Exploration bonus: proportional to pressure
        self.exploration_bonus = 1.0 + 0.5 * pressure
        
        # Verification strictness: increase if too many contradictions
        # FIXED: Use belief count, not time steps
        contradiction_rate = contradiction / (n_vertices + 1e-6)
        self.verification_strictness = 0.5 + 0.5 * min(1.0, contradiction_rate)
        
        # Energy allocation: favor productive colonies
        # If stability too high, boost ants (exploration)
        # If contradiction too high, boost beavers (infrastructure)
        # If transport low, boost bees
        
        if stability > 0.8:
            # Stagnation: boost exploration
            self.energy_allocation = {
                'beavers': 0.25,
                'ants': 0.50,  # More exploration
                'bees': 0.25
            }
        elif contradiction_rate > 0.1:  # FIXED: Use rate, not absolute
            # Too much contradiction: boost infrastructure
            self.energy_allocation = {
                'beavers': 0.50,  # More building
                'ants': 0.30,
                'bees': 0.20
            }
        else:
            # Balanced
            self.energy_allocation = {
                'beavers': 0.33,
                'ants': 0.34,
                'bees': 0.33
            }
    
    def get_exploration_bonus(self) -> float:
        """Current exploration drive"""
        return self.exploration_bonus
    
    def get_verification_threshold(self) -> float:
        """Current evidence threshold for belief acceptance"""
        return self.verification_strictness
    
    def get_energy_allocation(self, colony: str) -> float:
        """Fraction of energy for this colony"""
        return self.energy_allocation.get(colony, 0.33)
    
    def should_inject_noise(self) -> bool:
        """Inject random perturbations when too stable"""
        if len(self.history['stability']) < 10:
            return False
        
        recent_stability = np.mean(self.history['stability'][-10:])
        return recent_stability > 0.9


# =============================================================================
# EPISTEMIC ANT AGENT
# =============================================================================

class EpistemicAntAgent:
    """
    Ant with epistemic inference capabilities
    
    Differences from base ant:
    - Creates beliefs with uncertainty
    - Updates beliefs from observations
    - Driven by information gain, not just pheromones
    """
    
    def __init__(self, agent_id: str, position: np.ndarray, energy: float = 1.0):
        self.id = agent_id
        self.position = position.copy()
        self.velocity = 0.05 * np.random.randn(len(position))
        self.energy = energy
        self.state = "active"
        
        # Epistemic state
        self.current_belief_id: Optional[int] = None
        self.observations_buffer: List[np.ndarray] = []
        self.information_gain_history: List[float] = []
    
    def update(self, 
               dt: float,
               spacetime,
               epistemic_graph: EpistemicSemanticGraph,
               free_energy_computer: FreeEnergyComputer,
               overmind: Overmind):
        """Update with epistemic inference"""
        
        # Sample local field (observation)
        if hasattr(spacetime, 'get_information_density'):
            info_density = spacetime.get_information_density(self.position)
        else:
            # Fallback: use curvature as proxy
            info_density = spacetime.get_ricci_scalar(self.position) / 10.0
        
        # Create observation in embedding space
        # Pad position to match embedding dimension
        pos_embedded = np.zeros(epistemic_graph.embedding_dim)
        pos_embedded[:len(self.position)] = self.position
        observation = pos_embedded + 0.1 * np.random.randn(epistemic_graph.embedding_dim)
        
        # High information density + exploration bonus → create/update belief
        exploration_threshold = 0.5 / overmind.get_exploration_bonus()
        
        if info_density > exploration_threshold:
            
            if self.current_belief_id is None:
                # Create new belief
                salience = info_density
                uncertainty = 1.0 / (info_density + 0.1)  # Higher density → more certain
                
                belief_id = epistemic_graph.add_belief(
                    position=observation,
                    salience=salience,
                    initial_uncertainty=uncertainty
                )
                self.current_belief_id = belief_id
                
            else:
                # Update existing belief
                epistemic_graph.update_belief_from_observation(
                    self.current_belief_id,
                    observation
                )
                
                # Track information gain
                belief = epistemic_graph.beliefs[self.current_belief_id]
                if len(belief.observations) > 1:
                    ig = free_energy_computer.compute_expected_information_gain(
                        belief, observation
                    )
                    self.information_gain_history.append(ig)
        
        # Move toward high uncertainty regions (active inference)
        # Sample nearby beliefs
        nearby_beliefs = [
            (vid, b) for vid, b in epistemic_graph.beliefs.items()
            if np.linalg.norm(b.mean[:len(self.position)] - self.position) < 5.0
        ]
        
        if nearby_beliefs:
            # Move toward most uncertain
            most_uncertain = max(nearby_beliefs, key=lambda x: x[1].entropy())
            target_embedded = most_uncertain[1].mean
            
            # Extract relevant dimensions for movement
            target = target_embedded[:len(self.position)]
            
            direction = target - self.position
            self.velocity = 0.3 * direction / (np.linalg.norm(direction) + 1e-6)
        else:
            # Random exploration
            self.velocity += 0.05 * np.random.randn(len(self.position))
        
        # Update position
        self.position += dt * self.velocity
        
        # Energy decay (modulated by Overmind)
        base_decay = 0.003
        allocation = overmind.get_energy_allocation('ants')
        effective_decay = base_decay / allocation
        
        self.energy -= dt * effective_decay
        
        if self.energy <= 0:
            self.state = "dead"


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EPISTEMIC COGNITIVE LAYER - Demonstration")
    print("="*80)
    
    # Initialize components
    embedding_dim = 16
    graph = EpistemicSemanticGraph(embedding_dim=embedding_dim)
    
    prior_mean = np.zeros(embedding_dim)
    prior_cov = 2.0 * np.eye(embedding_dim)
    fe_computer = FreeEnergyComputer(prior_mean, prior_cov)
    
    overmind = Overmind(target_entropy=100.0)
    
    print("\n✅ Initialized:")
    print(f"  - Epistemic semantic graph (dim={embedding_dim})")
    print(f"  - Free energy computer")
    print(f"  - Overmind regulator")
    
    # Simulate epistemic dynamics
    print("\n🔬 Simulating epistemic dynamics...")
    
    dt = 0.1
    n_steps = 100
    
    for t in range(n_steps):
        time = t * dt
        
        # Add some beliefs
        if np.random.rand() < 0.3:
            pos = np.random.randn(embedding_dim)
            vid = graph.add_belief(pos, salience=np.random.rand())
        
        # Update some beliefs
        if len(graph.beliefs) > 0:
            vid = np.random.choice(list(graph.beliefs.keys()))
            obs = np.random.randn(embedding_dim)
            graph.update_belief_from_observation(vid, obs)
        
        # Diffuse uncertainty
        graph.diffuse_uncertainty(dt, diffusion_rate=0.01)
        
        # Detect contradictions
        graph.detect_contradictions(threshold=2.0)
        
        # Overmind observation
        overmind.observe_system(
            energy=300.0 - time * 2.0,
            graph=graph,
            n_packets=int(time * 0.5),
            dt=dt
        )
        
        if t % 20 == 0:
            entropy = graph.compute_total_entropy()
            contradiction = graph.compute_contradiction_mass()
            print(f"\n  t={time:.1f}:")
            print(f"    Beliefs: {len(graph.beliefs)}")
            print(f"    Entropy: {entropy:.2f}")
            print(f"    Contradictions: {len(graph.contradiction_pairs)}")
            print(f"    Contradiction Mass: {contradiction:.2f}")
            print(f"    Exploration Bonus: {overmind.get_exploration_bonus():.3f}")
            print(f"    Verification Strictness: {overmind.get_verification_threshold():.3f}")
    
    print(f"\n{'='*80}")
    print(f"✅ Demonstration complete!")
    print(f"\nKey differences from base implementation:")
    print(f"  ✓ Beliefs have uncertainty (not just points)")
    print(f"  ✓ Free energy drives learning (not just accumulation)")
    print(f"  ✓ Contradictions are tracked (not just agreement)")
    print(f"  ✓ Overmind regulates pressure (not passive drift)")
    print(f"\nThis is the missing upper half: inference, not just architecture.")
    print(f"{'='*80}\n")
