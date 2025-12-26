"""
CORRECTED FORMAL ACTIVE INFERENCE SYSTEM
Properly adapted for spacetime navigation domain

Key fixes from original:
1. State space: 4D position (not abstract 16D)
2. Observation matrix: C = I (direct position observation)
3. Dynamics: Near-identity (slow drift in spacetime)
4. Proper initialization and numerical stability
"""

import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Optional
from scipy.linalg import inv

# Import existing components
from blackhole_archive_main import SimulationConfig
from blackhole_archive_enhanced import EnhancedSpacetime, EnhancedBeaverAgent
from epistemic_cognitive_layer import EpistemicSemanticGraph, Overmind


# =============================================================================
# DOMAIN-SPECIFIC STATE-SPACE MODEL (4D Position Space)
# =============================================================================

class SpacetimeStateSpaceModel:
    """
    State-space model adapted for spacetime navigation
    
    State: x_t = position ∈ ℝ^4 (t, r, θ, φ)
    Observation: y_t = position + noise (direct observation)
    Transition: x_{t+1} ≈ x_t + velocity * dt + noise (near-identity)
    """
    
    def __init__(self, obs_noise: float = 0.1, process_noise: float = 0.05):
        self.d_state = 4
        self.d_obs = 4
        self.d_action = 4
        
        # Observation model: y = x + noise (identity mapping)
        self.C = np.eye(4)
        self.R = obs_noise**2 * np.eye(4)
        
        # Transition: Nearly identity (slow drift)
        self.A = 0.98 * np.eye(4)  # 98% persistence
        self.B = 0.05 * np.eye(4)  # Small control
        self.Q = process_noise**2 * np.eye(4)
        
        # Precompute for efficiency
        self.R_inv = inv(self.R)
        self.Q_inv = inv(self.Q)
    
    def log_observation_likelihood(self, y_t: np.ndarray, x_t: np.ndarray) -> float:
        """p(y|x) = N(y; x, R) since C = I"""
        residual = y_t - x_t
        mahal = residual.T @ self.R_inv @ residual
        _, logdet = np.linalg.slogdet(2 * np.pi * self.R)
        return float(-0.5 * (mahal + logdet))
    
    def sample_observation(self, x_t: np.ndarray) -> np.ndarray:
        """Sample y ~ N(x, R)"""
        noise = np.random.multivariate_normal(np.zeros(4), self.R)
        return x_t + noise


# =============================================================================
# SIMPLIFIED VARIATIONAL INFERENCE (Numerically Stable)
# =============================================================================

class PositionBelief:
    """
    Belief over 4D position with numerical stability
    """
    
    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        self.mean = mean
        self.covariance = covariance
        self._ensure_positive_definite()
    
    def _ensure_positive_definite(self):
        """Ensure covariance stays positive definite"""
        # Add small diagonal if needed
        min_eig = np.min(np.linalg.eigvals(self.covariance))
        if min_eig < 1e-6:
            self.covariance += (1e-5 - min_eig) * np.eye(len(self.mean))
    
    def entropy(self) -> float:
        """H[q] = 0.5 log|2πeΣ|"""
        sign, logdet = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            # Emergency: reset to safe value
            self.covariance = 0.5 * np.eye(len(self.mean))
            sign, logdet = np.linalg.slogdet(self.covariance)
        
        d = len(self.mean)
        return float(0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet)


class StableVariationalInference:
    """
    Numerically stable VI for position tracking
    """
    
    def __init__(self, model: SpacetimeStateSpaceModel):
        self.model = model
    
    def update(self, prior: PositionBelief, observation: np.ndarray) -> PositionBelief:
        """
        Kalman filter update (optimal for linear Gaussian)
        
        Numerically stable implementation
        """
        # Innovation (prediction error)
        innovation = observation - prior.mean
        
        # Innovation covariance S = C Σ_prior C^T + R = Σ_prior + R (since C=I)
        S = prior.covariance + self.model.R
        
        # Kalman gain K = Σ_prior S^{-1}
        try:
            S_inv = inv(S)
        except np.linalg.LinAlgError:
            # Emergency: use pseudoinverse
            S_inv = np.linalg.pinv(S)
        
        K = prior.covariance @ S_inv
        
        # Posterior mean
        mean_post = prior.mean + K @ innovation
        
        # Posterior covariance (Joseph form for stability)
        I_KC = np.eye(4) - K
        cov_post = I_KC @ prior.covariance @ I_KC.T + K @ self.model.R @ K.T
        
        # Ensure symmetry
        cov_post = 0.5 * (cov_post + cov_post.T)
        
        return PositionBelief(mean=mean_post, covariance=cov_post)
    
    def predict(self, belief: PositionBelief, action: np.ndarray) -> PositionBelief:
        """Predict forward"""
        mean_pred = self.model.A @ belief.mean + self.model.B @ action
        cov_pred = self.model.A @ belief.covariance @ self.model.A.T + self.model.Q
        
        return PositionBelief(mean=mean_pred, covariance=cov_pred)
    
    def compute_elbo(self, posterior: PositionBelief, prior: PositionBelief, 
                     observation: np.ndarray) -> float:
        """ELBO = E[log p(y|x)] - KL[q||p]"""
        # Expected log likelihood
        y_pred = posterior.mean  # E[x] under q
        residual = observation - y_pred
        log_lik = -0.5 * residual.T @ self.model.R_inv @ residual
        
        # Uncertainty penalty
        uncertainty = -0.5 * np.trace(self.model.R_inv @ posterior.covariance)
        
        expected_log_lik = log_lik + uncertainty
        
        # KL divergence
        kl = self._kl_divergence(posterior, prior)
        
        return float(expected_log_lik - kl)
    
    def _kl_divergence(self, q: PositionBelief, p: PositionBelief) -> float:
        """
        KL divergence KL[q||p] for Gaussian distributions.

        KL[N(μ₁,Σ₁) || N(μ₂,Σ₂)] =
            0.5 * [tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - d + log|Σ₂|/|Σ₁|]

        Handles numerical issues with proper regularization.
        """
        d = 4

        # Regularize covariance matrices for numerical stability
        eps = 1e-6
        Σp_reg = p.covariance + eps * np.eye(d)
        Σq_reg = q.covariance + eps * np.eye(d)

        try:
            Σp_inv = inv(Σp_reg)
        except np.linalg.LinAlgError:
            # Use pseudoinverse as fallback
            Σp_inv = np.linalg.pinv(Σp_reg)
            logging.warning("KL divergence: using pseudoinverse for Σp")

        diff = p.mean - q.mean

        trace_term = np.trace(Σp_inv @ Σq_reg)
        quad_term = diff.T @ Σp_inv @ diff

        sign_q, logdet_q = np.linalg.slogdet(Σq_reg)
        sign_p, logdet_p = np.linalg.slogdet(Σp_reg)

        # If determinants are non-positive, covariances are degenerate
        # Return a large but finite value to signal this
        if sign_q <= 0 or sign_p <= 0:
            logging.warning(f"KL divergence: degenerate covariance (sign_q={sign_q}, sign_p={sign_p})")
            return 10.0  # Large but finite penalty

        kl = 0.5 * (trace_term + quad_term - d + logdet_p - logdet_q)

        # KL divergence is always non-negative
        return max(0.0, float(kl))

    def expected_free_energy(self, belief: PositionBelief, action: np.ndarray,
                             target: Optional[np.ndarray] = None) -> float:
        """
        Compute Expected Free Energy (EFE) for action selection.

        G(a) = E_q[F(x', y' | a)]

        EFE decomposes into:
        - Epistemic value: -E[H[p(y|x)]] (information gain)
        - Pragmatic value: E[log p(y|C)] (goal achievement)

        For our model with C=I (direct observation):
        G(a) ≈ entropy_increase + goal_distance

        Args:
            belief: Current belief state
            action: Proposed action
            target: Optional goal position (if None, uses uncertainty reduction)

        Returns:
            Expected free energy (lower is better)
        """
        # Predict next state under this action
        predicted = self.predict(belief, action)

        # Epistemic term: expected entropy of observation given prediction
        # Higher uncertainty = higher EFE (we want to reduce uncertainty)
        epistemic_value = predicted.entropy()

        # Pragmatic term: distance to goal (if specified)
        if target is not None:
            goal_distance = np.linalg.norm(predicted.mean - target)
            pragmatic_value = goal_distance
        else:
            # Default: prefer lower radial coordinate (toward information sources)
            pragmatic_value = predicted.mean[1]  # r coordinate

        # EFE is weighted sum
        # λ controls exploration-exploitation tradeoff
        lambda_epistemic = 0.5
        lambda_pragmatic = 0.5

        efe = lambda_epistemic * epistemic_value + lambda_pragmatic * pragmatic_value

        return float(efe)


# =============================================================================
# CORRECTED ACTIVE INFERENCE AGENT
# =============================================================================

class CorrectedActiveInferenceAgent:
    """
    Agent with properly integrated active inference.

    Implements the full active inference loop:
    1. Observe: Get sensory observation
    2. Infer: Update beliefs via variational inference
    3. Plan: Select action by minimizing expected free energy
    4. Act: Execute selected action
    5. Predict: Update beliefs about future state
    """

    # Action sampling parameters
    N_ACTION_SAMPLES = 8  # Number of candidate actions to evaluate
    ACTION_SCALE = 0.15   # Scale of random action perturbations

    def __init__(self, agent_id: str, position: np.ndarray,
                 model: SpacetimeStateSpaceModel, vi: StableVariationalInference,
                 energy: float = 1.0, target: Optional[np.ndarray] = None):
        self.id = agent_id
        self.position = position.copy()
        self.velocity = 0.01 * np.random.randn(4)
        self.energy = energy
        self.state = "active"
        self.target = target  # Optional goal position

        self.model = model
        self.vi = vi

        # Belief initialized at current position with moderate uncertainty
        self.belief = PositionBelief(
            mean=position.copy(),
            covariance=0.5 * np.eye(4)
        )

        # Statistics
        self.elbo_history = []
        self.entropy_history = []
        self.efe_history = []
        self.action_history = []

    def observe(self) -> np.ndarray:
        """Get noisy observation of current position"""
        return self.model.sample_observation(self.position)

    def select_action(self) -> np.ndarray:
        """
        Select action by minimizing Expected Free Energy (EFE).

        Generates candidate actions and evaluates each one's EFE.
        Returns the action with lowest expected free energy.

        This implements true active inference action selection:
        a* = argmin_a G(a)

        where G(a) = E_q[F(x', y' | a)] is the expected free energy.
        """
        # Generate candidate actions
        # Include: current velocity direction, random samples, and zero
        candidates = []

        # 1. Continue current direction (momentum)
        if np.linalg.norm(self.velocity) > 1e-6:
            momentum_action = self.ACTION_SCALE * self.velocity / np.linalg.norm(self.velocity)
            candidates.append(momentum_action)

        # 2. Random exploration actions
        for _ in range(self.N_ACTION_SAMPLES - 2):
            random_action = self.ACTION_SCALE * np.random.randn(4)
            candidates.append(random_action)

        # 3. Zero action (stay still)
        candidates.append(np.zeros(4))

        # 4. If we have a target, add action toward it
        if self.target is not None:
            direction = self.target - self.belief.mean
            if np.linalg.norm(direction) > 1e-6:
                goal_action = self.ACTION_SCALE * direction / np.linalg.norm(direction)
                candidates.append(goal_action)

        # Evaluate EFE for each candidate
        efes = []
        for action in candidates:
            efe = self.vi.expected_free_energy(self.belief, action, self.target)
            efes.append(efe)

        # Select action with minimum EFE
        best_idx = np.argmin(efes)
        best_action = candidates[best_idx]
        best_efe = efes[best_idx]

        # Store for analysis
        self.efe_history.append(best_efe)

        return best_action

    def update(self, dt: float):
        """
        Update agent state using full active inference loop.

        Steps:
        1. OBSERVE: Get sensory data
        2. INFER: Update beliefs via Bayesian inference
        3. PLAN: Select action minimizing expected free energy
        4. ACT: Execute the selected action
        5. PREDICT: Update beliefs about future state
        """
        if self.state != "active":
            return

        # 1. OBSERVE
        y_t = self.observe()

        # 2. INFER (variational update)
        prior = self.belief
        self.belief = self.vi.update(prior, y_t)

        # Compute metrics
        elbo = self.vi.compute_elbo(self.belief, prior, y_t)
        entropy = self.belief.entropy()

        self.elbo_history.append(elbo)
        self.entropy_history.append(entropy)

        # 3. PLAN (action selection via EFE minimization)
        action = self.select_action()
        self.action_history.append(action.copy())

        # 4. ACT (execute action with smoothing)
        # Blend with current velocity for smoother trajectories
        self.velocity = 0.7 * self.velocity + 0.3 * action
        self.position += dt * self.velocity

        # Boundary constraints (stay in valid spacetime region)
        self.position[1] = max(self.position[1], 3.0)  # Stay outside horizon
        self.position[2] = np.clip(self.position[2], 0.01, np.pi - 0.01)  # θ ∈ (0, π)
        self.position[3] = self.position[3] % (2 * np.pi)  # φ ∈ [0, 2π)

        # 5. PREDICT (update beliefs about future)
        self.belief = self.vi.predict(self.belief, action)

        # Energy decay
        self.energy -= dt * 0.003
        if self.energy <= 0:
            self.state = "dead"


# =============================================================================
# CORRECTED SYSTEM
# =============================================================================

class CorrectedFormalSystem:
    """
    Formal active inference system with proper domain adaptation
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Domain-adapted components
        self.logger.info("Initializing domain-adapted state-space model...")
        self.state_space_model = SpacetimeStateSpaceModel(
            obs_noise=0.1,
            process_noise=0.05
        )
        
        self.logger.info("Initializing stable variational inference...")
        self.vi = StableVariationalInference(self.state_space_model)
        
        # Supporting components
        self.logger.info("Initializing substrate...")
        self.spacetime = EnhancedSpacetime(config)
        self.epistemic_graph = EpistemicSemanticGraph(embedding_dim=16)
        self.overmind = Overmind(target_entropy=20.0 * config.n_ants)
        
        # Agents
        self.logger.info("Initializing agents...")
        self.agents = self._initialize_agents()
        
        # Statistics
        self.stats = {
            'average_elbo': [],
            'average_free_energy': [],
            'average_entropy': [],
            'total_energy': [],
            'n_structures': [],
            'n_beliefs': []
        }
        
        self.logger.info("Corrected system initialized")
    
    def _setup_logging(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Fix encoding issue
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    f"{self.config.output_dir}/corrected_system.log",
                    encoding='utf-8'  # Fixed encoding
                ),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("CorrectedSystem")
    
    def _initialize_agents(self):
        agents = {'beavers': [], 'formal_ants': []}
        
        # Beavers
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
        
        # Corrected ants
        for i in range(self.config.n_ants):
            position = np.array([
                0.0,
                self.config.r_min + np.random.rand() * 20,
                np.random.rand() * np.pi,
                np.random.rand() * 2*np.pi
            ])
            agents['formal_ants'].append(CorrectedActiveInferenceAgent(
                agent_id=f"ant_{i}",
                position=position,
                model=self.state_space_model,
                vi=self.vi,
                energy=1.0
            ))
        
        return agents
    
    def run(self):
        """Run corrected simulation"""
        n_steps = int(self.config.t_max / self.config.dt)
        self.logger.info(f"Starting corrected simulation: {n_steps} steps")
        
        for step in tqdm(range(n_steps), desc="Corrected Active Inference"):
            t = step * self.config.dt
            
            # Update beavers
            for beaver in self.agents['beavers']:
                if beaver.state == "active":
                    beaver.update(self.config.dt, self.spacetime)
            
            # Update formal ants
            for ant in self.agents['formal_ants']:
                if ant.state == "active":
                    ant.update(self.config.dt)
            
            # Statistics
            if step % 10 == 0:
                active_ants = [a for a in self.agents['formal_ants'] if a.state == "active"]
                
                if active_ants:
                    elbos = [a.elbo_history[-1] for a in active_ants if len(a.elbo_history) > 0]
                    entropies = [a.entropy_history[-1] for a in active_ants if len(a.entropy_history) > 0]
                    
                    avg_elbo = float(np.mean(elbos)) if elbos else 0.0
                    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
                    
                    self.stats['average_elbo'].append(avg_elbo)
                    self.stats['average_free_energy'].append(-avg_elbo)
                    self.stats['average_entropy'].append(avg_entropy)
                
                total_energy = sum(a.energy for agents in self.agents.values() 
                                 for a in agents if a.state == "active")
                self.stats['total_energy'].append(total_energy)
                self.stats['n_structures'].append(sum(
                    b.structures_built for b in self.agents['beavers']
                ))
                self.stats['n_beliefs'].append(len(self.epistemic_graph.beliefs))
            
            # Logging
            if step % 100 == 0 and len(self.stats['average_elbo']) > 0:
                self.logger.info(
                    f"t={t:.2f}, E={self.stats['total_energy'][-1]:.2f}, "
                    f"ELBO={self.stats['average_elbo'][-1]:.2f}, "
                    f"H={self.stats['average_entropy'][-1]:.2f}"
                )
        
        self.logger.info("Corrected simulation complete")
        self._save_results()
    
    def _save_results(self):
        """Save corrected results"""
        report_path = Path(self.config.output_dir) / "corrected_report.json"
        
        report = {
            'system': 'Corrected Formal Active Inference System',
            'key_fixes': {
                'state_space': '4D position (not abstract 16D)',
                'observation_model': 'C = I (direct observation)',
                'dynamics': 'Near-identity with drift',
                'numerical_stability': 'Joseph form covariance update'
            },
            'final_statistics': {
                'average_elbo': self.stats['average_elbo'][-1] if self.stats['average_elbo'] else 0,
                'average_free_energy': self.stats['average_free_energy'][-1] if self.stats['average_free_energy'] else 0,
                'average_entropy': self.stats['average_entropy'][-1] if self.stats['average_entropy'] else 0,
                'final_energy': self.stats['total_energy'][-1] if self.stats['total_energy'] else 0,
                'n_structures': self.stats['n_structures'][-1] if self.stats['n_structures'] else 0
            },
            'validation': {
                'entropy_positive': self.stats['average_entropy'][-1] > 0 if self.stats['average_entropy'] else False,
                'elbo_reasonable': abs(self.stats['average_elbo'][-1]) < 100 if self.stats['average_elbo'] else False
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Corrected report saved to {report_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CORRECTED FORMAL ACTIVE INFERENCE SYSTEM")
    print("Domain-adapted for spacetime navigation")
    print("="*80)
    
    print("\n🔧 Key Fixes:")
    print("  1. State space: 4D position (not abstract 16D)")
    print("  2. Observation: C = I (direct position sensing)")
    print("  3. Dynamics: Near-identity A = 0.98I")
    print("  4. Numerical stability: Joseph form updates")
    
    config = SimulationConfig(
        t_max=20.0,
        dt=0.01,
        n_beavers=10,
        n_ants=20,
        n_bees=0,
        output_dir="./corrected_results"
    )
    
    print(f"\n📋 Configuration:")
    print(f"  Duration: {config.t_max} time units")
    print(f"  Beavers: {config.n_beavers}")
    print(f"  Formal ants: {config.n_ants}")
    
    print("\n🚀 Running corrected system...")
    system = CorrectedFormalSystem(config)
    system.run()
    
    print("\n✅ Complete!")
    print("\n📊 Final Statistics:")
    print(f"  Average ELBO: {system.stats['average_elbo'][-1]:.2f}")
    print(f"  Average entropy: {system.stats['average_entropy'][-1]:.2f}")
    print(f"  Final energy: {system.stats['total_energy'][-1]:.2f}")
    print(f"  Structures: {system.stats['n_structures'][-1]}")
    
    print(f"\n✓ Validation:")
    print(f"  Entropy > 0: {system.stats['average_entropy'][-1] > 0}")
    print(f"  ELBO reasonable: {abs(system.stats['average_elbo'][-1]) < 100}")
    
    print(f"\n📁 Results in: {config.output_dir}")
    print("="*80)
