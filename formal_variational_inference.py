"""
PHASE 2: FORMAL VARIATIONAL INFERENCE ENGINE
Rigorous ELBO minimization and posterior updates

This implements true variational Bayes for active inference:
- Closed-form Gaussian variational updates
- ELBO (Evidence Lower Bound) computation
- Expected free energy for action selection
"""

import numpy as np
from scipy.linalg import inv
from typing import Tuple, List, Optional
from dataclasses import dataclass
from formal_state_space import FormalStateSpaceModel, StateSpaceParameters


@dataclass
class VariationalPosterior:
    """
    Variational posterior q(x) = N(μ, Σ)
    
    This is what we optimize in variational inference
    """
    mean: np.ndarray  # μ [d_state]
    covariance: np.ndarray  # Σ [d_state × d_state]
    
    def entropy(self) -> float:
        """H[q(x)] = 0.5 * log|2πeΣ|"""
        d = len(self.mean)
        sign, logdet = np.linalg.slogdet(self.covariance)
        return 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from q(x)"""
        if n_samples == 1:
            return np.random.multivariate_normal(self.mean, self.covariance)
        else:
            return np.random.multivariate_normal(self.mean, self.covariance, size=n_samples)


class FormalVariationalInference:
    """
    Formal variational Bayes for state estimation
    
    Minimizes free energy F[q] = KL[q(x) || p(x)] - E_q[log p(y|x)]
    
    For linear Gaussian models, this has closed-form solution:
        Σ_post = (Σ_prior^{-1} + C^T R^{-1} C)^{-1}
        μ_post = Σ_post (Σ_prior^{-1} μ_prior + C^T R^{-1} y)
    """
    
    def __init__(self, state_space_model: FormalStateSpaceModel):
        self.model = state_space_model
        self.d_state = state_space_model.d_state
        self.d_obs = state_space_model.d_obs
        
        # Precompute frequently used matrices
        self.R_inv = inv(state_space_model.params.R)
        self.Q_inv = inv(state_space_model.params.Q)
        self.obs_precision = state_space_model.params.C.T @ self.R_inv @ state_space_model.params.C
    
    def variational_update(self,
                          prior: VariationalPosterior,
                          observation: np.ndarray) -> VariationalPosterior:
        """
        Compute posterior q(x|y) via closed-form variational Bayes
        
        This is the core of variational inference:
        - Prior: q(x) = N(μ_prior, Σ_prior)
        - Likelihood: p(y|x) = N(y; Cx, R)
        - Posterior: q(x|y) = N(μ_post, Σ_post)
        
        Args:
            prior: Prior distribution
            observation: Observed data
            
        Returns:
            Posterior distribution (optimal q*)
        """
        # Prior precision
        Σ_prior_inv = inv(prior.covariance)
        
        # Posterior precision (information form)
        Σ_post_inv = Σ_prior_inv + self.obs_precision
        
        # Posterior covariance
        Σ_post = inv(Σ_post_inv)
        
        # Posterior mean
        C = self.model.params.C
        information_vec = (
            Σ_prior_inv @ prior.mean +
            C.T @ self.R_inv @ observation
        )
        μ_post = Σ_post @ information_vec
        
        return VariationalPosterior(mean=μ_post, covariance=Σ_post)
    
    def predict_forward(self,
                       posterior: VariationalPosterior,
                       action: np.ndarray) -> VariationalPosterior:
        """
        Predict next state distribution via state transition
        
        Given q(x_t) = N(μ_t, Σ_t) and action a_t,
        compute predictive distribution q(x_{t+1}) = N(μ_{t+1}, Σ_{t+1})
        
        For linear Gaussian:
            μ_{t+1} = A μ_t + B a_t
            Σ_{t+1} = A Σ_t A^T + Q
        
        Args:
            posterior: Current state distribution
            action: Control input
            
        Returns:
            Predicted next state distribution
        """
        A = self.model.params.A
        B = self.model.params.B
        Q = self.model.params.Q
        
        # Predicted mean
        μ_next = A @ posterior.mean + B @ action
        
        # Predicted covariance
        Σ_next = A @ posterior.covariance @ A.T + Q
        
        return VariationalPosterior(mean=μ_next, covariance=Σ_next)
    
    def compute_elbo(self,
                    posterior: VariationalPosterior,
                    prior: VariationalPosterior,
                    observation: np.ndarray) -> float:
        """
        Compute Evidence Lower Bound (ELBO)
        
        ELBO = E_q[log p(y|x)] - KL[q(x) || p(x)]
             = -F[q]  (negative free energy)
        
        Maximizing ELBO = Minimizing free energy
        
        Args:
            posterior: Variational posterior q(x)
            prior: Prior distribution p(x)
            observation: Observed data y
            
        Returns:
            ELBO value (higher is better)
        """
        # KL divergence term
        kl = self._gaussian_kl(posterior, prior)
        
        # Expected log likelihood E_q[log p(y|x)]
        # For Gaussian: E[log p(y|x)] = log p(y|μ) - 0.5 tr(C^T R^{-1} C Σ)
        C = self.model.params.C
        
        # Log likelihood at mean
        y_pred = C @ posterior.mean
        residual = observation - y_pred
        log_lik_mean = -0.5 * residual.T @ self.R_inv @ residual
        
        # Correction for uncertainty
        uncertainty_penalty = -0.5 * np.trace(C.T @ self.R_inv @ C @ posterior.covariance)
        
        expected_log_lik = log_lik_mean + uncertainty_penalty
        
        # ELBO = Expected log likelihood - KL
        elbo = expected_log_lik - kl
        
        return float(elbo)
    
    def compute_free_energy(self,
                           posterior: VariationalPosterior,
                           prior: VariationalPosterior,
                           observation: np.ndarray) -> float:
        """
        Compute free energy F[q]
        
        F = KL[q(x) || p(x)] - E_q[log p(y|x)]
          = -ELBO
        
        Minimizing F = Maximizing ELBO
        
        Args:
            posterior: Variational posterior q(x)
            prior: Prior distribution p(x)
            observation: Observed data y
            
        Returns:
            Free energy (lower is better)
        """
        return -self.compute_elbo(posterior, prior, observation)
    
    def expected_free_energy(self,
                            current_posterior: VariationalPosterior,
                            action: np.ndarray,
                            target_prior: Optional[VariationalPosterior] = None) -> float:
        """
        Compute Expected Free Energy (EFE) for action selection
        
        G(a) = E_q(x)[F[q(x'|a)]]
        
        This is used for active inference: choose action that minimizes EFE
        
        EFE trades off:
        - Epistemic value: Reducing uncertainty (information gain)
        - Pragmatic value: Achieving goals (extrinsic value)
        
        Args:
            current_posterior: Current belief q(x_t)
            action: Proposed action a_t
            target_prior: Goal distribution (if None, use predicted)
            
        Returns:
            Expected free energy (lower is better)
        """
        # Predict next state
        predicted = self.predict_forward(current_posterior, action)
        
        # Use predicted as prior if no target specified
        if target_prior is None:
            target_prior = predicted
        
        # PERFORMANCE FIX: Reduced samples from 100 to 10 (10× speedup)
        n_samples = 10
        efe_samples = []
        
        for _ in range(n_samples):
            # Sample next state from predicted distribution
            x_next = predicted.sample()
            
            # Sample observation from that state
            y_next = self.model.sample_observation(x_next)
            
            # Compute free energy for this sample
            # (Using predicted as posterior is approximate)
            F = self.compute_free_energy(predicted, target_prior, y_next)
            efe_samples.append(F)
        
        # Expected free energy
        efe = float(np.mean(efe_samples))
        
        return efe
    
    def _gaussian_kl(self,
                    q: VariationalPosterior,
                    p: VariationalPosterior) -> float:
        """
        KL divergence KL[q || p] for two Gaussians
        
        KL[N(μ₁,Σ₁) || N(μ₂,Σ₂)] = 
            0.5 * [tr(Σ₂^{-1}Σ₁) + (μ₂-μ₁)^T Σ₂^{-1}(μ₂-μ₁) - d + log|Σ₂|/|Σ₁|]
        """
        d = len(q.mean)
        
        Σ2_inv = inv(p.covariance)
        diff = p.mean - q.mean
        
        trace_term = np.trace(Σ2_inv @ q.covariance)
        quad_term = diff.T @ Σ2_inv @ diff
        
        sign1, logdet1 = np.linalg.slogdet(q.covariance)
        sign2, logdet2 = np.linalg.slogdet(p.covariance)
        
        kl = 0.5 * (trace_term + quad_term - d + logdet2 - logdet1)
        
        return max(0.0, float(kl))


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PHASE 2: FORMAL VARIATIONAL INFERENCE")
    print("="*80)
    
    # Create state-space model
    from formal_state_space import create_default_state_space_model
    
    model = create_default_state_space_model(
        state_dim=16,
        obs_dim=4,
        action_dim=4
    )
    
    # Create VI engine
    vi_engine = FormalVariationalInference(model)
    
    print(f"\n✅ VI Engine created for {model.d_state}D state space")
    
    # Test variational update
    print(f"\n📊 Testing variational update:")
    
    # Prior
    μ_prior = np.random.randn(16)
    Σ_prior = 2.0 * np.eye(16)
    prior = VariationalPosterior(mean=μ_prior, covariance=Σ_prior)
    
    print(f"  Prior entropy: {prior.entropy():.3f}")
    
    # Generate observation
    x_true = np.random.randn(16)
    y = model.sample_observation(x_true)
    
    # Variational update
    posterior = vi_engine.variational_update(prior, y)
    
    print(f"  Posterior entropy: {posterior.entropy():.3f}")
    print(f"  Entropy reduction: {prior.entropy() - posterior.entropy():.3f}")
    
    # Compute ELBO
    elbo = vi_engine.compute_elbo(posterior, prior, y)
    free_energy = vi_engine.compute_free_energy(posterior, prior, y)
    
    print(f"\n🎯 Optimization metrics:")
    print(f"  ELBO: {elbo:.3f} (higher is better)")
    print(f"  Free Energy: {free_energy:.3f} (lower is better)")
    print(f"  ELBO = -F: {abs(elbo + free_energy) < 1e-10}")
    
    # Test forward prediction
    print(f"\n🔄 Testing forward prediction:")
    a = np.random.randn(4)
    predicted = vi_engine.predict_forward(posterior, a)
    
    print(f"  Predicted entropy: {predicted.entropy():.3f}")
    print(f"  Uncertainty increases: {predicted.entropy() > posterior.entropy()}")
    
    # Test Expected Free Energy
    print(f"\n🎲 Testing Expected Free Energy (action selection):")
    actions = [np.random.randn(4) for _ in range(3)]
    efes = []
    
    for i, a in enumerate(actions):
        efe = vi_engine.expected_free_energy(posterior, a)
        efes.append(efe)
        print(f"  Action {i+1}: EFE = {efe:.3f}")
    
    best_action = np.argmin(efes)
    print(f"  Best action: {best_action + 1} (lowest EFE)")
    
    # Test full inference cycle
    print(f"\n🔁 Testing full inference cycle (5 steps):")
    q = prior
    x = x_true.copy()
    
    for t in range(5):
        # Sample action (random for test)
        a = 0.1 * np.random.randn(4)
        
        # Generate observation
        y = model.sample_observation(x)
        
        # Variational update
        q = vi_engine.variational_update(q, y)
        
        # Predict forward
        q = vi_engine.predict_forward(q, a)
        
        # True state evolution
        x = model.sample_transition(x, a)
        
        # Compute tracking error
        error = np.linalg.norm(q.mean - x)
        
        print(f"  t={t}: H[q]={q.entropy():.2f}, error={error:.3f}")
    
    print(f"\n✅ Phase 2 complete: Formal variational inference implemented")
    print("="*80)
