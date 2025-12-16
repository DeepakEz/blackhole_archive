"""
PHASE 1: FORMAL STATE-SPACE MODEL
Explicit probabilistic observation and transition models

This provides the foundation for rigorous active inference:
- p(y_t | x_t): Observation likelihood
- p(x_{t+1} | x_t, a_t): State transition
- Complete generative model
"""

import numpy as np
from scipy.linalg import inv, cholesky
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class StateSpaceParameters:
    """Parameters for linear Gaussian state-space model"""
    state_dim: int
    obs_dim: int
    action_dim: int
    
    # Observation model: y_t = C x_t + v_t, v_t ~ N(0, R)
    C: np.ndarray  # Observation matrix [obs_dim × state_dim]
    R: np.ndarray  # Observation noise covariance [obs_dim × obs_dim]
    
    # Transition model: x_{t+1} = A x_t + B a_t + w_t, w_t ~ N(0, Q)
    A: np.ndarray  # State transition matrix [state_dim × state_dim]
    B: np.ndarray  # Control matrix [state_dim × action_dim]
    Q: np.ndarray  # Process noise covariance [state_dim × state_dim]


class FormalStateSpaceModel:
    """
    Rigorous state-space model for active inference
    
    Generative model:
        x_0 ~ N(μ_0, Σ_0)           [Initial state]
        x_{t+1} = A x_t + B a_t + w_t,  w_t ~ N(0, Q)  [Transition]
        y_t = C x_t + v_t,           v_t ~ N(0, R)     [Observation]
    
    This is the foundation for formal variational inference.
    """
    
    def __init__(self, params: StateSpaceParameters):
        self.params = params
        self.d_state = params.state_dim
        self.d_obs = params.obs_dim
        self.d_action = params.action_dim
        
        # Validate dimensions
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Ensure all matrices have correct dimensions"""
        assert self.params.C.shape == (self.d_obs, self.d_state)
        assert self.params.R.shape == (self.d_obs, self.d_obs)
        assert self.params.A.shape == (self.d_state, self.d_state)
        assert self.params.B.shape == (self.d_state, self.d_action)
        assert self.params.Q.shape == (self.d_state, self.d_state)
        
        # Check positive definiteness
        assert np.all(np.linalg.eigvals(self.params.R) > 0), "R must be positive definite"
        assert np.all(np.linalg.eigvals(self.params.Q) > 0), "Q must be positive definite"
    
    def log_observation_likelihood(self, y_t: np.ndarray, x_t: np.ndarray) -> float:
        """
        Compute log p(y_t | x_t)
        
        For Gaussian observation model:
        log p(y_t | x_t) = -0.5 * [(y_t - C x_t)^T R^{-1} (y_t - C x_t) + log|2πR|]
        
        Args:
            y_t: Observation [d_obs]
            x_t: State [d_state]
            
        Returns:
            Log probability (scalar)
        """
        # Predicted observation
        y_pred = self.params.C @ x_t
        residual = y_t - y_pred
        
        # Mahalanobis distance
        R_inv = inv(self.params.R)
        mahalanobis = residual.T @ R_inv @ residual
        
        # Log determinant
        sign, logdet = np.linalg.slogdet(2 * np.pi * self.params.R)
        
        log_prob = -0.5 * (mahalanobis + logdet)
        return float(log_prob)
    
    def log_transition_likelihood(self, 
                                  x_next: np.ndarray, 
                                  x_t: np.ndarray, 
                                  a_t: np.ndarray) -> float:
        """
        Compute log p(x_{t+1} | x_t, a_t)
        
        For Gaussian transition model:
        log p(x_{t+1} | x_t, a_t) = -0.5 * [(x_{t+1} - A x_t - B a_t)^T Q^{-1} (...) + log|2πQ|]
        
        Args:
            x_next: Next state [d_state]
            x_t: Current state [d_state]
            a_t: Action [d_action]
            
        Returns:
            Log probability (scalar)
        """
        # Predicted next state
        x_pred = self.params.A @ x_t + self.params.B @ a_t
        residual = x_next - x_pred
        
        # Mahalanobis distance
        Q_inv = inv(self.params.Q)
        mahalanobis = residual.T @ Q_inv @ residual
        
        # Log determinant
        sign, logdet = np.linalg.slogdet(2 * np.pi * self.params.Q)
        
        log_prob = -0.5 * (mahalanobis + logdet)
        return float(log_prob)
    
    def sample_observation(self, x_t: np.ndarray) -> np.ndarray:
        """
        Sample y_t ~ p(y_t | x_t)
        
        Args:
            x_t: State [d_state]
            
        Returns:
            Observation [d_obs]
        """
        mean = self.params.C @ x_t
        
        # Sample from N(mean, R)
        noise = np.random.multivariate_normal(np.zeros(self.d_obs), self.params.R)
        y_t = mean + noise
        
        return y_t
    
    def sample_transition(self, x_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        """
        Sample x_{t+1} ~ p(x_{t+1} | x_t, a_t)
        
        Args:
            x_t: Current state [d_state]
            a_t: Action [d_action]
            
        Returns:
            Next state [d_state]
        """
        mean = self.params.A @ x_t + self.params.B @ a_t
        
        # Sample from N(mean, Q)
        noise = np.random.multivariate_normal(np.zeros(self.d_state), self.params.Q)
        x_next = mean + noise
        
        return x_next
    
    def expected_observation(self, x_t: np.ndarray) -> np.ndarray:
        """E[y_t | x_t] = C x_t"""
        return self.params.C @ x_t
    
    def expected_next_state(self, x_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        """E[x_{t+1} | x_t, a_t] = A x_t + B a_t"""
        return self.params.A @ x_t + self.params.B @ a_t


def create_default_state_space_model(
    state_dim: int = 16,
    obs_dim: int = 4,
    action_dim: int = 4,
    observation_noise: float = 0.1,
    process_noise: float = 0.01,
    stability: float = 0.95
) -> FormalStateSpaceModel:
    """
    Create default state-space model with reasonable parameters
    
    Args:
        state_dim: Dimension of latent state (semantic embedding)
        obs_dim: Dimension of observations (spacetime coords)
        action_dim: Dimension of actions (movement)
        observation_noise: Std dev of observation noise
        process_noise: Std dev of process noise
        stability: Decay rate (0 < stability < 1, higher = more stable)
        
    Returns:
        Configured state-space model
    """
    # Observation matrix: Project high-dim state to low-dim obs
    # Use random Gaussian initialization with proper scaling
    C = np.random.randn(obs_dim, state_dim) / np.sqrt(state_dim)
    
    # Observation noise: Isotropic
    R = observation_noise**2 * np.eye(obs_dim)
    
    # State transition: Stable decay with some mixing
    A = stability * np.eye(state_dim)
    # Add small off-diagonal elements for state mixing
    A += 0.01 * np.random.randn(state_dim, state_dim) / np.sqrt(state_dim)
    
    # Control matrix: How actions affect state
    B = np.random.randn(state_dim, action_dim) / np.sqrt(action_dim)
    
    # Process noise: Isotropic
    Q = process_noise**2 * np.eye(state_dim)
    
    params = StateSpaceParameters(
        state_dim=state_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        C=C, R=R, A=A, B=B, Q=Q
    )
    
    return FormalStateSpaceModel(params)


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PHASE 1: FORMAL STATE-SPACE MODEL")
    print("="*80)
    
    # Create model
    model = create_default_state_space_model(
        state_dim=16,
        obs_dim=4,
        action_dim=4
    )
    
    print(f"\n✅ Model created:")
    print(f"  State dimension: {model.d_state}")
    print(f"  Observation dimension: {model.d_obs}")
    print(f"  Action dimension: {model.d_action}")
    
    # Test observation likelihood
    x_t = np.random.randn(16)
    y_t = model.sample_observation(x_t)
    
    log_lik = model.log_observation_likelihood(y_t, x_t)
    print(f"\n📊 Observation likelihood test:")
    print(f"  log p(y|x) = {log_lik:.3f}")
    
    # Test transition likelihood
    a_t = np.random.randn(4)
    x_next = model.sample_transition(x_t, a_t)
    
    log_trans = model.log_transition_likelihood(x_next, x_t, a_t)
    print(f"\n🔄 Transition likelihood test:")
    print(f"  log p(x'|x,a) = {log_trans:.3f}")
    
    # Test forward simulation
    print(f"\n🚀 Forward simulation:")
    x = np.random.randn(16)
    for t in range(5):
        a = 0.1 * np.random.randn(4)
        y = model.sample_observation(x)
        x = model.sample_transition(x, a)
        print(f"  t={t}: ||x|| = {np.linalg.norm(x):.3f}, ||y|| = {np.linalg.norm(y):.3f}")
    
    print(f"\n✅ Phase 1 complete: Formal observation model implemented")
    print("="*80)
