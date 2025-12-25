"""
BLACKHOLE ARCHIVE: EPISTEMIC ENHANCEMENTS
==========================================

Tier 1 Critical Missing Features:
1. Packet Value Scoring: V = ΔF_expected - λ·cost
2. Belief Compression: MDL-based merging and pruning
3. Overmind Control Laws: Real parameter control, not just noise

This fixes the fundamental issues where:
- Bees can't learn routing (no value signal)
- Beliefs only accumulate, never condense (entropy always wins)
- Overmind only injects noise (no real control)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from scipy.linalg import inv
import heapq


# =============================================================================
# 1. PACKET VALUE SCORING
# =============================================================================

@dataclass
class PacketValue:
    """
    Information value of a packet for routing decisions

    V_packet = ΔF_expected - λ·cost

    Where:
    - ΔF_expected: Expected free energy reduction from delivering this packet
    - λ: Cost weight (transport energy, time dilation, queue position)
    - cost: Actual transport cost
    """
    packet_id: str

    # Information value components
    expected_free_energy_reduction: float = 0.0  # ΔF
    semantic_salience: float = 0.0  # How important the content is
    uncertainty_reduction: float = 0.0  # How much this reduces system uncertainty
    contradiction_resolution: float = 0.0  # Does this resolve contradictions?

    # Cost components
    transport_energy: float = 0.0
    time_dilation_cost: float = 0.0
    queue_delay_cost: float = 0.0

    # Computed value
    cost_weight: float = 0.1  # λ

    @property
    def total_benefit(self) -> float:
        """Total expected benefit from delivery"""
        return (
            self.expected_free_energy_reduction +
            0.5 * self.semantic_salience +
            0.3 * self.uncertainty_reduction +
            0.2 * self.contradiction_resolution
        )

    @property
    def total_cost(self) -> float:
        """Total transport cost"""
        return (
            self.transport_energy +
            self.time_dilation_cost +
            self.queue_delay_cost
        )

    @property
    def value(self) -> float:
        """V = ΔF_expected - λ·cost"""
        return self.total_benefit - self.cost_weight * self.total_cost


class PacketValueComputer:
    """
    Computes packet value for routing decisions

    Used by:
    - Bees: To decide which packets to pick up
    - Wormhole: To decide acceptance/rejection
    - Overmind: To regulate flow
    """

    def __init__(self, free_energy_computer, epistemic_graph):
        self.fe_computer = free_energy_computer
        self.graph = epistemic_graph

        # Learning: track value predictions vs outcomes
        self.predicted_values: Dict[str, float] = {}
        self.actual_outcomes: Dict[str, float] = {}
        self.value_prediction_error: float = 0.0

    def compute_packet_value(self,
                            packet,
                            source_belief_id: int,
                            target_curvature: float,
                            queue_size: int,
                            max_queue: int) -> PacketValue:
        """
        Compute the information value of a packet

        Args:
            packet: The packet to evaluate
            source_belief_id: Which belief this packet carries
            target_curvature: Curvature at delivery point
            queue_size: Current queue depth
            max_queue: Maximum queue capacity
        """
        pv = PacketValue(packet_id=packet.packet_id)

        # Get source belief
        if source_belief_id in self.graph.beliefs:
            belief = self.graph.beliefs[source_belief_id]

            # Expected free energy reduction
            # Higher for high-salience, high-uncertainty beliefs
            pv.expected_free_energy_reduction = belief.salience * belief.entropy()

            # Semantic salience
            pv.semantic_salience = belief.salience

            # Uncertainty reduction
            # Delivering to high-curvature regions reduces uncertainty faster
            pv.uncertainty_reduction = belief.entropy() * (1 + target_curvature)

            # Contradiction resolution
            # Higher if this belief contradicts many others
            n_contradictions = len(belief.contradicted_by)
            pv.contradiction_resolution = n_contradictions * belief.confidence

        # Costs
        pv.transport_energy = 0.01 * (1 + target_curvature)  # More energy near horizon
        pv.time_dilation_cost = target_curvature * 0.1  # Time dilation penalty
        pv.queue_delay_cost = queue_size / max(1, max_queue)  # Congestion penalty

        # Track prediction
        self.predicted_values[packet.packet_id] = pv.value

        return pv

    def record_outcome(self, packet_id: str, actual_benefit: float):
        """
        Record actual outcome after delivery for learning
        """
        self.actual_outcomes[packet_id] = actual_benefit

        if packet_id in self.predicted_values:
            error = actual_benefit - self.predicted_values[packet_id]
            # Exponential moving average of prediction error
            self.value_prediction_error = 0.9 * self.value_prediction_error + 0.1 * abs(error)

    def should_accept_packet(self, pv: PacketValue, threshold: float = 0.0) -> bool:
        """
        Wormhole decision: should this packet be accepted?

        Accept if value > threshold (adjusted by learning)
        """
        # Adjust threshold based on prediction accuracy
        adjusted_threshold = threshold + 0.5 * self.value_prediction_error
        return pv.value > adjusted_threshold


# =============================================================================
# 2. BELIEF COMPRESSION (MDL-based)
# =============================================================================

class BeliefCompressor:
    """
    Minimum Description Length (MDL) based belief compression

    Beliefs should condense, not just accumulate.

    Key operations:
    1. Merge similar beliefs
    2. Prune beliefs that add no information
    3. Collapse contradictions into resolved beliefs
    """

    def __init__(self, epistemic_graph, merge_threshold: float = 0.5):
        self.graph = epistemic_graph
        self.merge_threshold = merge_threshold

        # Statistics
        self.merges_performed = 0
        self.pruned_beliefs = 0
        self.contradictions_resolved = 0

    def compute_description_length(self, belief) -> float:
        """
        MDL: bits needed to describe this belief

        L(B) = L(μ) + L(Σ) + L(observations)
             ≈ d * log(precision) + d² * log(det(Σ)) + n_obs
        """
        d = len(belief.mean)

        # Mean encoding: depends on precision
        sign, logdet = np.linalg.slogdet(belief.covariance)
        precision = np.exp(-logdet / d)
        mean_bits = d * np.log2(max(1, precision))

        # Covariance encoding
        cov_bits = 0.5 * d * d * abs(logdet)

        # Observation encoding
        obs_bits = len(belief.observations) * d * 0.5

        return mean_bits + cov_bits + obs_bits

    def compute_joint_description_length(self, belief1, belief2) -> float:
        """
        MDL for merged belief

        If L(merged) < L(b1) + L(b2), merging is beneficial
        """
        d = len(belief1.mean)

        # Merged mean (weighted by confidence)
        w1 = belief1.confidence / (belief1.confidence + belief2.confidence + 1e-6)
        w2 = 1 - w1
        merged_mean = w1 * belief1.mean + w2 * belief2.mean

        # Merged covariance (union of uncertainties)
        diff1 = belief1.mean - merged_mean
        diff2 = belief2.mean - merged_mean
        merged_cov = (
            w1 * (belief1.covariance + np.outer(diff1, diff1)) +
            w2 * (belief2.covariance + np.outer(diff2, diff2))
        )

        # Combined observations
        n_obs = len(belief1.observations) + len(belief2.observations)

        # Description length
        sign, logdet = np.linalg.slogdet(merged_cov)
        precision = np.exp(-logdet / d)

        mean_bits = d * np.log2(max(1, precision))
        cov_bits = 0.5 * d * d * abs(logdet)
        obs_bits = n_obs * d * 0.5

        return mean_bits + cov_bits + obs_bits

    def find_mergeable_pairs(self, max_pairs: int = 100) -> List[Tuple[int, int, float]]:
        """
        Find pairs of beliefs that should be merged (MDL criterion)

        Returns: List of (id1, id2, compression_ratio)
        """
        mergeable = []
        beliefs = list(self.graph.beliefs.items())
        n = len(beliefs)

        # Sample pairs for efficiency
        checked = 0
        for i in range(n):
            if checked >= max_pairs:
                break
            for j in range(i + 1, n):
                if checked >= max_pairs:
                    break

                vid1, b1 = beliefs[i]
                vid2, b2 = beliefs[j]

                # Quick filter: must be somewhat close
                distance = np.linalg.norm(b1.mean - b2.mean)
                combined_std = np.sqrt(np.trace(b1.covariance) + np.trace(b2.covariance))

                if distance < 2.0 * combined_std:
                    # Compute MDL criterion
                    L_separate = self.compute_description_length(b1) + self.compute_description_length(b2)
                    L_merged = self.compute_joint_description_length(b1, b2)

                    compression_ratio = L_merged / (L_separate + 1e-6)

                    if compression_ratio < self.merge_threshold:
                        mergeable.append((vid1, vid2, compression_ratio))

                checked += 1

        # Sort by compression ratio (best merges first)
        mergeable.sort(key=lambda x: x[2])
        return mergeable

    def merge_beliefs(self, vid1: int, vid2: int) -> int:
        """
        Merge two beliefs into one

        Returns: ID of merged belief
        """
        b1 = self.graph.beliefs[vid1]
        b2 = self.graph.beliefs[vid2]

        # Weighted merge
        w1 = b1.confidence / (b1.confidence + b2.confidence + 1e-6)
        w2 = 1 - w1

        merged_mean = w1 * b1.mean + w2 * b2.mean

        diff1 = b1.mean - merged_mean
        diff2 = b2.mean - merged_mean
        merged_cov = (
            w1 * (b1.covariance + np.outer(diff1, diff1)) +
            w2 * (b2.covariance + np.outer(diff2, diff2))
        )

        # Update b1 in place (keep its ID)
        b1.mean = merged_mean
        b1.covariance = merged_cov
        b1.observations.extend(b2.observations)
        b1.observation_count += b2.observation_count
        b1.salience = max(b1.salience, b2.salience)

        # Update confidence
        d = len(b1.mean)
        sign, logdet = np.linalg.slogdet(b1.covariance)
        b1.confidence = np.exp(-0.5 * logdet / d)

        # Update graph
        self.graph.graph.nodes[vid1]['mean'] = b1.mean
        self.graph.graph.nodes[vid1]['confidence'] = b1.confidence
        self.graph.graph.nodes[vid1]['salience'] = b1.salience

        # Remove b2 using proper cleanup to avoid ghost references
        self.graph._remove_belief(vid2)

        self.merges_performed += 1
        return vid1

    def prune_redundant_beliefs(self, redundancy_threshold: float = 0.9) -> int:
        """
        Remove beliefs that are fully explained by others

        A belief is redundant if another belief has:
        - Very similar mean (within 1σ)
        - Higher confidence
        - Higher salience
        """
        to_remove = []
        beliefs = list(self.graph.beliefs.values())

        for b in beliefs:
            for other in beliefs:
                if b.vertex_id == other.vertex_id:
                    continue

                # Check if other dominates b
                distance = np.linalg.norm(b.mean - other.mean)
                b_std = np.sqrt(np.trace(b.covariance))

                if (distance < b_std and
                    other.confidence > b.confidence * redundancy_threshold and
                    other.salience >= b.salience):
                    to_remove.append(b.vertex_id)
                    break

        # Remove redundant beliefs using proper cleanup
        for vid in set(to_remove):
            if vid in self.graph.beliefs:
                self.graph._remove_belief(vid)
                self.pruned_beliefs += 1

        return len(set(to_remove))

    def resolve_contradictions_by_collapse(self, evidence_threshold: float = 5) -> int:
        """
        Resolve contradictions by collapsing weaker belief into stronger

        If two beliefs contradict and one has much more evidence,
        collapse the weaker into the stronger.
        """
        resolved = 0

        for (v1, v2) in list(self.graph.contradiction_pairs):
            if v1 not in self.graph.beliefs or v2 not in self.graph.beliefs:
                continue

            b1 = self.graph.beliefs[v1]
            b2 = self.graph.beliefs[v2]

            evidence_ratio = (b1.observation_count + 1) / (b2.observation_count + 1)

            if evidence_ratio > evidence_threshold:
                # b1 dominates: collapse b2 into b1
                self.merge_beliefs(v1, v2)
                resolved += 1
            elif 1 / evidence_ratio > evidence_threshold:
                # b2 dominates: collapse b1 into b2
                self.merge_beliefs(v2, v1)
                resolved += 1

        self.contradictions_resolved += resolved
        return resolved

    def compress_step(self) -> Dict[str, int]:
        """
        Single compression step: merge, prune, resolve

        Returns statistics
        """
        stats = {'merged': 0, 'pruned': 0, 'resolved': 0}

        # Find and perform merges
        mergeable = self.find_mergeable_pairs(max_pairs=50)
        merged_ids = set()

        for vid1, vid2, _ in mergeable[:10]:  # Limit merges per step
            if vid1 not in merged_ids and vid2 not in merged_ids:
                if vid1 in self.graph.beliefs and vid2 in self.graph.beliefs:
                    self.merge_beliefs(vid1, vid2)
                    merged_ids.add(vid2)
                    stats['merged'] += 1

        # Prune redundant
        stats['pruned'] = self.prune_redundant_beliefs()

        # Resolve contradictions
        stats['resolved'] = self.resolve_contradictions_by_collapse()

        return stats


# =============================================================================
# 3. ENHANCED OVERMIND WITH REAL CONTROL LAWS
# =============================================================================

@dataclass
class OvermindControlState:
    """Full control state for Overmind regulation"""

    # Learning rates
    belief_learning_rate: float = 0.1  # How fast beliefs update
    pheromone_decay_rate: float = 0.01  # How fast pheromones decay
    uncertainty_growth_rate: float = 0.01  # How fast uncertainty grows

    # Entropy bounds
    entropy_floor: float = 10.0  # Minimum system entropy (prevent collapse)
    entropy_ceiling: float = 1000.0  # Maximum entropy (prevent explosion)

    # Communication budgets
    packet_budget_per_step: int = 10  # Max packets in flight
    belief_budget: int = 500  # Max beliefs before forced compression

    # Colony controls
    beaver_activity: float = 1.0  # 0=silenced, 1=normal, 2=amplified
    ant_activity: float = 1.0
    bee_activity: float = 1.0

    # Sync parameters
    global_clock_scale: float = 1.0  # Speed up/slow down everything


class EnhancedOvermind:
    """
    Overmind with real control laws

    π_OM(Ω_t) → {Δα, Δτ, ΔB, Δsync}

    Not just noise injection, but:
    - Learning rate adjustment
    - Entropy floor/ceiling enforcement
    - Communication budget allocation
    - Colony silencing/amplification
    """

    def __init__(self, target_entropy: float = 100.0):
        self.target_entropy = target_entropy

        # Control state
        self.control = OvermindControlState()

        # State tracking
        self.history = {
            'entropy': [],
            'contradiction_mass': [],
            'packet_throughput': [],
            'compression_ratio': [],
            'control_actions': []
        }

        # PID-like control
        self.entropy_error_integral = 0.0
        self.last_entropy_error = 0.0

        # Adaptive thresholds
        self.stagnation_counter = 0
        self.explosion_counter = 0

    def observe_and_control(self,
                           energy: float,
                           epistemic_graph,
                           n_packets_delivered: int,
                           n_packets_dropped: int,
                           belief_compressor: BeliefCompressor,
                           dt: float) -> Dict[str, any]:
        """
        Main control loop: observe state and compute control actions

        Returns dict of control actions taken
        """
        actions = {}

        # Compute current state
        current_entropy = epistemic_graph.compute_total_entropy()
        contradiction_mass = epistemic_graph.compute_contradiction_mass()
        n_beliefs = len(epistemic_graph.beliefs)
        throughput = n_packets_delivered / (n_packets_delivered + n_packets_dropped + 1)

        # Track history
        self.history['entropy'].append(current_entropy)
        self.history['contradiction_mass'].append(contradiction_mass)
        self.history['packet_throughput'].append(throughput)

        # =====================================================================
        # CONTROL LAW 1: Entropy Regulation (PID-like)
        # =====================================================================
        entropy_error = current_entropy - self.target_entropy
        self.entropy_error_integral += entropy_error * dt
        entropy_error_derivative = (entropy_error - self.last_entropy_error) / dt if dt > 0 else 0
        self.last_entropy_error = entropy_error

        # PID coefficients
        Kp, Ki, Kd = 0.1, 0.01, 0.05
        entropy_control = Kp * entropy_error + Ki * self.entropy_error_integral + Kd * entropy_error_derivative

        # Adjust learning rate based on entropy error
        if entropy_error > 0:
            # Too much entropy: slow down learning, speed up compression
            self.control.belief_learning_rate = max(0.01, 0.1 - 0.01 * entropy_control)
            self.control.uncertainty_growth_rate = max(0.001, 0.01 - 0.001 * entropy_control)
            actions['slow_learning'] = True
        else:
            # Too little entropy: speed up exploration
            self.control.belief_learning_rate = min(0.5, 0.1 + 0.01 * abs(entropy_control))
            self.control.uncertainty_growth_rate = min(0.1, 0.01 + 0.001 * abs(entropy_control))
            actions['boost_exploration'] = True

        # =====================================================================
        # CONTROL LAW 2: Entropy Bounds Enforcement
        # =====================================================================
        if current_entropy < self.control.entropy_floor:
            # Below floor: inject uncertainty
            self.explosion_counter = 0
            self.stagnation_counter += 1

            if self.stagnation_counter > 10:
                # Force exploration
                self.control.ant_activity = min(2.0, self.control.ant_activity + 0.1)
                actions['amplify_ants'] = self.control.ant_activity

                # Add noise to beliefs
                for belief in list(epistemic_graph.beliefs.values())[:10]:
                    belief.increase_uncertainty(0.5)
                actions['inject_uncertainty'] = True

        elif current_entropy > self.control.entropy_ceiling:
            # Above ceiling: force compression
            self.stagnation_counter = 0
            self.explosion_counter += 1

            if self.explosion_counter > 5:
                # Force compression
                compression_stats = belief_compressor.compress_step()
                actions['forced_compression'] = compression_stats

                # Slow down ant exploration
                self.control.ant_activity = max(0.3, self.control.ant_activity - 0.1)
                actions['dampen_ants'] = self.control.ant_activity
        else:
            # In bounds: reset counters
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
            self.explosion_counter = max(0, self.explosion_counter - 1)

        # =====================================================================
        # CONTROL LAW 3: Communication Budget
        # =====================================================================
        if n_packets_dropped > n_packets_delivered:
            # Congestion: reduce packet budget
            self.control.packet_budget_per_step = max(1, self.control.packet_budget_per_step - 1)
            self.control.bee_activity = max(0.3, self.control.bee_activity - 0.1)
            actions['reduce_traffic'] = True
        elif throughput > 0.9:
            # Smooth sailing: increase budget
            self.control.packet_budget_per_step = min(50, self.control.packet_budget_per_step + 1)
            self.control.bee_activity = min(1.5, self.control.bee_activity + 0.05)
            actions['increase_traffic'] = True

        # =====================================================================
        # CONTROL LAW 4: Belief Budget
        # =====================================================================
        if n_beliefs > self.control.belief_budget:
            # Too many beliefs: force aggressive compression
            for _ in range(3):
                belief_compressor.compress_step()
            actions['aggressive_compression'] = n_beliefs - self.control.belief_budget

        # =====================================================================
        # CONTROL LAW 5: Contradiction Management
        # =====================================================================
        contradiction_rate = contradiction_mass / (n_beliefs + 1)

        if contradiction_rate > 0.2:
            # High contradiction: boost beaver infrastructure
            self.control.beaver_activity = min(2.0, self.control.beaver_activity + 0.1)
            actions['boost_infrastructure'] = True

            # Force contradiction resolution
            belief_compressor.resolve_contradictions_by_collapse()
            actions['resolve_contradictions'] = True
        elif contradiction_rate < 0.01:
            # Too little contradiction (groupthink): reduce beaver influence
            self.control.beaver_activity = max(0.5, self.control.beaver_activity - 0.05)

        # Track actions
        self.history['control_actions'].append(actions)

        return actions

    def get_control_state(self) -> OvermindControlState:
        """Get current control parameters"""
        return self.control

    def get_colony_activity(self, colony: str) -> float:
        """Get activity multiplier for a colony"""
        if colony == 'beavers':
            return self.control.beaver_activity
        elif colony == 'ants':
            return self.control.ant_activity
        elif colony == 'bees':
            return self.control.bee_activity
        return 1.0

    def get_packet_budget(self) -> int:
        """Current packet transmission budget"""
        return self.control.packet_budget_per_step

    def get_belief_learning_rate(self) -> float:
        """Current belief update learning rate"""
        return self.control.belief_learning_rate


# =============================================================================
# 4. STRUCTURAL-EPISTEMIC COUPLING
# =============================================================================

class StructuralEpistemicCoupler:
    """
    Rewards beavers for epistemically useful structures

    Reward for:
    - Structures that reduce contradiction density
    - Paths that lower transport entropy
    - Penalty for unused infrastructure
    """

    def __init__(self, epistemic_graph):
        self.graph = epistemic_graph

        # Track structure usage
        self.structure_usage: Dict[Tuple[float, float, float], int] = {}
        self.structure_epistemic_value: Dict[Tuple[float, float, float], float] = {}

    def compute_structure_reward(self,
                                position: np.ndarray,
                                nearby_beliefs: List,
                                packets_through: int) -> float:
        """
        Compute reward for building at this position

        Reward = contradiction_reduction + transport_value - unused_penalty
        """
        reward = 0.0

        # Contradiction reduction: reward if structure is between contradicting beliefs
        for b in nearby_beliefs:
            if len(b.contradicted_by) > 0:
                # Structure near contradicting belief is valuable
                reward += 0.1 * len(b.contradicted_by)

        # Transport value: reward if structure is on a useful path
        reward += 0.05 * packets_through

        # Unused penalty
        pos_key = tuple(position[:3].round(1))
        usage = self.structure_usage.get(pos_key, 0)
        if usage < 1:
            reward -= 0.02  # Penalty for unused structure

        return reward

    def record_structure_usage(self, position: np.ndarray):
        """Record that a structure was used"""
        pos_key = tuple(position[:3].round(1))
        self.structure_usage[pos_key] = self.structure_usage.get(pos_key, 0) + 1

    def compute_global_infrastructure_efficiency(self) -> float:
        """
        Overall efficiency: used structures / total structures
        """
        if not self.structure_usage:
            return 0.0

        used = sum(1 for v in self.structure_usage.values() if v > 0)
        return used / len(self.structure_usage)


# =============================================================================
# 5. TRANSPORT LEARNING LOOP
# =============================================================================

class TransportLearner:
    """
    Learning-based transport routing

    Learns from:
    - Congestion history
    - Drop penalties
    - Delivery success rates

    Outputs:
    - Routing preferences
    - Capacity predictions
    - Timing recommendations
    """

    def __init__(self, learning_rate: float = 0.1):
        self.lr = learning_rate

        # Q-values for routing decisions
        # State: (congestion_level, packet_priority)
        # Action: (route_choice)
        self.q_values: Dict[Tuple[int, int], Dict[str, float]] = {}

        # History
        self.delivery_history: List[Tuple[float, bool]] = []  # (time, success)
        self.congestion_history: List[float] = []

        # Learned parameters
        self.optimal_send_rate: float = 1.0
        self.congestion_prediction: float = 0.0

    def discretize_state(self, congestion: float, priority: float) -> Tuple[int, int]:
        """Convert continuous state to discrete for Q-learning"""
        cong_level = min(4, int(congestion * 5))
        prio_level = min(4, int(priority * 5))
        return (cong_level, prio_level)

    def get_send_probability(self, congestion: float, priority: float) -> float:
        """
        Should we send this packet now?

        Returns probability of sending based on learned Q-values
        """
        state = self.discretize_state(congestion, priority)

        if state not in self.q_values:
            # Default: send if priority > congestion
            return 1.0 if priority > congestion else 0.5

        q_send = self.q_values[state].get('send', 0.0)
        q_wait = self.q_values[state].get('wait', 0.0)

        # Softmax
        exp_send = np.exp(q_send)
        exp_wait = np.exp(q_wait)
        return exp_send / (exp_send + exp_wait + 1e-6)

    def record_outcome(self,
                      congestion_at_send: float,
                      priority: float,
                      action: str,
                      success: bool,
                      wait_time: float):
        """
        Update Q-values based on outcome

        Reward:
        - +1 for successful delivery
        - -1 for drop
        - -0.1 * wait_time for delays
        """
        state = self.discretize_state(congestion_at_send, priority)

        # Compute reward
        reward = 1.0 if success else -1.0
        reward -= 0.1 * wait_time

        # Initialize state if needed
        if state not in self.q_values:
            self.q_values[state] = {'send': 0.0, 'wait': 0.0}

        # Q-learning update
        old_q = self.q_values[state][action]
        self.q_values[state][action] = old_q + self.lr * (reward - old_q)

        # Update congestion prediction
        self.congestion_history.append(congestion_at_send)
        if len(self.congestion_history) > 100:
            self.congestion_prediction = np.mean(self.congestion_history[-20:])

        # Update optimal send rate
        if len(self.delivery_history) > 50:
            recent_success = np.mean([1 if s else 0 for _, s in self.delivery_history[-50:]])
            if recent_success < 0.5:
                self.optimal_send_rate = max(0.1, self.optimal_send_rate * 0.9)
            elif recent_success > 0.9:
                self.optimal_send_rate = min(2.0, self.optimal_send_rate * 1.1)

        self.delivery_history.append((wait_time, success))

    def get_recommended_rate(self) -> float:
        """Recommended packet sending rate"""
        return self.optimal_send_rate

    def predict_congestion(self, steps_ahead: int = 10) -> float:
        """Predict future congestion"""
        if len(self.congestion_history) < 10:
            return 0.5

        # Simple trend extrapolation
        recent = self.congestion_history[-10:]
        trend = (recent[-1] - recent[0]) / len(recent)
        return max(0, min(1, self.congestion_prediction + trend * steps_ahead))


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EPISTEMIC ENHANCEMENTS - Tier 1 Critical Features")
    print("="*80)

    print("\n1. Packet Value Scoring")
    print("   V_packet = ΔF_expected - λ·cost")
    print("   - Enables bee routing decisions")
    print("   - Enables wormhole acceptance criteria")
    print("   - Provides signal for Overmind regulation")

    print("\n2. Belief Compression (MDL)")
    print("   - Merge similar beliefs")
    print("   - Prune redundant beliefs")
    print("   - Resolve contradictions by collapse")
    print("   - Prevents entropy explosion")

    print("\n3. Enhanced Overmind Control Laws")
    print("   π_OM(Ω_t) → {Δα, Δτ, ΔB, Δsync}")
    print("   - Learning rate adjustment")
    print("   - Entropy floor/ceiling enforcement")
    print("   - Communication budgets")
    print("   - Colony silencing/amplification")

    print("\n4. Structural-Epistemic Coupling")
    print("   - Reward structures that reduce contradictions")
    print("   - Reward paths that lower transport entropy")
    print("   - Penalize unused infrastructure")

    print("\n5. Transport Learning Loop")
    print("   - Q-learning for routing decisions")
    print("   - Congestion prediction")
    print("   - Adaptive send rates")

    print("\n" + "="*80)
    print("These features close the learning loops that were missing.")
    print("="*80)
