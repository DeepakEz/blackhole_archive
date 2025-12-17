"""
MycoBeaver Communication Protocol System
==========================================
Bee-inspired consensus and distributed communication.

Based on Distributed Cognitive Architecture in Adversarial Information Environments:
- Message passing between agents
- Quorum-based consensus for decisions
- Vector clocks for causal ordering
- Bandwidth-limited communication
- Clock synchronization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq

from .config import CommunicationConfig, InfoCostConfig

# Forward reference for type hints - actual AgentState imported at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .environment import AgentState


class MessageType(Enum):
    """Types of messages agents can send"""
    # Project-related
    PROJECT_PROPOSAL = "project_proposal"
    PROJECT_VOTE = "project_vote"
    PROJECT_COMMIT = "project_commit"

    # Resource-related
    RESOURCE_FOUND = "resource_found"
    RESOURCE_DEPLETED = "resource_depleted"

    # Danger-related
    DANGER_ALERT = "danger_alert"
    DANGER_CLEARED = "danger_cleared"

    # Coordination
    HELP_REQUEST = "help_request"
    POSITION_UPDATE = "position_update"
    STATUS_UPDATE = "status_update"

    # Consensus
    CONSENSUS_PROPOSE = "consensus_propose"
    CONSENSUS_ACCEPT = "consensus_accept"
    CONSENSUS_REJECT = "consensus_reject"

    # Synchronization
    CLOCK_SYNC = "clock_sync"
    HEARTBEAT = "heartbeat"


@dataclass
class VectorClock:
    """
    Vector clock for causal ordering of events.

    Each agent maintains a vector of logical timestamps.
    """
    n_agents: int
    clock: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if len(self.clock) == 0:
            self.clock = np.zeros(self.n_agents, dtype=np.int64)

    def tick(self, agent_id: int):
        """Increment own clock component"""
        self.clock[agent_id] += 1

    def merge(self, other: 'VectorClock'):
        """Merge with another vector clock (take max)"""
        self.clock = np.maximum(self.clock, other.clock)

    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this clock happens-before other"""
        return (np.all(self.clock <= other.clock) and
                np.any(self.clock < other.clock))

    def concurrent(self, other: 'VectorClock') -> bool:
        """Check if two clocks are concurrent (neither happens-before)"""
        return not self.happens_before(other) and not other.happens_before(self)

    def copy(self) -> 'VectorClock':
        """Create a copy of this clock"""
        new_clock = VectorClock(self.n_agents)
        new_clock.clock = self.clock.copy()
        return new_clock


@dataclass
class Message:
    """Message passed between agents"""
    id: int
    sender_id: int
    msg_type: MessageType
    content: Any
    timestamp: int  # Logical timestamp
    vector_clock: VectorClock
    recipients: Optional[List[int]] = None  # None = broadcast

    # Delivery tracking
    delivered_to: Set[int] = field(default_factory=set)
    creation_step: int = 0

    # Priority (for bandwidth-limited scenarios)
    priority: int = 1


@dataclass
class ConsensusProposal:
    """Proposal for quorum-based consensus"""
    id: int
    proposer_id: int
    proposal_type: str  # e.g., "project", "role_change"
    content: Any

    # Voting state
    votes_for: Set[int] = field(default_factory=set)
    votes_against: Set[int] = field(default_factory=set)

    # Status
    decided: bool = False
    accepted: bool = False

    # Timing
    creation_step: int = 0
    deadline_step: int = 0


class CommunicationSystem:
    """
    Distributed communication system for the colony.

    Handles:
    1. Message passing with causal ordering
    2. Bandwidth-limited delivery
    3. Consensus protocols
    4. Clock synchronization

    PHASE 2: Information Thermodynamics
    ------------------------------------
    All communication actions have info costs. Agents must have sufficient
    info_energy to send messages, vote, or propose consensus.
    Silent failure: if insufficient energy, action simply doesn't happen.
    """

    def __init__(self, config: CommunicationConfig, n_agents: int,
                 info_costs: Optional[InfoCostConfig] = None):
        self.config = config
        self.n_agents = n_agents

        # PHASE 2: Info cost configuration
        self.info_costs = info_costs or InfoCostConfig()

        # Message storage
        self.pending_messages: List[Message] = []
        self.delivered_messages: List[Message] = []
        self.message_id_counter = 0

        # Per-agent message queues
        self.agent_queues: Dict[int, List[Message]] = {
            i: [] for i in range(n_agents)
        }

        # Vector clocks
        self.agent_clocks: Dict[int, VectorClock] = {
            i: VectorClock(n_agents) for i in range(n_agents)
        }

        # Consensus proposals
        self.active_proposals: Dict[int, ConsensusProposal] = {}
        self.proposal_id_counter = 0

        # Synchronization state
        self.global_time = 0
        self.agent_local_times: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.sync_offsets: Dict[int, float] = {i: 0.0 for i in range(n_agents)}

        # Statistics
        self.total_messages_sent = 0
        self.total_messages_delivered = 0

        # PHASE 2: Info dissipation tracking (for Overmind observation)
        self.info_spent_this_step = 0.0
        self.messages_blocked_by_info = 0

    def send_message(self, sender_id: int, msg_type: MessageType,
                     content: Any, recipients: Optional[List[int]] = None,
                     priority: int = 1,
                     agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """
        Send a message from one agent.

        PHASE 2: Info Cost Integration
        ------------------------------
        If agent_state is provided, checks and deducts info_energy.
        Silent failure: returns None if insufficient energy.

        Args:
            sender_id: Sending agent ID
            msg_type: Type of message
            content: Message content
            recipients: Specific recipients (None = broadcast)
            priority: Message priority
            agent_state: Optional agent state for info cost checking

        Returns:
            Message ID, or None if action was blocked by info cost
        """
        # PHASE 2: Info cost check
        if agent_state is not None:
            cost = self.info_costs.cost_send_message
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                # Silent failure - agent doesn't have enough info energy
                self.messages_blocked_by_info += 1
                return None
            # Deduct the cost
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost

        # Update sender's clock
        self.agent_clocks[sender_id].tick(sender_id)

        msg = Message(
            id=self.message_id_counter,
            sender_id=sender_id,
            msg_type=msg_type,
            content=content,
            timestamp=int(self.agent_clocks[sender_id].clock[sender_id]),
            vector_clock=self.agent_clocks[sender_id].copy(),
            recipients=recipients,
            priority=priority,
            creation_step=self.global_time,
        )

        self.pending_messages.append(msg)
        self.message_id_counter += 1
        self.total_messages_sent += 1

        return msg.id

    def broadcast(self, sender_id: int, msg_type: MessageType,
                  content: Any, priority: int = 1,
                  agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """
        Broadcast message to all agents.

        PHASE 2: Broadcasts have higher info cost than targeted messages.
        """
        # PHASE 2: Use broadcast cost instead of send cost
        if agent_state is not None:
            cost = self.info_costs.cost_broadcast  # Higher cost for broadcasts
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                self.messages_blocked_by_info += 1
                return None
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost
            # Pass None for agent_state to avoid double-charging
            return self.send_message(sender_id, msg_type, content, None, priority, None)

        return self.send_message(sender_id, msg_type, content, None, priority, None)

    def receive_messages(self, agent_id: int) -> List[Message]:
        """
        Receive pending messages for an agent.

        Returns messages in causal order.
        """
        messages = self.agent_queues[agent_id].copy()
        self.agent_queues[agent_id].clear()

        # Sort by vector clock (causal order)
        messages.sort(key=lambda m: np.sum(m.vector_clock.clock))

        # Update agent's clock
        for msg in messages:
            self.agent_clocks[agent_id].merge(msg.vector_clock)
            self.agent_clocks[agent_id].tick(agent_id)

        return messages

    def process_deliveries(self, current_step: int):
        """
        Process message deliveries for current step.

        Respects bandwidth limits.
        """
        self.global_time = current_step

        # Sort pending messages by priority
        self.pending_messages.sort(key=lambda m: -m.priority)

        # Track bandwidth usage
        deliveries_this_step = 0
        remaining_messages = []

        for msg in self.pending_messages:
            if deliveries_this_step >= self.config.max_bandwidth_per_step:
                remaining_messages.append(msg)
                continue

            # Determine recipients
            if msg.recipients is None:
                targets = list(range(self.n_agents))
            else:
                targets = msg.recipients

            # Deliver to each target
            for target_id in targets:
                if target_id == msg.sender_id:
                    continue  # Don't send to self

                if target_id in msg.delivered_to:
                    continue  # Already delivered

                if deliveries_this_step >= self.config.max_bandwidth_per_step:
                    break

                # Check queue capacity
                if len(self.agent_queues[target_id]) < self.config.max_message_queue:
                    self.agent_queues[target_id].append(msg)
                    msg.delivered_to.add(target_id)
                    deliveries_this_step += 1
                    self.total_messages_delivered += 1

            # Check if fully delivered
            if msg.recipients is None:
                all_delivered = len(msg.delivered_to) >= self.n_agents - 1
            else:
                all_delivered = all(r in msg.delivered_to or r == msg.sender_id
                                   for r in msg.recipients)

            if not all_delivered:
                remaining_messages.append(msg)
            else:
                self.delivered_messages.append(msg)

        self.pending_messages = remaining_messages

        # Clean old delivered messages
        if len(self.delivered_messages) > 1000:
            self.delivered_messages = self.delivered_messages[-500:]

    def propose_consensus(self, proposer_id: int, proposal_type: str,
                          content: Any, deadline_steps: int = 100,
                          agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """
        Propose a consensus decision.

        PHASE 2: Proposals have high info cost (cost_project_proposal).

        Args:
            proposer_id: ID of proposing agent
            proposal_type: Type of proposal
            content: Proposal content
            deadline_steps: Steps until deadline
            agent_state: Optional agent state for info cost checking

        Returns:
            Proposal ID, or None if blocked by info cost
        """
        # PHASE 2: Check info cost for proposals (high cost)
        if agent_state is not None:
            cost = self.info_costs.cost_project_proposal
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                self.messages_blocked_by_info += 1
                return None
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost

        proposal = ConsensusProposal(
            id=self.proposal_id_counter,
            proposer_id=proposer_id,
            proposal_type=proposal_type,
            content=content,
            creation_step=self.global_time,
            deadline_step=self.global_time + deadline_steps,
        )

        # Proposer automatically votes for
        proposal.votes_for.add(proposer_id)

        self.active_proposals[proposal.id] = proposal
        self.proposal_id_counter += 1

        # Broadcast proposal (no extra charge - already paid the proposal cost)
        self.broadcast(
            proposer_id,
            MessageType.CONSENSUS_PROPOSE,
            {"proposal_id": proposal.id, "content": content},
            priority=2,
            agent_state=None  # No double-charge
        )

        return proposal.id

    def vote_on_proposal(self, voter_id: int, proposal_id: int,
                         vote: bool,
                         agent_state: Optional['AgentState'] = None) -> bool:
        """
        Vote on a consensus proposal.

        PHASE 2: Voting has info cost (cost_consensus_vote).

        Args:
            voter_id: Voting agent ID
            proposal_id: Proposal to vote on
            vote: True = accept, False = reject
            agent_state: Optional agent state for info cost checking

        Returns:
            Whether vote was recorded (False if blocked by info cost)
        """
        if proposal_id not in self.active_proposals:
            return False

        proposal = self.active_proposals[proposal_id]

        if proposal.decided:
            return False

        # PHASE 2: Check info cost for voting
        if agent_state is not None:
            cost = self.info_costs.cost_consensus_vote
            if not agent_state.can_afford_info(cost, self.info_costs.min_info_for_action):
                self.messages_blocked_by_info += 1
                return False
            agent_state.spend_info(cost)
            self.info_spent_this_step += cost

        # Remove from opposite set if changing vote
        if vote:
            proposal.votes_against.discard(voter_id)
            proposal.votes_for.add(voter_id)
            msg_type = MessageType.CONSENSUS_ACCEPT
        else:
            proposal.votes_for.discard(voter_id)
            proposal.votes_against.add(voter_id)
            msg_type = MessageType.CONSENSUS_REJECT

        # Broadcast vote (no extra charge - already paid voting cost)
        self.broadcast(voter_id, msg_type, {"proposal_id": proposal_id}, agent_state=None)

        return True

    def check_consensus(self, proposal_id: int) -> Optional[bool]:
        """
        Check if consensus has been reached.

        Returns:
            True if accepted, False if rejected, None if undecided
        """
        if proposal_id not in self.active_proposals:
            return None

        proposal = self.active_proposals[proposal_id]

        if proposal.decided:
            return proposal.accepted

        n_for = len(proposal.votes_for)
        n_against = len(proposal.votes_against)

        # Check quorum
        quorum_size = int(self.n_agents * self.config.quorum_fraction)

        if n_for >= quorum_size:
            proposal.decided = True
            proposal.accepted = True
            return True

        if n_against > self.n_agents - quorum_size:
            proposal.decided = True
            proposal.accepted = False
            return False

        # Check timeout
        if self.global_time >= proposal.deadline_step:
            proposal.decided = True
            proposal.accepted = n_for > n_against
            return proposal.accepted

        return None

    def update_proposals(self):
        """Update all active proposals"""
        to_remove = []

        for proposal_id, proposal in self.active_proposals.items():
            result = self.check_consensus(proposal_id)

            if result is not None:
                # Broadcast decision
                content = {
                    "proposal_id": proposal_id,
                    "accepted": result,
                    "votes_for": len(proposal.votes_for),
                    "votes_against": len(proposal.votes_against),
                }
                self.broadcast(
                    proposal.proposer_id,
                    MessageType.PROJECT_COMMIT if result else MessageType.CONSENSUS_REJECT,
                    content,
                    priority=3
                )

                # Keep decided proposals for a while
                if self.global_time > proposal.deadline_step + 50:
                    to_remove.append(proposal_id)

        for pid in to_remove:
            del self.active_proposals[pid]

    def sync_clocks(self, current_step: int):
        """
        Synchronize agent clocks.

        Implements simple clock synchronization protocol.
        """
        if current_step % self.config.sync_interval != 0:
            return

        # Master agent (ID 0) broadcasts time
        master_time = self.global_time

        self.broadcast(0, MessageType.CLOCK_SYNC, {"time": master_time}, priority=3)

        # Agents adjust their local times
        for agent_id in range(self.n_agents):
            # Add some drift
            drift = np.random.uniform(
                -self.config.max_clock_drift,
                self.config.max_clock_drift
            )
            self.agent_local_times[agent_id] = master_time + drift
            self.sync_offsets[agent_id] = -drift

    def update(self, current_step: int):
        """
        Full update for communication system.

        1. Process deliveries
        2. Update proposals
        3. Sync clocks (periodically)
        """
        self.process_deliveries(current_step)
        self.update_proposals()
        self.sync_clocks(current_step)

    def get_agent_stats(self, agent_id: int) -> Dict[str, Any]:
        """Get statistics for a specific agent"""
        queue_size = len(self.agent_queues[agent_id])
        clock = self.agent_clocks[agent_id].clock.copy()

        return {
            "queue_size": queue_size,
            "vector_clock": clock.tolist(),
            "local_time": self.agent_local_times[agent_id],
            "sync_offset": self.sync_offsets[agent_id],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get global communication statistics"""
        return {
            "total_sent": self.total_messages_sent,
            "total_delivered": self.total_messages_delivered,
            "pending_messages": len(self.pending_messages),
            "active_proposals": len(self.active_proposals),
            "avg_queue_size": np.mean([len(q) for q in self.agent_queues.values()]),
            # PHASE 2: Info dissipation metrics
            "info_spent_this_step": self.info_spent_this_step,
            "messages_blocked_by_info": self.messages_blocked_by_info,
        }

    def get_info_dissipation(self) -> float:
        """
        PHASE 2: Get info energy spent this step.

        Used by Overmind to observe global info dissipation rate.
        """
        return self.info_spent_this_step

    def reset_step_tracking(self):
        """
        PHASE 2: Reset per-step tracking variables.

        Call at the beginning of each step.
        """
        self.info_spent_this_step = 0.0
        self.messages_blocked_by_info = 0

    def reset(self):
        """Reset communication system"""
        self.pending_messages.clear()
        self.delivered_messages.clear()
        self.message_id_counter = 0

        for agent_id in range(self.n_agents):
            self.agent_queues[agent_id].clear()
            self.agent_clocks[agent_id] = VectorClock(self.n_agents)
            self.agent_local_times[agent_id] = 0.0
            self.sync_offsets[agent_id] = 0.0

        self.active_proposals.clear()
        self.proposal_id_counter = 0
        self.global_time = 0
        self.total_messages_sent = 0
        self.total_messages_delivered = 0

        # PHASE 2: Reset info tracking
        self.info_spent_this_step = 0.0
        self.messages_blocked_by_info = 0


class ColonyCommunicationHub:
    """
    High-level communication hub for the colony.

    Provides convenient methods for common communication patterns.

    PHASE 2: All convenience methods now accept optional agent_state
    for info cost checking.
    """

    def __init__(self, config: CommunicationConfig, n_agents: int,
                 info_costs: Optional[InfoCostConfig] = None):
        self.system = CommunicationSystem(config, n_agents, info_costs)
        self.n_agents = n_agents

        # Message handlers registered by agents
        self.message_handlers: Dict[int, Dict[MessageType, callable]] = {}

    def register_handler(self, agent_id: int, msg_type: MessageType,
                         handler: callable):
        """Register a message handler for an agent"""
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = {}
        self.message_handlers[agent_id][msg_type] = handler

    def alert_danger(self, sender_id: int, position: Tuple[int, int],
                     danger_type: str = "flood",
                     agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """
        Broadcast danger alert.

        PHASE 2: High priority messages still have info cost.
        """
        return self.system.broadcast(
            sender_id,
            MessageType.DANGER_ALERT,
            {"position": position, "type": danger_type},
            priority=5,
            agent_state=agent_state
        )

    def announce_resource(self, sender_id: int, position: Tuple[int, int],
                          resource_type: str, amount: float,
                          agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """Announce resource discovery"""
        return self.system.broadcast(
            sender_id,
            MessageType.RESOURCE_FOUND,
            {"position": position, "type": resource_type, "amount": amount},
            priority=2,
            agent_state=agent_state
        )

    def request_help(self, sender_id: int, position: Tuple[int, int],
                     task: str,
                     agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """Request help from nearby agents"""
        return self.system.broadcast(
            sender_id,
            MessageType.HELP_REQUEST,
            {"position": position, "task": task},
            priority=3,
            agent_state=agent_state
        )

    def propose_project(self, sender_id: int, project_type: str,
                        location: Tuple[int, int],
                        agent_state: Optional['AgentState'] = None) -> Optional[int]:
        """
        Propose a new project for consensus.

        PHASE 2: Project proposals have high info cost.
        """
        return self.system.propose_consensus(
            sender_id,
            "project",
            {"type": project_type, "location": location},
            deadline_steps=100,
            agent_state=agent_state
        )

    def process_agent_messages(self, agent_id: int) -> List[Any]:
        """
        Process messages for an agent using registered handlers.

        Returns list of handler results.
        """
        messages = self.system.receive_messages(agent_id)
        results = []

        handlers = self.message_handlers.get(agent_id, {})

        for msg in messages:
            if msg.msg_type in handlers:
                result = handlers[msg.msg_type](msg)
                results.append(result)

        return results

    def update(self, current_step: int):
        """Update the communication hub"""
        self.system.update(current_step)

    def get_info_dissipation(self) -> float:
        """PHASE 2: Get info energy spent this step."""
        return self.system.get_info_dissipation()

    def reset_step_tracking(self):
        """PHASE 2: Reset per-step tracking."""
        self.system.reset_step_tracking()

    def reset(self):
        """Reset the hub"""
        self.system.reset()
        self.message_handlers.clear()
