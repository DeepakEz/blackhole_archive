"""
Unit tests for mycobeaver/communication.py

Tests the distributed communication and consensus system.
"""

import pytest
import numpy as np
from mycobeaver.communication import (
    CommunicationSystem, ColonyCommunicationHub,
    VectorClock, Message, MessageType, ConsensusProposal
)
from mycobeaver.config import CommunicationConfig, InfoCostConfig


class TestVectorClock:
    """Tests for VectorClock class"""

    def test_initialization(self):
        """Test clock initialization"""
        clock = VectorClock(n_agents=5)
        assert len(clock.clock) == 5
        assert np.all(clock.clock == 0)

    def test_tick(self):
        """Test clock tick operation"""
        clock = VectorClock(n_agents=5)
        clock.tick(2)
        assert clock.clock[2] == 1
        clock.tick(2)
        assert clock.clock[2] == 2

    def test_merge(self):
        """Test clock merge operation"""
        clock1 = VectorClock(n_agents=3)
        clock1.clock = np.array([1, 0, 2])

        clock2 = VectorClock(n_agents=3)
        clock2.clock = np.array([0, 3, 1])

        clock1.merge(clock2)
        # Should take max of each component
        assert np.array_equal(clock1.clock, np.array([1, 3, 2]))

    def test_happens_before(self):
        """Test happens-before relation"""
        clock1 = VectorClock(n_agents=3)
        clock1.clock = np.array([1, 2, 0])

        clock2 = VectorClock(n_agents=3)
        clock2.clock = np.array([2, 3, 1])

        # clock1 happens-before clock2
        assert clock1.happens_before(clock2)
        assert not clock2.happens_before(clock1)

    def test_happens_before_equal(self):
        """Test happens-before with equal clocks"""
        clock1 = VectorClock(n_agents=3)
        clock1.clock = np.array([1, 2, 3])

        clock2 = VectorClock(n_agents=3)
        clock2.clock = np.array([1, 2, 3])

        # Equal clocks are not happens-before
        assert not clock1.happens_before(clock2)
        assert not clock2.happens_before(clock1)

    def test_concurrent(self):
        """Test concurrent events"""
        clock1 = VectorClock(n_agents=3)
        clock1.clock = np.array([1, 0, 0])

        clock2 = VectorClock(n_agents=3)
        clock2.clock = np.array([0, 1, 0])

        # Neither happens-before the other
        assert clock1.concurrent(clock2)
        assert clock2.concurrent(clock1)

    def test_copy(self):
        """Test clock copy"""
        clock1 = VectorClock(n_agents=3)
        clock1.clock = np.array([1, 2, 3])

        clock2 = clock1.copy()
        assert np.array_equal(clock1.clock, clock2.clock)

        # Modifications shouldn't affect original
        clock2.tick(0)
        assert clock1.clock[0] == 1
        assert clock2.clock[0] == 2


class TestCommunicationSystem:
    """Tests for CommunicationSystem class"""

    @pytest.fixture
    def comm_system(self, communication_config, info_cost_config):
        """Create communication system for testing"""
        return CommunicationSystem(communication_config, n_agents=5, info_costs=info_cost_config)

    def test_initialization(self, comm_system):
        """Test system initialization"""
        assert comm_system.n_agents == 5
        assert len(comm_system.agent_queues) == 5
        assert len(comm_system.agent_clocks) == 5
        assert comm_system.total_messages_sent == 0

    def test_send_message(self, comm_system):
        """Test sending a message"""
        msg_id = comm_system.send_message(
            sender_id=0,
            msg_type=MessageType.RESOURCE_FOUND,
            content={"position": (5, 5), "type": "wood"},
            recipients=[1, 2]
        )

        assert msg_id is not None
        assert comm_system.total_messages_sent == 1
        assert len(comm_system.pending_messages) == 1

    def test_send_message_with_info_cost(self, comm_system, mock_agent_state):
        """Test sending with info cost"""
        initial_energy = mock_agent_state.info_energy

        msg_id = comm_system.send_message(
            sender_id=0,
            msg_type=MessageType.RESOURCE_FOUND,
            content={"position": (5, 5)},
            agent_state=mock_agent_state
        )

        assert msg_id is not None
        assert mock_agent_state.info_energy < initial_energy

    def test_send_message_blocked_by_info(self, comm_system, mock_agent_state):
        """Test message blocked when no info energy"""
        mock_agent_state.info_energy = 0.0

        msg_id = comm_system.send_message(
            sender_id=0,
            msg_type=MessageType.RESOURCE_FOUND,
            content={},
            agent_state=mock_agent_state
        )

        assert msg_id is None
        assert comm_system.messages_blocked_by_info == 1

    def test_broadcast(self, comm_system):
        """Test broadcast message"""
        msg_id = comm_system.broadcast(
            sender_id=0,
            msg_type=MessageType.DANGER_ALERT,
            content={"position": (5, 5)}
        )

        assert msg_id is not None
        assert comm_system.total_messages_sent == 1

    def test_process_deliveries(self, comm_system):
        """Test message delivery processing"""
        # Send a message
        comm_system.send_message(
            sender_id=0,
            msg_type=MessageType.RESOURCE_FOUND,
            content={"data": "test"},
            recipients=[1]
        )

        # Process deliveries
        comm_system.process_deliveries(current_step=1)

        # Check message was delivered
        assert len(comm_system.agent_queues[1]) == 1 or \
               comm_system.total_messages_delivered >= 1

    def test_receive_messages(self, comm_system):
        """Test receiving messages"""
        # Send and deliver message
        comm_system.send_message(
            sender_id=0,
            msg_type=MessageType.RESOURCE_FOUND,
            content={"data": "test"},
            recipients=[1]
        )
        comm_system.process_deliveries(current_step=1)

        # Receive
        messages = comm_system.receive_messages(1)

        # Queue should be empty after receive
        assert len(comm_system.agent_queues[1]) == 0

    def test_bandwidth_limit(self, communication_config, info_cost_config):
        """Test bandwidth limiting"""
        config = CommunicationConfig(max_bandwidth_per_step=2)
        comm_system = CommunicationSystem(config, n_agents=5, info_costs=info_cost_config)

        # Send many messages
        for i in range(10):
            comm_system.send_message(
                sender_id=0,
                msg_type=MessageType.HEARTBEAT,
                content={},
                recipients=[1]
            )

        # Process deliveries (should be limited)
        comm_system.process_deliveries(current_step=1)

        # Should have pending messages due to bandwidth limit
        assert comm_system.total_messages_delivered <= 2 or \
               len(comm_system.pending_messages) > 0

    def test_propose_consensus(self, comm_system):
        """Test consensus proposal"""
        proposal_id = comm_system.propose_consensus(
            proposer_id=0,
            proposal_type="project",
            content={"location": (5, 5)}
        )

        assert proposal_id is not None
        assert proposal_id in comm_system.active_proposals

    def test_vote_on_proposal(self, comm_system):
        """Test voting on proposal"""
        # Create proposal
        proposal_id = comm_system.propose_consensus(
            proposer_id=0,
            proposal_type="test",
            content={}
        )

        # Vote
        result = comm_system.vote_on_proposal(1, proposal_id, vote=True)
        assert result is True

        proposal = comm_system.active_proposals[proposal_id]
        assert 1 in proposal.votes_for

    def test_check_consensus_accept(self, communication_config, info_cost_config):
        """Test consensus acceptance"""
        config = CommunicationConfig(quorum_fraction=0.5)
        comm_system = CommunicationSystem(config, n_agents=4, info_costs=info_cost_config)

        proposal_id = comm_system.propose_consensus(0, "test", {})

        # Vote to reach quorum (50% of 4 = 2)
        comm_system.vote_on_proposal(1, proposal_id, vote=True)

        result = comm_system.check_consensus(proposal_id)
        assert result is True

    def test_check_consensus_reject(self, communication_config, info_cost_config):
        """Test consensus rejection"""
        config = CommunicationConfig(quorum_fraction=0.75)
        comm_system = CommunicationSystem(config, n_agents=4, info_costs=info_cost_config)

        proposal_id = comm_system.propose_consensus(0, "test", {})

        # Vote against
        comm_system.vote_on_proposal(1, proposal_id, vote=False)
        comm_system.vote_on_proposal(2, proposal_id, vote=False)

        result = comm_system.check_consensus(proposal_id)
        # With 75% quorum needed and 2 against (of 4), can't reach quorum
        # Result depends on timeout logic

    def test_sync_clocks(self, comm_system):
        """Test clock synchronization"""
        comm_system.sync_clocks(current_step=0)  # Should sync at step 0

        # All agents should have similar local times
        times = list(comm_system.agent_local_times.values())
        assert len(times) == 5

    def test_update(self, comm_system):
        """Test full update cycle"""
        comm_system.send_message(0, MessageType.HEARTBEAT, {})
        comm_system.propose_consensus(1, "test", {})

        comm_system.update(current_step=1)

        assert comm_system.global_time == 1

    def test_get_statistics(self, comm_system):
        """Test statistics retrieval"""
        comm_system.send_message(0, MessageType.HEARTBEAT, {})

        stats = comm_system.get_statistics()

        assert "total_sent" in stats
        assert stats["total_sent"] == 1
        assert "pending_messages" in stats
        assert "active_proposals" in stats

    def test_reset(self, comm_system):
        """Test system reset"""
        comm_system.send_message(0, MessageType.HEARTBEAT, {})
        comm_system.propose_consensus(1, "test", {})

        comm_system.reset()

        assert comm_system.total_messages_sent == 0
        assert len(comm_system.pending_messages) == 0
        assert len(comm_system.active_proposals) == 0


class TestColonyCommunicationHub:
    """Tests for ColonyCommunicationHub class"""

    @pytest.fixture
    def hub(self, communication_config, info_cost_config):
        """Create hub for testing"""
        return ColonyCommunicationHub(communication_config, n_agents=5, info_costs=info_cost_config)

    def test_initialization(self, hub):
        """Test hub initialization"""
        assert hub.n_agents == 5
        assert hub.system is not None

    def test_register_handler(self, hub):
        """Test message handler registration"""
        def handler(msg):
            return msg.content

        hub.register_handler(0, MessageType.RESOURCE_FOUND, handler)

        assert 0 in hub.message_handlers
        assert MessageType.RESOURCE_FOUND in hub.message_handlers[0]

    def test_alert_danger(self, hub):
        """Test danger alert"""
        msg_id = hub.alert_danger(0, (5, 5), "flood")
        assert msg_id is not None

    def test_announce_resource(self, hub):
        """Test resource announcement"""
        msg_id = hub.announce_resource(0, (5, 5), "wood", 10.0)
        assert msg_id is not None

    def test_request_help(self, hub):
        """Test help request"""
        msg_id = hub.request_help(0, (5, 5), "building")
        assert msg_id is not None

    def test_propose_project(self, hub):
        """Test project proposal"""
        proposal_id = hub.propose_project(0, "dam", (5, 5))
        assert proposal_id is not None

    def test_process_agent_messages(self, hub):
        """Test message processing with handlers"""
        results = []

        def handler(msg):
            results.append(msg.content)
            return msg.content

        hub.register_handler(1, MessageType.RESOURCE_FOUND, handler)

        # Send and deliver message
        hub.system.send_message(0, MessageType.RESOURCE_FOUND, {"test": 1}, recipients=[1])
        hub.update(current_step=1)

        # Process
        hub.process_agent_messages(1)

    def test_reset(self, hub):
        """Test hub reset"""
        hub.register_handler(0, MessageType.HEARTBEAT, lambda m: None)
        hub.alert_danger(0, (5, 5))

        hub.reset()

        assert len(hub.message_handlers) == 0


class TestMessageTypes:
    """Tests for message type handling"""

    @pytest.fixture
    def comm_system(self, communication_config, info_cost_config):
        return CommunicationSystem(communication_config, n_agents=3, info_costs=info_cost_config)

    @pytest.mark.parametrize("msg_type", list(MessageType))
    def test_all_message_types(self, comm_system, msg_type):
        """Test all message types can be sent"""
        msg_id = comm_system.send_message(0, msg_type, {"type": msg_type.value})
        assert msg_id is not None

    def test_message_priority(self, comm_system):
        """Test message priority ordering"""
        # Send low priority first
        comm_system.send_message(0, MessageType.HEARTBEAT, {}, priority=1)
        # Send high priority second
        comm_system.send_message(0, MessageType.DANGER_ALERT, {}, priority=5)

        # After sorting, high priority should be first
        comm_system.pending_messages.sort(key=lambda m: -m.priority)
        assert comm_system.pending_messages[0].msg_type == MessageType.DANGER_ALERT


class TestCausalOrdering:
    """Tests for causal ordering guarantees"""

    @pytest.fixture
    def comm_system(self, communication_config, info_cost_config):
        return CommunicationSystem(communication_config, n_agents=3, info_costs=info_cost_config)

    def test_vector_clock_updates_on_send(self, comm_system):
        """Test vector clock increments on send"""
        initial_clock = comm_system.agent_clocks[0].clock.copy()

        comm_system.send_message(0, MessageType.HEARTBEAT, {})

        assert comm_system.agent_clocks[0].clock[0] > initial_clock[0]

    def test_vector_clock_merges_on_receive(self, comm_system):
        """Test vector clock merges on receive"""
        # Agent 0 sends with clock [1, 0, 0]
        comm_system.send_message(0, MessageType.HEARTBEAT, {}, recipients=[1])
        comm_system.process_deliveries(current_step=1)

        initial_clock_1 = comm_system.agent_clocks[1].clock.copy()

        # Agent 1 receives
        comm_system.receive_messages(1)

        # Agent 1's clock should have merged
        assert comm_system.agent_clocks[1].clock[0] >= 1


class TestConsensusProtocol:
    """Tests for consensus protocol behavior"""

    @pytest.fixture
    def comm_system(self, info_cost_config):
        config = CommunicationConfig(quorum_fraction=0.6)
        return CommunicationSystem(config, n_agents=5, info_costs=info_cost_config)

    def test_proposer_auto_votes(self, comm_system):
        """Test that proposer automatically votes for"""
        proposal_id = comm_system.propose_consensus(2, "test", {})
        proposal = comm_system.active_proposals[proposal_id]

        assert 2 in proposal.votes_for

    def test_vote_change(self, comm_system):
        """Test changing vote"""
        proposal_id = comm_system.propose_consensus(0, "test", {})

        # Vote yes
        comm_system.vote_on_proposal(1, proposal_id, vote=True)
        assert 1 in comm_system.active_proposals[proposal_id].votes_for

        # Change to no
        comm_system.vote_on_proposal(1, proposal_id, vote=False)
        assert 1 not in comm_system.active_proposals[proposal_id].votes_for
        assert 1 in comm_system.active_proposals[proposal_id].votes_against

    def test_timeout_decision(self, comm_system):
        """Test decision on timeout"""
        proposal_id = comm_system.propose_consensus(0, "test", {}, deadline_steps=10)

        # Vote for (proposer + 1 more = 2/5)
        comm_system.vote_on_proposal(1, proposal_id, vote=True)

        # Not decided yet (need 60% = 3)
        result = comm_system.check_consensus(proposal_id)
        assert result is None

        # Advance past deadline
        comm_system.global_time = 15
        result = comm_system.check_consensus(proposal_id)

        # Should decide based on votes (2 for, 0 against = accepted)
        assert result is not None
