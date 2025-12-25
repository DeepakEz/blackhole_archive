# Blackhole Archive Project: Protocol Specifications
# Complete Communication and Coordination Protocols

"""
PROTOCOL ARCHITECTURE OVERVIEW

The Blackhole Archive system requires coordinated communication between:
1. Colony agents (Beavers, Ants, Bees) within the same colony
2. Cross-colony coordination (Beaver-Ant, Ant-Bee, Bee-Beaver)
3. Wormhole transport layer
4. External archive interface

This document specifies all protocols in detail, including:
- Message formats
- Packet structures
- Synchronization mechanisms
- Error correction codes
- Causal ordering
- Bandwidth management
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum
import hashlib
import json
from datetime import datetime
import uuid

# =============================================================================
# 1. MESSAGE LAYER (Inter-Agent Communication)
# =============================================================================

class MessageType(Enum):
    """Types of inter-agent messages"""
    # Beaver messages
    STRUCTURAL_UPDATE = "structural_update"
    STABILITY_REPORT = "stability_report"
    SCAFFOLD_REQUEST = "scaffold_request"
    
    # Ant messages
    GRAPH_UPDATE = "graph_update"
    PHEROMONE_BROADCAST = "pheromone_broadcast"
    VERTEX_DISCOVERY = "vertex_discovery"
    EDGE_WEIGHT_UPDATE = "edge_weight_update"
    CONSENSUS_REQUEST = "consensus_request"
    
    # Bee messages
    WAGGLE_DANCE = "waggle_dance"
    SCOUT_REPORT = "scout_report"
    PACKET_NOTIFICATION = "packet_notification"
    CONGESTION_ALERT = "congestion_alert"
    
    # Cross-colony
    ROUTING_INFO = "routing_info"
    CAPACITY_QUERY = "capacity_query"
    SYNCHRONIZATION = "synchronization"

@dataclass
class MessageHeader:
    """
    Header for all inter-agent messages
    
    Ensures causal ordering and authentication
    """
    message_id: str
    sender_id: str
    receiver_id: str
    timestamp: float  # Sender's proper time
    message_type: MessageType
    hop_count: int = 0
    ttl: int = 10  # Time-to-live
    signature: str = ""  # HMAC for authentication
    
    def serialize(self) -> bytes:
        """Serialize header to bytes"""
        data = {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'timestamp': self.timestamp,
            'message_type': self.message_type.value,
            'hop_count': self.hop_count,
            'ttl': self.ttl
        }
        return json.dumps(data).encode('utf-8')
    
    def sign(self, secret_key: bytes):
        """Generate HMAC signature"""
        h = hashlib.sha256()
        h.update(secret_key)
        h.update(self.serialize())
        self.signature = h.hexdigest()
    
    def verify(self, secret_key: bytes) -> bool:
        """Verify message authenticity"""
        h = hashlib.sha256()
        h.update(secret_key)
        h.update(self.serialize())
        return h.hexdigest() == self.signature

@dataclass
class Message:
    """
    Complete message structure
    """
    header: MessageHeader
    payload: Dict[str, Any]
    
    @classmethod
    def create(cls, 
               sender_id: str,
               receiver_id: str,
               message_type: MessageType,
               payload: Dict[str, Any],
               timestamp: float) -> 'Message':
        """Factory method to create message"""
        header = MessageHeader(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestamp=timestamp,
            message_type=message_type
        )
        return cls(header=header, payload=payload)
    
    def serialize(self) -> bytes:
        """Serialize entire message"""
        header_bytes = self.header.serialize()
        payload_bytes = json.dumps(self.payload).encode('utf-8')
        
        # Format: [header_length (4 bytes)][header][payload]
        header_len = len(header_bytes).to_bytes(4, 'big')
        return header_len + header_bytes + payload_bytes
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'Message':
        """Deserialize message from bytes"""
        # Extract header length
        header_len = int.from_bytes(data[:4], 'big')
        
        # Extract header
        header_bytes = data[4:4+header_len]
        header_dict = json.loads(header_bytes.decode('utf-8'))
        header = MessageHeader(
            message_id=header_dict['message_id'],
            sender_id=header_dict['sender_id'],
            receiver_id=header_dict['receiver_id'],
            timestamp=header_dict['timestamp'],
            message_type=MessageType(header_dict['message_type']),
            hop_count=header_dict['hop_count'],
            ttl=header_dict['ttl']
        )
        
        # Extract payload
        payload_bytes = data[4+header_len:]
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        return cls(header=header, payload=payload)

# =============================================================================
# 2. PACKET LAYER (Wormhole Transport)
# =============================================================================

class PacketType(Enum):
    """Types of wormhole packets"""
    DATA = "data"
    CONTROL = "control"
    ACKNOWLEDGMENT = "ack"
    ERROR = "error"

@dataclass
class CausalCertificate:
    """
    Encodes causal relationships between events
    
    Uses vector clocks for partial ordering
    """
    vector_clock: Dict[str, int]  # agent_id -> clock value
    happened_before: List[Tuple[str, str]]  # (event_i, event_j) where i -> j
    
    def __init__(self):
        self.vector_clock = {}
        self.happened_before = []
    
    def increment(self, agent_id: str):
        """Increment clock for agent"""
        if agent_id not in self.vector_clock:
            self.vector_clock[agent_id] = 0
        self.vector_clock[agent_id] += 1
    
    def merge(self, other: 'CausalCertificate'):
        """Merge two causal certificates"""
        for agent_id, clock_val in other.vector_clock.items():
            if agent_id not in self.vector_clock:
                self.vector_clock[agent_id] = 0
            self.vector_clock[agent_id] = max(self.vector_clock[agent_id], clock_val)
        
        self.happened_before.extend(other.happened_before)
    
    def causally_precedes(self, other: 'CausalCertificate') -> bool:
        """Check if self causally precedes other"""
        # self -> other iff for all agents, self.clock[a] <= other.clock[a]
        # and exists agent a such that self.clock[a] < other.clock[a]
        
        strictly_less = False
        for agent_id, self_clock in self.vector_clock.items():
            other_clock = other.vector_clock.get(agent_id, 0)
            
            if self_clock > other_clock:
                return False
            elif self_clock < other_clock:
                strictly_less = True
        
        return strictly_less
    
    def concurrent_with(self, other: 'CausalCertificate') -> bool:
        """Check if self and other are concurrent (incomparable)"""
        return not (self.causally_precedes(other) or other.causally_precedes(self))
    
    def serialize(self) -> bytes:
        """Serialize causal certificate"""
        data = {
            'vector_clock': self.vector_clock,
            'happened_before': self.happened_before
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'CausalCertificate':
        """Deserialize causal certificate"""
        cert_dict = json.loads(data.decode('utf-8'))
        cert = cls()
        cert.vector_clock = cert_dict['vector_clock']
        cert.happened_before = [tuple(x) for x in cert_dict['happened_before']]
        return cert

@dataclass
class EntropySignature:
    """
    Entropy characteristics of data at origin
    
    Used for error detection and semantic validation
    """
    total_entropy: float  # Shannon entropy of data
    local_curvature: float  # Spacetime curvature at origin
    temperature: float  # Local temperature (for Hawking radiation correlation)
    checksum: str  # Hash of data
    
    def serialize(self) -> bytes:
        data = {
            'total_entropy': self.total_entropy,
            'local_curvature': self.local_curvature,
            'temperature': self.temperature,
            'checksum': self.checksum
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'EntropySignature':
        sig_dict = json.loads(data.decode('utf-8'))
        return cls(**sig_dict)
    
    @staticmethod
    def compute_from_data(data: np.ndarray, position: np.ndarray, metric) -> 'EntropySignature':
        """Compute entropy signature from raw data"""
        # Shannon entropy
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Checksum
        checksum = hashlib.sha256(data.tobytes()).hexdigest()
        
        # Placeholder values for curvature and temperature
        curvature = 0.0  # Would compute Ricci scalar at position
        temperature = 0.0  # Would compute local Hawking temperature
        
        return EntropySignature(
            total_entropy=float(entropy),
            local_curvature=curvature,
            temperature=temperature,
            checksum=checksum
        )

@dataclass
class SemanticCoordinate:
    """
    Location in semantic graph space
    
    Represents data's conceptual position
    """
    vertex_id: int
    embedding: np.ndarray  # High-dimensional semantic embedding
    salience: float  # Importance score
    confidence: float  # Confidence in classification
    
    def serialize(self) -> bytes:
        data = {
            'vertex_id': self.vertex_id,
            'embedding': self.embedding.tolist(),
            'salience': self.salience,
            'confidence': self.confidence
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'SemanticCoordinate':
        coord_dict = json.loads(data.decode('utf-8'))
        coord_dict['embedding'] = np.array(coord_dict['embedding'])
        return cls(**coord_dict)

@dataclass
class Packet:
    """
    Data packet for wormhole transport

    Conforms to holographic bound: size limited by throat area.
    Full protocol-compliant packet with entropy signatures and causal certificates.
    """
    packet_id: str
    packet_type: PacketType

    # Core payload
    data: bytes
    semantic_coord: SemanticCoordinate

    # Cryptographic metadata for holographic verification
    entropy_signature: EntropySignature
    causal_cert: CausalCertificate
    origin_time: float  # Proper time at origin
    origin_position: np.ndarray  # (t, r, θ, φ)

    # Transport metadata
    priority: float  # 0-1, higher = more urgent
    size_bytes: int
    created_at: float

    # Error correction
    error_correction_code: Optional[bytes] = None
    redundancy_level: int = 1

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(self.data)
    
    @classmethod
    def create_from_semantic_data(cls,
                                   data: np.ndarray,
                                   semantic_coord: SemanticCoordinate,
                                   position: np.ndarray,
                                   proper_time: float,
                                   metric,
                                   priority: float = 0.5) -> 'Packet':
        """Create packet from semantic graph data"""
        # Compute entropy signature
        entropy_sig = EntropySignature.compute_from_data(data, position, metric)
        
        # Initialize causal certificate
        causal_cert = CausalCertificate()
        
        # Serialize data
        data_bytes = data.tobytes()
        
        # Create packet
        packet = cls(
            packet_id=str(uuid.uuid4()),
            packet_type=PacketType.DATA,
            data=data_bytes,
            semantic_coord=semantic_coord,
            entropy_signature=entropy_sig,
            causal_cert=causal_cert,
            origin_time=proper_time,
            origin_position=position,
            priority=priority,
            size_bytes=len(data_bytes),
            created_at=proper_time
        )
        
        # Add error correction
        packet.add_error_correction()
        
        return packet
    
    def add_error_correction(self):
        """Add Reed-Solomon error correction code"""
        from reedsolo import RSCodec
        
        rsc = RSCodec(10)  # 10 error correction symbols
        self.error_correction_code = rsc.encode(self.data)
        self.size_bytes = len(self.error_correction_code)
    
    def verify_integrity(self) -> bool:
        """Verify packet integrity using entropy signature"""
        # Recompute checksum
        actual_checksum = hashlib.sha256(self.data).hexdigest()
        return actual_checksum == self.entropy_signature.checksum
    
    def correct_errors(self) -> bool:
        """Attempt to correct errors using ECC"""
        if self.error_correction_code is None:
            return False
        
        try:
            from reedsolo import RSCodec, ReedSolomonError
            rsc = RSCodec(10)
            self.data = rsc.decode(self.error_correction_code)[0]
            return True
        except ImportError:
            # reedsolo library not installed
            return False
        except (ReedSolomonError, IndexError, ValueError) as e:
            # Decoding failed - data is too corrupted to recover
            return False
    
    def serialize(self) -> bytes:
        """Serialize packet for transport"""
        # Header
        header = {
            'packet_id': self.packet_id,
            'packet_type': self.packet_type.value,
            'size_bytes': self.size_bytes,
            'origin_time': self.origin_time,
            'origin_position': self.origin_position.tolist(),
            'priority': self.priority,
            'created_at': self.created_at,
            'redundancy_level': self.redundancy_level
        }
        header_bytes = json.dumps(header).encode('utf-8')
        
        # Components
        semantic_bytes = self.semantic_coord.serialize()
        entropy_bytes = self.entropy_signature.serialize()
        causal_bytes = self.causal_cert.serialize()
        
        # Lengths
        header_len = len(header_bytes).to_bytes(4, 'big')
        semantic_len = len(semantic_bytes).to_bytes(4, 'big')
        entropy_len = len(entropy_bytes).to_bytes(4, 'big')
        causal_len = len(causal_bytes).to_bytes(4, 'big')
        data_len = len(self.data).to_bytes(4, 'big')
        
        # Concatenate
        result = (
            header_len + header_bytes +
            semantic_len + semantic_bytes +
            entropy_len + entropy_bytes +
            causal_len + causal_bytes +
            data_len + self.data
        )
        
        if self.error_correction_code:
            ecc_len = len(self.error_correction_code).to_bytes(4, 'big')
            result += ecc_len + self.error_correction_code
        
        return result
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'Packet':
        """Deserialize packet from bytes"""
        offset = 0
        
        # Header
        header_len = int.from_bytes(data[offset:offset+4], 'big')
        offset += 4
        header_dict = json.loads(data[offset:offset+header_len].decode('utf-8'))
        offset += header_len
        
        # Semantic coordinate
        semantic_len = int.from_bytes(data[offset:offset+4], 'big')
        offset += 4
        semantic_coord = SemanticCoordinate.deserialize(data[offset:offset+semantic_len])
        offset += semantic_len
        
        # Entropy signature
        entropy_len = int.from_bytes(data[offset:offset+4], 'big')
        offset += 4
        entropy_sig = EntropySignature.deserialize(data[offset:offset+entropy_len])
        offset += entropy_len
        
        # Causal certificate
        causal_len = int.from_bytes(data[offset:offset+4], 'big')
        offset += 4
        causal_cert = CausalCertificate.deserialize(data[offset:offset+causal_len])
        offset += causal_len
        
        # Data
        data_len = int.from_bytes(data[offset:offset+4], 'big')
        offset += 4
        packet_data = data[offset:offset+data_len]
        offset += data_len
        
        # Error correction code (if present)
        ecc = None
        if offset < len(data):
            ecc_len = int.from_bytes(data[offset:offset+4], 'big')
            offset += 4
            ecc = data[offset:offset+ecc_len]
        
        return cls(
            packet_id=header_dict['packet_id'],
            packet_type=PacketType(header_dict['packet_type']),
            data=packet_data,
            semantic_coord=semantic_coord,
            entropy_signature=entropy_sig,
            causal_cert=causal_cert,
            origin_time=header_dict['origin_time'],
            origin_position=np.array(header_dict['origin_position']),
            priority=header_dict['priority'],
            size_bytes=header_dict['size_bytes'],
            created_at=header_dict['created_at'],
            error_correction_code=ecc,
            redundancy_level=header_dict['redundancy_level']
        )

# =============================================================================
# 3. CHANNEL LAYER (Wormhole Transport Protocol)
# =============================================================================

@dataclass
class ChannelState:
    """State of wormhole channel"""
    capacity: float  # Maximum bits per unit time
    current_load: float  # Current utilization
    packet_queue: List[Packet] = field(default_factory=list)
    in_transit: Dict[str, Packet] = field(default_factory=dict)
    delivered: Dict[str, Packet] = field(default_factory=dict)
    
    # Statistics
    total_packets_sent: int = 0
    total_packets_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    packet_loss_rate: float = 0.0
    average_latency: float = 0.0

class WormholeTransportProtocol:
    """
    Protocol for reliable packet transport through wormhole
    
    Features:
    - Priority-based scheduling
    - Congestion control
    - Error correction
    - Causal ordering preservation
    - Bandwidth management
    """
    
    def __init__(self,
                 throat_area: float,
                 holographic_constant: float = 1.0):
        # Channel capacity from holographic bound
        # I_max = (c³ A) / (4ħG) in geometric units
        # FIX: Scale capacity to be in bytes (realistic for packet transport)
        # Base capacity scaled by 100 to allow reasonable packet sizes
        self.max_capacity_bytes = holographic_constant * throat_area * 100  # ~5500 bytes
        self.max_capacity = self.max_capacity_bytes  # Alias for compatibility

        # Max packets in flight (for holographic utilization calculation)
        # Assume ~128 bytes/packet (16 float64 values)
        self.max_packets_in_flight = max(10, int(self.max_capacity_bytes / 128))

        self.channel_state = ChannelState(capacity=self.max_capacity_bytes, current_load=0.0)

        # Congestion control
        self.congestion_window = 1.0  # AIMD parameter
        self.slow_start_threshold = 10.0

        # Timing
        self.rtt_estimate = 1.0  # Round-trip time estimate
        self.rtt_variance = 0.1

        # FIX: Track current byte load separately from packet count
        self.current_byte_load = 0.0

        # FIX: PPI tracking - enqueue attempts and acceptance
        self.enqueue_attempts = 0
        self.enqueue_accepted = 0
        self.enqueue_rejected = 0

    def enqueue_packet(self, packet: Packet) -> bool:
        """
        Add packet to transmission queue

        Returns True if packet accepted, False if rejected
        """
        # FIX: Track all enqueue attempts for PPI
        self.enqueue_attempts += 1

        # FIX: Use byte-based load tracking consistently
        # Calculate total bytes in queue + in transit
        queued_bytes = sum(p.size_bytes for p in self.channel_state.packet_queue)
        in_transit_bytes = sum(p.size_bytes for p in self.channel_state.in_transit.values())
        current_bytes = queued_bytes + in_transit_bytes

        projected_load = current_bytes + packet.size_bytes
        if projected_load > self.max_capacity_bytes * self.congestion_window:
            self.enqueue_rejected += 1
            return False

        # Add to queue (sorted by priority)
        self.channel_state.packet_queue.append(packet)
        self.channel_state.packet_queue.sort(key=lambda p: p.priority, reverse=True)

        self.enqueue_accepted += 1
        return True

    def get_acceptance_rate(self) -> float:
        """Get enqueue acceptance rate for PPI monitoring"""
        if self.enqueue_attempts == 0:
            return 1.0  # No attempts = no rejections
        return self.enqueue_accepted / self.enqueue_attempts

    def reset_counters(self):
        """Reset PPI counters (call at logging intervals)"""
        self.enqueue_attempts = 0
        self.enqueue_accepted = 0
        self.enqueue_rejected = 0
    
    def transmit_packets(self, dt: float) -> List[Packet]:
        """
        Transmit packets according to bandwidth constraints

        Returns list of packets that completed transmission
        """
        transmitted = []
        # FIX: Use byte-based capacity consistently
        available_bandwidth = self.max_capacity_bytes * self.congestion_window * dt
        used_bandwidth = 0.0

        while self.channel_state.packet_queue and used_bandwidth < available_bandwidth:
            packet = self.channel_state.packet_queue[0]

            # Check if packet fits in remaining bandwidth
            if used_bandwidth + packet.size_bytes <= available_bandwidth:
                # Remove from queue
                self.channel_state.packet_queue.pop(0)

                # Add to in-transit
                self.channel_state.in_transit[packet.packet_id] = packet

                # Update statistics
                self.channel_state.total_packets_sent += 1
                self.channel_state.total_bytes_sent += packet.size_bytes
                used_bandwidth += packet.size_bytes

                transmitted.append(packet)
            else:
                break

        # FIX: Update load as packet count (for compatibility) and byte load
        self.channel_state.current_load = len(self.channel_state.in_transit)
        self.current_byte_load = sum(p.size_bytes for p in self.channel_state.in_transit.values())

        return transmitted
    
    def receive_packet(self, packet_id: str) -> Optional[Packet]:
        """Mark packet as received"""
        if packet_id in self.channel_state.in_transit:
            packet = self.channel_state.in_transit.pop(packet_id)
            self.channel_state.delivered[packet_id] = packet
            
            # Update statistics
            self.channel_state.total_packets_received += 1
            self.channel_state.total_bytes_received += packet.size_bytes
            
            # Update RTT estimate (placeholder)
            # In real implementation, would use actual timestamps
            self._update_rtt_estimate(1.0)
            
            # Congestion control: increase window (additive increase)
            self._increase_congestion_window()
            
            return packet
        return None
    
    def handle_packet_loss(self, packet_id: str):
        """Handle lost packet"""
        if packet_id in self.channel_state.in_transit:
            packet = self.channel_state.in_transit.pop(packet_id)
            
            # Re-enqueue with higher priority
            packet.priority = min(1.0, packet.priority * 1.5)
            self.channel_state.packet_queue.insert(0, packet)
            
            # Update loss rate
            total = self.channel_state.total_packets_sent
            losses = total - self.channel_state.total_packets_received
            self.channel_state.packet_loss_rate = losses / max(1, total)
            
            # Congestion control: decrease window (multiplicative decrease)
            self._decrease_congestion_window()
    
    def _increase_congestion_window(self):
        """Additive increase in congestion window (AIMD)"""
        if self.congestion_window < self.slow_start_threshold:
            # Slow start: exponential increase
            self.congestion_window *= 2
        else:
            # Congestion avoidance: linear increase
            self.congestion_window += 1
    
    def _decrease_congestion_window(self):
        """Multiplicative decrease in congestion window (AIMD)"""
        self.slow_start_threshold = self.congestion_window / 2
        self.congestion_window = max(1.0, self.congestion_window / 2)
    
    def _update_rtt_estimate(self, measured_rtt: float):
        """Update RTT estimate using exponential smoothing"""
        alpha = 0.125  # Smoothing factor
        self.rtt_estimate = (1 - alpha) * self.rtt_estimate + alpha * measured_rtt
        
        # Update variance
        beta = 0.25
        self.rtt_variance = (1 - beta) * self.rtt_variance + beta * abs(measured_rtt - self.rtt_estimate)
    
    def get_timeout(self) -> float:
        """Compute retransmission timeout"""
        return self.rtt_estimate + 4 * self.rtt_variance

    def get_queue_size(self) -> int:
        """Get current number of packets in queue"""
        return len(self.channel_state.packet_queue)

    def get_in_transit_count(self) -> int:
        """Get number of packets currently in transit"""
        return len(self.channel_state.in_transit)

    def get_utilization(self) -> float:
        """Get current channel utilization (0-1)"""
        return self.channel_state.current_load / max(1, self.max_capacity)

    def get_stats(self) -> dict:
        """Get protocol statistics"""
        return {
            'queue_size': self.get_queue_size(),
            'in_transit': self.get_in_transit_count(),
            'total_sent': self.channel_state.total_packets_sent,
            'total_received': self.channel_state.total_packets_received,
            'bytes_sent': self.channel_state.total_bytes_sent,
            'bytes_received': self.channel_state.total_bytes_received,
            'loss_rate': self.channel_state.packet_loss_rate,
            'utilization': self.get_utilization(),
            'congestion_window': self.congestion_window,
        }

# =============================================================================
# 4. SYNCHRONIZATION LAYER (Clock Coordination)
# =============================================================================

@dataclass
class ClockState:
    """Local clock state for an agent"""
    proper_time: float  # Agent's proper time
    coordinate_time: float  # External coordinate time
    time_dilation_factor: float  # γ = dt_proper / dt_coordinate
    
    # Synchronization
    reference_agent_id: Optional[str] = None
    last_sync_time: float = 0.0
    drift_rate: float = 0.0  # Estimated clock drift

class ClockSynchronizationProtocol:
    """
    Lamport clock + GPS-like time synchronization for relativistic regime
    
    Ensures causal consistency despite time dilation
    """
    
    def __init__(self):
        self.agent_clocks: Dict[str, ClockState] = {}
        self.synchronization_interval = 10.0  # Sync every 10 time units
        
    def register_agent(self, agent_id: str, initial_time: float = 0.0):
        """Register agent with local clock"""
        self.agent_clocks[agent_id] = ClockState(
            proper_time=initial_time,
            coordinate_time=initial_time,
            time_dilation_factor=1.0
        )
    
    def update_agent_clock(self, 
                          agent_id: str, 
                          dt_coordinate: float,
                          time_dilation: float):
        """Update agent's clock accounting for time dilation"""
        if agent_id not in self.agent_clocks:
            self.register_agent(agent_id)
        
        clock = self.agent_clocks[agent_id]
        
        # Update coordinate time
        clock.coordinate_time += dt_coordinate
        
        # Update proper time with time dilation
        clock.proper_time += dt_coordinate / time_dilation
        clock.time_dilation_factor = time_dilation
    
    def synchronize_clocks(self, agent_id_1: str, agent_id_2: str) -> float:
        """
        Synchronize two agent clocks
        
        Returns time offset (how much to adjust agent_2's clock)
        """
        clock1 = self.agent_clocks[agent_id_1]
        clock2 = self.agent_clocks[agent_id_2]
        
        # Use coordinate time as reference
        # Offset = clock1.coordinate_time - clock2.coordinate_time
        offset = clock1.coordinate_time - clock2.coordinate_time
        
        # Apply correction to agent_2
        clock2.coordinate_time += offset
        clock2.last_sync_time = clock2.proper_time
        clock2.reference_agent_id = agent_id_1
        
        return offset
    
    def estimate_remote_time(self, 
                            local_agent_id: str,
                            remote_agent_id: str,
                            message_timestamp: float) -> float:
        """
        Estimate remote agent's current time given a message timestamp
        
        Accounts for propagation delay and clock drift
        """
        local_clock = self.agent_clocks[local_agent_id]
        remote_clock = self.agent_clocks[remote_agent_id]
        
        # Time elapsed since message sent
        elapsed = local_clock.proper_time - message_timestamp
        
        # Estimated remote time = message_timestamp + elapsed * (remote_dilation / local_dilation)
        dilation_ratio = remote_clock.time_dilation_factor / local_clock.time_dilation_factor
        estimated_remote_time = message_timestamp + elapsed * dilation_ratio
        
        return estimated_remote_time
    
    def compute_happened_before(self, 
                                event1: Tuple[str, float],  # (agent_id, timestamp)
                                event2: Tuple[str, float]) -> bool:
        """
        Determine if event1 -> event2 (event1 causally precedes event2)
        
        Uses vector clocks embedded in causal certificates
        """
        agent1, time1 = event1
        agent2, time2 = event2
        
        # Convert to coordinate time
        clock1 = self.agent_clocks[agent1]
        clock2 = self.agent_clocks[agent2]
        
        # Simple comparison in coordinate time
        # (In full implementation, would use vector clocks)
        return clock1.coordinate_time < clock2.coordinate_time

# =============================================================================
# 5. CROSS-COLONY COORDINATION PROTOCOLS
# =============================================================================

@dataclass
class BeaverAntInterface:
    """Protocol for Beaver-Ant coordination"""
    
    @staticmethod
    def beaver_provides_stability(beaver_position: np.ndarray,
                                   structural_field: float,
                                   stability_duration: float) -> Message:
        """Beaver broadcasts structural stability information to ants"""
        return Message.create(
            sender_id="beaver",
            receiver_id="ants",
            message_type=MessageType.STRUCTURAL_UPDATE,
            payload={
                'position': beaver_position.tolist(),
                'field_strength': structural_field,
                'stability_duration': stability_duration,
                'curvature': 0.0  # Placeholder
            },
            timestamp=0.0
        )
    
    @staticmethod
    def ant_requests_scaffold(ant_position: np.ndarray,
                             required_stability: float,
                             duration: float) -> Message:
        """Ant requests structural support from beavers"""
        return Message.create(
            sender_id="ant",
            receiver_id="beavers",
            message_type=MessageType.SCAFFOLD_REQUEST,
            payload={
                'position': ant_position.tolist(),
                'required_stability': required_stability,
                'duration': duration
            },
            timestamp=0.0
        )

@dataclass
class AntBeeInterface:
    """Protocol for Ant-Bee coordination"""
    
    @staticmethod
    def ant_provides_routing(vertex_id: int,
                            semantic_embedding: np.ndarray,
                            salience: float,
                            pheromone_strength: float) -> Message:
        """Ant provides routing information to bees"""
        return Message.create(
            sender_id="ant",
            receiver_id="bees",
            message_type=MessageType.ROUTING_INFO,
            payload={
                'vertex_id': vertex_id,
                'embedding': semantic_embedding.tolist(),
                'salience': salience,
                'pheromone': pheromone_strength,
                'gradient': [0.0, 0.0, 0.0]  # Semantic gradient direction
            },
            timestamp=0.0
        )
    
    @staticmethod
    def bee_reports_congestion(wormhole_position: np.ndarray,
                              congestion_level: float,
                              estimated_delay: float) -> Message:
        """Bee reports congestion to ants"""
        return Message.create(
            sender_id="bee",
            receiver_id="ants",
            message_type=MessageType.CONGESTION_ALERT,
            payload={
                'position': wormhole_position.tolist(),
                'congestion': congestion_level,
                'delay': estimated_delay
            },
            timestamp=0.0
        )

@dataclass
class BeeBeaverInterface:
    """Protocol for Bee-Beaver coordination"""
    
    @staticmethod
    def bee_queries_capacity(wormhole_edge: str) -> Message:
        """Bee queries wormhole capacity from beavers"""
        return Message.create(
            sender_id="bee",
            receiver_id="beavers",
            message_type=MessageType.CAPACITY_QUERY,
            payload={
                'edge': wormhole_edge,
                'current_load': 0.0
            },
            timestamp=0.0
        )
    
    @staticmethod
    def beaver_provides_capacity(wormhole_edge: str,
                                 available_capacity: float,
                                 throat_stability: float) -> Message:
        """Beaver reports wormhole capacity to bees"""
        return Message.create(
            sender_id="beaver",
            receiver_id="bees",
            message_type=MessageType.STRUCTURAL_UPDATE,
            payload={
                'edge': wormhole_edge,
                'capacity': available_capacity,
                'stability': throat_stability,
                'recommended_packet_size': 0.0  # Based on current throat geometry
            },
            timestamp=0.0
        )

# =============================================================================
# 6. EXTERNAL INTERFACE PROTOCOL
# =============================================================================

class QueryType(Enum):
    """Types of external queries to archive"""
    KEYWORD_SEARCH = "keyword_search"
    SEMANTIC_SEARCH = "semantic_search"
    TEMPORAL_QUERY = "temporal_query"
    CAUSAL_QUERY = "causal_query"
    STATISTICAL_QUERY = "statistical_query"

@dataclass
class ArchiveQuery:
    """Query to blackhole archive"""
    query_id: str
    query_type: QueryType
    parameters: Dict[str, Any]
    confidence_threshold: float = 0.5
    max_results: int = 10
    
    def serialize(self) -> bytes:
        data = {
            'query_id': self.query_id,
            'query_type': self.query_type.value,
            'parameters': self.parameters,
            'confidence_threshold': self.confidence_threshold,
            'max_results': self.max_results
        }
        return json.dumps(data).encode('utf-8')

@dataclass
class ArchiveResponse:
    """Response from archive"""
    query_id: str
    results: List[Packet]
    confidence_scores: List[float]
    causal_ordering: List[Tuple[int, int]]  # Pairs of indices with causal relations
    metadata: Dict[str, Any]
    
    def serialize(self) -> bytes:
        # Serialize packets
        packet_bytes = [p.serialize() for p in self.results]
        
        data = {
            'query_id': self.query_id,
            'num_results': len(self.results),
            'packet_lengths': [len(pb) for pb in packet_bytes],
            'confidence_scores': self.confidence_scores,
            'causal_ordering': self.causal_ordering,
            'metadata': self.metadata
        }
        
        header = json.dumps(data).encode('utf-8')
        header_len = len(header).to_bytes(4, 'big')
        
        # Concatenate
        result = header_len + header
        for pb in packet_bytes:
            result += pb
        
        return result

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_protocol_usage():
    """Demonstrate protocol usage"""
    
    # 1. Create packet
    data = np.random.rand(100)
    semantic_coord = SemanticCoordinate(
        vertex_id=42,
        embedding=np.random.rand(128),
        salience=0.8,
        confidence=0.9
    )
    
    packet = Packet.create_from_semantic_data(
        data=data,
        semantic_coord=semantic_coord,
        position=np.array([0, 10, np.pi/2, 0]),  # Outside horizon
        proper_time=0.0,
        metric=None,  # Would pass actual metric
        priority=0.7
    )
    
    # 2. Setup wormhole transport
    throat_area = 4 * np.pi * (2.0)**2  # Throat radius = 2
    transport = WormholeTransportProtocol(throat_area=throat_area)
    
    # 3. Enqueue packet
    success = transport.enqueue_packet(packet)
    print(f"Packet enqueued: {success}")
    
    # 4. Transmit
    transmitted = transport.transmit_packets(dt=1.0)
    print(f"Transmitted {len(transmitted)} packets")
    
    # 5. Serialize for storage
    serialized = packet.serialize()
    print(f"Serialized packet size: {len(serialized)} bytes")
    
    # 6. Deserialize
    reconstructed = Packet.deserialize(serialized)
    print(f"Packet ID match: {reconstructed.packet_id == packet.packet_id}")
    
    # 7. Clock synchronization
    sync_protocol = ClockSynchronizationProtocol()
    sync_protocol.register_agent("beaver_1")
    sync_protocol.register_agent("ant_1")
    
    sync_protocol.update_agent_clock("beaver_1", dt_coordinate=1.0, time_dilation=1.5)
    sync_protocol.update_agent_clock("ant_1", dt_coordinate=1.0, time_dilation=1.0)
    
    offset = sync_protocol.synchronize_clocks("beaver_1", "ant_1")
    print(f"Clock offset: {offset}")

if __name__ == "__main__":
    example_protocol_usage()
