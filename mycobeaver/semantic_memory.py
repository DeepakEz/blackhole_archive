"""
MycoBeaver Semantic Memory System
==================================
Event-based memory with kNN retrieval and time decay.

This module implements a real semantic memory system (not just a TypedDict) that:
1. Stores events (floods, dam breaks, resource discoveries, etc.)
2. Creates embeddings for events (location + time + semantic features)
3. Retrieves similar events using kNN search
4. Applies time decay to event relevance
5. Provides query interface for Overmind and Agents

MEMORY TYPES:
- EpisodicMemory: Recent events with full details
- SemanticMemory: Aggregated knowledge (patterns, statistics)
- ProceduralMemory: Learned action sequences (implicit in policy)

EVENT TYPES:
- FLOOD: Water level spike at location
- DAM_BREAK: Dam failure event
- RESOURCE_DISCOVERY: Found vegetation/food source
- DANGER_ZONE: Area with hazards
- SAFE_HAVEN: Protected area with resources
- SUCCESSFUL_BUILD: Dam construction that worked well
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq


class EventType(Enum):
    """Types of events that can be stored in memory."""
    FLOOD = auto()
    DAM_BREAK = auto()
    RESOURCE_DISCOVERY = auto()
    DANGER_ZONE = auto()
    SAFE_HAVEN = auto()
    SUCCESSFUL_BUILD = auto()
    REPAIR_SUCCESS = auto()
    AGENT_DEATH = auto()
    COLONY_MILESTONE = auto()


@dataclass
class MemoryEvent:
    """
    Single event stored in semantic memory.

    Each event has:
    - Type and metadata
    - Spatial location (y, x)
    - Temporal information (step created, last accessed)
    - Severity/importance score
    - Embedding vector for similarity search
    """
    event_type: EventType
    location: Tuple[int, int]  # (y, x)
    step_created: int
    severity: float  # 0.0 to 1.0, higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Access tracking
    access_count: int = 0
    last_accessed: int = 0

    # Computed embedding (set by SemanticMemory)
    embedding: Optional[np.ndarray] = None

    # Unique identifier
    event_id: int = 0

    def __lt__(self, other: "MemoryEvent") -> bool:
        """For heap comparison - higher severity = higher priority."""
        return self.severity > other.severity


class RetrievalResult(NamedTuple):
    """Result of a memory query."""
    event: MemoryEvent
    similarity: float
    decayed_relevance: float


@dataclass
class MemoryConfig:
    """Configuration for semantic memory system."""
    # Capacity
    max_events: int = 1000
    max_events_per_type: int = 200

    # Embedding dimensions
    embedding_dim: int = 16

    # Time decay
    decay_halflife: int = 500  # Steps until relevance halves
    min_relevance: float = 0.01  # Below this, event can be pruned

    # Retrieval
    default_k: int = 5  # Default number of neighbors for kNN
    location_weight: float = 0.4  # Weight for spatial similarity
    time_weight: float = 0.3  # Weight for temporal similarity
    type_weight: float = 0.3  # Weight for type matching

    # Consolidation
    consolidation_interval: int = 100  # Steps between consolidation


class SemanticMemory:
    """
    Semantic memory system with kNN retrieval and time decay.

    This is a REAL memory system, not just a TypedDict:
    - Stores events with embeddings
    - Performs kNN similarity search
    - Applies time decay to relevance
    - Consolidates and prunes old memories
    - Tracks retrieval statistics
    """

    def __init__(self, config: MemoryConfig, grid_size: Tuple[int, int]):
        self.config = config
        self.grid_size = grid_size

        # Event storage
        self.events: List[MemoryEvent] = []
        self.events_by_type: Dict[EventType, List[MemoryEvent]] = {
            t: [] for t in EventType
        }

        # ID counter
        self._next_id = 0

        # Current step (for time decay)
        self.current_step = 0

        # Statistics
        self.total_events_stored = 0
        self.total_retrievals = 0
        self.retrieval_hits = 0  # Retrieved events that were actually useful
        self.consolidation_count = 0

        # Embedding matrix for fast kNN (rebuilt on changes)
        self._embedding_matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True

        # Spatial index for location-based queries
        self._spatial_grid: Dict[Tuple[int, int], List[int]] = {}

    def store_event(
        self,
        event_type: EventType,
        location: Tuple[int, int],
        severity: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a new event in memory.

        Args:
            event_type: Type of event
            location: (y, x) location
            severity: Importance score 0-1
            metadata: Additional event data

        Returns:
            Event ID
        """
        event = MemoryEvent(
            event_type=event_type,
            location=location,
            step_created=self.current_step,
            severity=float(np.clip(severity, 0.0, 1.0)),
            metadata=metadata or {},
            event_id=self._next_id,
            last_accessed=self.current_step
        )

        # Compute embedding
        event.embedding = self._compute_embedding(event)

        # Add to storage
        self.events.append(event)
        self.events_by_type[event_type].append(event)

        # Update spatial index
        self._add_to_spatial_index(event)

        # Mark matrix as needing rebuild
        self._matrix_dirty = True

        self._next_id += 1
        self.total_events_stored += 1

        # Prune if over capacity
        if len(self.events) > self.config.max_events:
            self._prune_oldest()

        return event.event_id

    def _compute_embedding(self, event: MemoryEvent) -> np.ndarray:
        """
        Compute embedding vector for an event.

        Embedding structure:
        - [0:2] Normalized location
        - [2:4] Normalized time (sin/cos for periodicity)
        - [4:12] One-hot event type
        - [12:14] Severity and log-severity
        - [14:16] Spatial context features
        """
        embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Normalized location
        y, x = event.location
        embedding[0] = y / self.grid_size[0]
        embedding[1] = x / self.grid_size[1]

        # Normalized time with periodicity (useful for daily/seasonal patterns)
        t = event.step_created / 1000.0  # Normalize to reasonable range
        embedding[2] = np.sin(2 * np.pi * t)
        embedding[3] = np.cos(2 * np.pi * t)

        # One-hot event type (8 dimensions for EventType enum)
        type_idx = event.event_type.value - 1  # Enum values start at 1
        if 4 <= type_idx + 4 < 12:
            embedding[4 + type_idx] = 1.0

        # Severity features
        embedding[12] = event.severity
        embedding[13] = np.log1p(event.severity * 10) / 3.0  # Log-scale

        # Spatial context: distance from center
        center_y, center_x = self.grid_size[0] / 2, self.grid_size[1] / 2
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        embedding[14] = dist_from_center / max_dist

        # Edge proximity
        edge_dist = min(y, x, self.grid_size[0] - y, self.grid_size[1] - x)
        max_edge_dist = min(self.grid_size) / 2
        embedding[15] = edge_dist / max_edge_dist

        return embedding

    def _add_to_spatial_index(self, event: MemoryEvent) -> None:
        """Add event to spatial index for fast location queries."""
        key = event.location
        if key not in self._spatial_grid:
            self._spatial_grid[key] = []
        self._spatial_grid[key].append(event.event_id)

    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the embedding matrix for fast kNN."""
        if not self.events:
            self._embedding_matrix = None
            return

        self._embedding_matrix = np.array(
            [e.embedding for e in self.events],
            dtype=np.float32
        )
        self._matrix_dirty = False

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
        event_type: Optional[EventType] = None,
        min_severity: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve k most similar events using kNN.

        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors to retrieve
            event_type: Filter by event type (optional)
            min_severity: Minimum severity threshold

        Returns:
            List of RetrievalResult sorted by decayed relevance
        """
        k = k or self.config.default_k
        self.total_retrievals += 1

        # Filter events
        candidates = self.events
        if event_type is not None:
            candidates = self.events_by_type[event_type]

        if not candidates:
            return []

        # Compute similarities
        results = []
        for event in candidates:
            if event.severity < min_severity:
                continue

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, event.embedding)

            # Time decay
            age = self.current_step - event.step_created
            decay = 2.0 ** (-age / self.config.decay_halflife)
            decayed_relevance = similarity * decay * event.severity

            if decayed_relevance >= self.config.min_relevance:
                results.append(RetrievalResult(
                    event=event,
                    similarity=similarity,
                    decayed_relevance=decayed_relevance
                ))

                # Update access tracking
                event.access_count += 1
                event.last_accessed = self.current_step

        # Sort by decayed relevance and return top k
        results.sort(key=lambda r: r.decayed_relevance, reverse=True)
        return results[:k]

    def retrieve_by_location(
        self,
        location: Tuple[int, int],
        radius: int = 3,
        k: Optional[int] = None,
        event_type: Optional[EventType] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve events near a location.

        Args:
            location: (y, x) center location
            radius: Search radius in grid cells
            k: Maximum number of results
            event_type: Filter by type (optional)

        Returns:
            List of RetrievalResult sorted by distance and relevance
        """
        k = k or self.config.default_k
        self.total_retrievals += 1

        y, x = location
        results = []

        # Search in radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if (ny, nx) in self._spatial_grid:
                    for event_id in self._spatial_grid[(ny, nx)]:
                        event = self._get_event_by_id(event_id)
                        if event is None:
                            continue
                        if event_type is not None and event.event_type != event_type:
                            continue

                        # Distance-based similarity
                        dist = np.sqrt(dy**2 + dx**2)
                        spatial_sim = 1.0 / (1.0 + dist)

                        # Time decay
                        age = self.current_step - event.step_created
                        decay = 2.0 ** (-age / self.config.decay_halflife)
                        decayed_relevance = spatial_sim * decay * event.severity

                        if decayed_relevance >= self.config.min_relevance:
                            results.append(RetrievalResult(
                                event=event,
                                similarity=spatial_sim,
                                decayed_relevance=decayed_relevance
                            ))

                            event.access_count += 1
                            event.last_accessed = self.current_step

        results.sort(key=lambda r: r.decayed_relevance, reverse=True)
        return results[:k]

    def retrieve_recent(
        self,
        n: int = 10,
        event_type: Optional[EventType] = None
    ) -> List[MemoryEvent]:
        """Retrieve n most recent events."""
        candidates = self.events
        if event_type is not None:
            candidates = self.events_by_type[event_type]

        # Sort by step_created descending
        sorted_events = sorted(
            candidates,
            key=lambda e: e.step_created,
            reverse=True
        )
        return sorted_events[:n]

    def query_for_overmind(
        self,
        query_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        High-level query interface for Overmind.

        Query types:
        - "flood_risk": Get flood history and high-risk areas
        - "dam_status": Get dam break history and vulnerable dams
        - "resource_map": Get resource discovery patterns
        - "danger_zones": Get known danger areas
        - "success_patterns": Get successful build locations

        Returns:
            Dict with query results and statistics
        """
        if query_type == "flood_risk":
            return self._query_flood_risk(**kwargs)
        elif query_type == "dam_status":
            return self._query_dam_status(**kwargs)
        elif query_type == "resource_map":
            return self._query_resource_map(**kwargs)
        elif query_type == "danger_zones":
            return self._query_danger_zones(**kwargs)
        elif query_type == "success_patterns":
            return self._query_success_patterns(**kwargs)
        else:
            return {"error": f"Unknown query type: {query_type}"}

    def _query_flood_risk(self, lookback: int = 500) -> Dict[str, Any]:
        """Query flood risk based on history."""
        recent_floods = [
            e for e in self.events_by_type[EventType.FLOOD]
            if self.current_step - e.step_created < lookback
        ]

        # Aggregate by location
        flood_counts: Dict[Tuple[int, int], int] = {}
        flood_severity: Dict[Tuple[int, int], float] = {}

        for flood in recent_floods:
            loc = flood.location
            flood_counts[loc] = flood_counts.get(loc, 0) + 1
            flood_severity[loc] = max(
                flood_severity.get(loc, 0),
                flood.severity
            )

        # Find high-risk areas
        high_risk_areas = [
            loc for loc, count in flood_counts.items()
            if count >= 2 or flood_severity.get(loc, 0) > 0.7
        ]

        return {
            "total_floods": len(recent_floods),
            "flood_locations": list(flood_counts.keys()),
            "high_risk_areas": high_risk_areas,
            "max_severity": max(flood_severity.values()) if flood_severity else 0.0,
            "avg_severity": (
                sum(e.severity for e in recent_floods) / len(recent_floods)
                if recent_floods else 0.0
            )
        }

    def _query_dam_status(self, lookback: int = 500) -> Dict[str, Any]:
        """Query dam break history."""
        recent_breaks = [
            e for e in self.events_by_type[EventType.DAM_BREAK]
            if self.current_step - e.step_created < lookback
        ]

        successful_builds = [
            e for e in self.events_by_type[EventType.SUCCESSFUL_BUILD]
            if self.current_step - e.step_created < lookback
        ]

        break_locations = [e.location for e in recent_breaks]
        success_locations = [e.location for e in successful_builds]

        return {
            "total_breaks": len(recent_breaks),
            "break_locations": break_locations,
            "total_successful_builds": len(successful_builds),
            "success_locations": success_locations,
            "failure_rate": (
                len(recent_breaks) / (len(recent_breaks) + len(successful_builds))
                if (recent_breaks or successful_builds) else 0.0
            )
        }

    def _query_resource_map(self, lookback: int = 300) -> Dict[str, Any]:
        """Query resource discovery patterns."""
        discoveries = [
            e for e in self.events_by_type[EventType.RESOURCE_DISCOVERY]
            if self.current_step - e.step_created < lookback
        ]

        # Aggregate by rough grid regions (4x4 chunks)
        chunk_size = max(self.grid_size[0] // 4, 1)
        chunk_resources: Dict[Tuple[int, int], float] = {}

        for disc in discoveries:
            chunk = (disc.location[0] // chunk_size, disc.location[1] // chunk_size)
            amount = disc.metadata.get("amount", disc.severity)
            chunk_resources[chunk] = chunk_resources.get(chunk, 0) + amount

        # Find best resource chunks
        best_chunks = sorted(
            chunk_resources.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "total_discoveries": len(discoveries),
            "resource_chunks": dict(chunk_resources),
            "best_chunks": [c[0] for c in best_chunks],
            "total_resources_found": sum(chunk_resources.values())
        }

    def _query_danger_zones(self, lookback: int = 500) -> Dict[str, Any]:
        """Query known danger areas."""
        dangers = [
            e for e in self.events_by_type[EventType.DANGER_ZONE]
            if self.current_step - e.step_created < lookback
        ]

        deaths = [
            e for e in self.events_by_type[EventType.AGENT_DEATH]
            if self.current_step - e.step_created < lookback
        ]

        # Combine danger sources
        danger_locations = set(e.location for e in dangers)
        death_locations = set(e.location for e in deaths)

        # Expand death locations to nearby danger zones
        for loc in death_locations:
            danger_locations.add(loc)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    danger_locations.add((loc[0] + dy, loc[1] + dx))

        return {
            "danger_zones": list(danger_locations),
            "death_locations": list(death_locations),
            "total_deaths": len(deaths),
            "danger_severity": max(
                (e.severity for e in dangers), default=0.0
            )
        }

    def _query_success_patterns(self, lookback: int = 500) -> Dict[str, Any]:
        """Query successful build patterns."""
        successes = [
            e for e in self.events_by_type[EventType.SUCCESSFUL_BUILD]
            if self.current_step - e.step_created < lookback
        ]

        repairs = [
            e for e in self.events_by_type[EventType.REPAIR_SUCCESS]
            if self.current_step - e.step_created < lookback
        ]

        # Analyze successful locations
        success_locations = [e.location for e in successes]

        # Find common patterns in metadata
        build_types: Dict[str, int] = {}
        for s in successes:
            build_type = s.metadata.get("structure_type", "dam")
            build_types[build_type] = build_types.get(build_type, 0) + 1

        return {
            "total_successes": len(successes),
            "success_locations": success_locations,
            "total_repairs": len(repairs),
            "build_type_distribution": build_types,
            "avg_success_severity": (
                sum(e.severity for e in successes) / len(successes)
                if successes else 0.0
            )
        }

    def query_for_agent(
        self,
        agent_location: Tuple[int, int],
        query_type: str,
        radius: int = 5
    ) -> List[MemoryEvent]:
        """
        Query interface for individual agents.

        Returns relevant events near agent's location.

        Query types:
        - "nearby_resources": Resource discoveries nearby
        - "nearby_dangers": Danger zones nearby
        - "nearby_floods": Flood history nearby
        - "nearby_builds": Successful builds nearby
        """
        if query_type == "nearby_resources":
            results = self.retrieve_by_location(
                agent_location, radius,
                event_type=EventType.RESOURCE_DISCOVERY
            )
        elif query_type == "nearby_dangers":
            results = self.retrieve_by_location(
                agent_location, radius,
                event_type=EventType.DANGER_ZONE
            )
        elif query_type == "nearby_floods":
            results = self.retrieve_by_location(
                agent_location, radius,
                event_type=EventType.FLOOD
            )
        elif query_type == "nearby_builds":
            results = self.retrieve_by_location(
                agent_location, radius,
                event_type=EventType.SUCCESSFUL_BUILD
            )
        else:
            # General nearby query
            results = self.retrieve_by_location(agent_location, radius)

        return [r.event for r in results]

    def consolidate(self) -> int:
        """
        Consolidate memory by pruning low-relevance events.

        Called periodically to maintain memory efficiency.

        Returns:
            Number of events pruned
        """
        self.consolidation_count += 1
        initial_count = len(self.events)

        # Calculate relevance for all events
        events_with_relevance = []
        for event in self.events:
            age = self.current_step - event.step_created
            decay = 2.0 ** (-age / self.config.decay_halflife)
            relevance = decay * event.severity * (1 + np.log1p(event.access_count))
            events_with_relevance.append((event, relevance))

        # Keep events above threshold
        surviving_events = [
            e for e, r in events_with_relevance
            if r >= self.config.min_relevance
        ]

        # If still over capacity, keep only top events
        if len(surviving_events) > self.config.max_events:
            events_with_relevance.sort(key=lambda x: x[1], reverse=True)
            surviving_events = [e for e, _ in events_with_relevance[:self.config.max_events]]

        # Rebuild indices
        pruned_count = len(self.events) - len(surviving_events)
        self.events = surviving_events
        self._rebuild_indices()

        return pruned_count

    def _rebuild_indices(self) -> None:
        """Rebuild all indices after consolidation."""
        # Rebuild type index
        self.events_by_type = {t: [] for t in EventType}
        for event in self.events:
            self.events_by_type[event.event_type].append(event)

        # Rebuild spatial index
        self._spatial_grid = {}
        for event in self.events:
            self._add_to_spatial_index(event)

        # Mark embedding matrix for rebuild
        self._matrix_dirty = True

    def _prune_oldest(self) -> None:
        """Prune oldest low-priority events to make room."""
        # Sort by (step_created, -severity) to remove old low-severity first
        self.events.sort(key=lambda e: (e.step_created, -e.severity))

        # Remove bottom 10%
        n_remove = max(1, len(self.events) // 10)
        removed = self.events[:n_remove]
        self.events = self.events[n_remove:]

        # Update indices
        self._rebuild_indices()

    def _get_event_by_id(self, event_id: int) -> Optional[MemoryEvent]:
        """Get event by ID (linear search - could optimize with dict)."""
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def step(self, current_step: int) -> None:
        """Update memory state for new step."""
        self.current_step = current_step

        # Periodic consolidation
        if current_step % self.config.consolidation_interval == 0:
            self.consolidate()

    def mark_retrieval_useful(self, event_id: int) -> None:
        """Mark a retrieved event as useful (for accuracy tracking)."""
        self.retrieval_hits += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics for logging."""
        return {
            "total_events": len(self.events),
            "events_by_type": {
                t.name: len(self.events_by_type[t])
                for t in EventType
            },
            "total_stored": self.total_events_stored,
            "total_retrievals": self.total_retrievals,
            "retrieval_accuracy": (
                self.retrieval_hits / self.total_retrievals
                if self.total_retrievals > 0 else 0.0
            ),
            "consolidation_count": self.consolidation_count,
            "current_step": self.current_step,
        }

    def reset(self) -> None:
        """Reset memory to initial state."""
        self.events.clear()
        self.events_by_type = {t: [] for t in EventType}
        self._next_id = 0
        self.current_step = 0
        self.total_events_stored = 0
        self.total_retrievals = 0
        self.retrieval_hits = 0
        self.consolidation_count = 0
        self._embedding_matrix = None
        self._matrix_dirty = True
        self._spatial_grid.clear()


# =============================================================================
# Helper functions for environment integration
# =============================================================================

def create_flood_event(
    memory: SemanticMemory,
    location: Tuple[int, int],
    water_level: float,
    affected_area: int = 1
) -> int:
    """Create a flood event from environment data."""
    severity = min(1.0, water_level / 3.0)  # Normalize by flood threshold
    return memory.store_event(
        EventType.FLOOD,
        location,
        severity,
        metadata={
            "water_level": water_level,
            "affected_area": affected_area
        }
    )


def create_dam_break_event(
    memory: SemanticMemory,
    location: Tuple[int, int],
    dam_id: int,
    integrity_at_break: float
) -> int:
    """Create a dam break event."""
    severity = 1.0 - integrity_at_break  # Lower integrity = more severe
    return memory.store_event(
        EventType.DAM_BREAK,
        location,
        severity,
        metadata={
            "dam_id": dam_id,
            "integrity_at_break": integrity_at_break
        }
    )


def create_resource_event(
    memory: SemanticMemory,
    location: Tuple[int, int],
    amount: float,
    resource_type: str = "vegetation"
) -> int:
    """Create a resource discovery event."""
    severity = min(1.0, amount / 2.0)  # Normalize
    return memory.store_event(
        EventType.RESOURCE_DISCOVERY,
        location,
        severity,
        metadata={
            "amount": amount,
            "resource_type": resource_type
        }
    )


def create_danger_event(
    memory: SemanticMemory,
    location: Tuple[int, int],
    danger_type: str,
    severity: float
) -> int:
    """Create a danger zone event."""
    return memory.store_event(
        EventType.DANGER_ZONE,
        location,
        severity,
        metadata={"danger_type": danger_type}
    )


def create_build_success_event(
    memory: SemanticMemory,
    location: Tuple[int, int],
    structure_type: str,
    effectiveness: float
) -> int:
    """Create a successful build event."""
    return memory.store_event(
        EventType.SUCCESSFUL_BUILD,
        location,
        effectiveness,
        metadata={"structure_type": structure_type}
    )
