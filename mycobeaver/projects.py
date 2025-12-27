"""
MycoBeaver Project Manager
===========================
Bee-inspired waggle dance recruitment system for colony projects.

Based on MycoBeaver Simulator Design Plan Section 2.3:
- Project quality estimation: Q_p = w1*r + w2*d_H + w3*s + w4*c
- Recruitment signals: dR_p/dt = γ_dance * Σ[advertising scouts] * f(Q_p) - κ * R_p - χ * Σ R_q
- Agent recruitment probability based on R_p and thresholds
- Cross-inhibition between competing projects
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .config import ProjectConfig, ProjectType, ProjectStatus


@dataclass
class Project:
    """Represents a colony construction project"""
    id: int
    project_type: ProjectType
    status: ProjectStatus

    # Location
    center: Tuple[int, int]  # (y, x) center of project
    region_cells: List[Tuple[int, int]]  # All cells in project region

    # Quality metrics
    resource_proximity: float = 0.0  # r - resources nearby
    hydro_impact: float = 0.0  # d_H - hydrological benefit
    safety_score: float = 0.0  # s - safety from hazards
    distance_cost: float = 0.0  # c - distance from lodge

    # Construction progress
    wood_deposited: int = 0
    wood_required: int = 100

    # Recruitment
    recruitment_signal: float = 0.0  # R_p
    quality: float = 0.0  # Q_p (computed)

    # Tracking
    proposer_id: int = -1
    assigned_workers: Set[int] = field(default_factory=set)
    creation_step: int = 0
    completion_step: Optional[int] = None

    # Advertising scouts
    advertising_scouts: Set[int] = field(default_factory=set)


class ProjectManager:
    """
    Manages colony construction projects with bee-style recruitment.

    Key mechanics:
    1. Scouts discover and propose project sites
    2. Quality Q_p computed from site features
    3. Scouts "dance" (advertise) to recruit workers
    4. Recruitment signal R_p grows with dancing, decays naturally
    5. Cross-inhibition: competing projects suppress each other
    6. Workers probabilistically join based on R_p and personal thresholds
    """

    def __init__(self, config: ProjectConfig, grid_size: int):
        self.config = config
        self.grid_size = grid_size

        # Project storage
        self.projects: Dict[int, Project] = {}  # All projects
        self.active_projects: Dict[int, Project] = {}  # Currently active
        self.completed_projects: Dict[int, Project] = {}  # Finished

        # ID counter
        self.next_project_id = 0

        # Recruitment signal history for smoothing
        self.signal_history_len = 10
        self.signal_history: Dict[int, List[float]] = {}

    def propose_project(self, project_type: ProjectType,
                        center: Tuple[int, int],
                        proposer_id: int,
                        grid_state,
                        current_step: int = 0) -> Optional[int]:
        """
        Propose a new project at a location.

        Args:
            project_type: Type of project (dam, lodge, canal)
            center: (y, x) center location
            proposer_id: ID of proposing scout
            grid_state: Current grid state for quality evaluation
            current_step: Current simulation step

        Returns:
            Project ID if created, None otherwise
        """
        # Check if location is valid
        y, x = center
        if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
            return None

        # Check for existing projects nearby
        for proj in self.active_projects.values():
            py, px = proj.center
            if abs(py - y) < 3 and abs(px - x) < 3:
                return None  # Too close to existing project

        # Determine project region
        region_cells = self._get_project_region(center, project_type)

        # Compute quality metrics
        resource_proximity = self._compute_resource_proximity(center, grid_state)
        hydro_impact = self._compute_hydro_impact(center, project_type, grid_state)
        safety_score = self._compute_safety_score(center, grid_state)
        distance_cost = self._compute_distance_cost(center, grid_state)

        # Set wood requirements based on type
        if project_type == ProjectType.DAM:
            wood_required = self.config.dam_wood_required
        elif project_type == ProjectType.LODGE:
            wood_required = self.config.lodge_wood_required
        else:
            wood_required = 75  # Default for canal

        # Create project
        project = Project(
            id=self.next_project_id,
            project_type=project_type,
            status=ProjectStatus.PROPOSED,
            center=center,
            region_cells=region_cells,
            resource_proximity=resource_proximity,
            hydro_impact=hydro_impact,
            safety_score=safety_score,
            distance_cost=distance_cost,
            wood_required=wood_required,
            proposer_id=proposer_id,
            creation_step=current_step,
        )

        # Compute overall quality
        project.quality = self._compute_quality(project)

        # Store project
        self.projects[project.id] = project
        self.active_projects[project.id] = project
        self.signal_history[project.id] = []

        self.next_project_id += 1

        return project.id

    def _get_project_region(self, center: Tuple[int, int],
                            project_type: ProjectType) -> List[Tuple[int, int]]:
        """Get cells that belong to a project region"""
        y, x = center
        cells = []

        if project_type == ProjectType.DAM:
            # Dam spans horizontally (blocking water flow)
            for dx in range(-2, 3):
                nx = x + dx
                if 0 <= nx < self.grid_size:
                    cells.append((y, nx))

        elif project_type == ProjectType.LODGE:
            # Lodge is a 3x3 area
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                        cells.append((ny, nx))

        elif project_type == ProjectType.CANAL:
            # Canal spans vertically
            for dy in range(-2, 3):
                ny = y + dy
                if 0 <= ny < self.grid_size:
                    cells.append((ny, x))

        return cells

    def _compute_resource_proximity(self, center: Tuple[int, int],
                                    grid_state) -> float:
        """Compute resource proximity score"""
        y, x = center
        score = 0.0
        radius = 5

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    dist = max(abs(dy), abs(dx))
                    if dist > 0:
                        # Resources closer are worth more
                        veg = grid_state.vegetation[ny, nx]
                        score += veg / dist

        # Normalize
        return min(1.0, score / 10.0)

    def _compute_hydro_impact(self, center: Tuple[int, int],
                              project_type: ProjectType,
                              grid_state) -> float:
        """Compute expected hydrological benefit"""
        y, x = center

        if project_type == ProjectType.DAM:
            # Dam benefit: water upstream, low water downstream
            upstream_water = 0.0
            downstream_water = 0.0

            for dy in range(-3, 0):  # Upstream
                ny = y + dy
                if 0 <= ny < self.grid_size:
                    upstream_water += grid_state.water_depth[ny, x]

            for dy in range(1, 4):  # Downstream
                ny = y + dy
                if 0 <= ny < self.grid_size:
                    downstream_water += grid_state.water_depth[ny, x]

            # Good dam site: water flows here, can create pond
            return min(1.0, (upstream_water + 0.5) / 2.0)

        elif project_type == ProjectType.LODGE:
            # Lodge benefit: moderate water nearby but not flooded
            water_here = grid_state.water_depth[y, x]
            if water_here > 0.5:  # Too flooded
                return 0.2
            elif water_here < 0.05:  # Too dry
                return 0.5
            else:
                return 0.9  # Good water access

        return 0.5  # Default

    def _compute_safety_score(self, center: Tuple[int, int],
                              grid_state) -> float:
        """Compute safety score for location"""
        y, x = center

        # Check for flood risk
        water = grid_state.water_depth[y, x]
        flood_risk = min(1.0, water / 0.8)

        # Check for predator proximity (if tracked)
        # For now, safety based on lodge proximity
        lodge_positions = np.argwhere(grid_state.lodge_map)
        if len(lodge_positions) > 0:
            min_dist = min(abs(lp[0] - y) + abs(lp[1] - x) for lp in lodge_positions)
            lodge_safety = 1.0 - min(1.0, min_dist / 20.0)
        else:
            lodge_safety = 0.5

        return 0.5 * (1.0 - flood_risk) + 0.5 * lodge_safety

    def _compute_distance_cost(self, center: Tuple[int, int],
                               grid_state) -> float:
        """Compute distance cost from lodge (lower = closer = better)"""
        y, x = center

        lodge_positions = np.argwhere(grid_state.lodge_map)
        if len(lodge_positions) == 0:
            return 0.5  # Default if no lodge

        min_dist = min(abs(lp[0] - y) + abs(lp[1] - x) for lp in lodge_positions)

        # Invert: lower distance = higher score
        return 1.0 - min(1.0, min_dist / (self.grid_size / 2))

    def _compute_quality(self, project: Project) -> float:
        """
        Compute overall project quality.

        Q_p = w1*r + w2*d_H + w3*s + w4*(1-c)

        Higher quality = more attractive project.
        """
        c = self.config

        quality = (
            c.resource_weight * project.resource_proximity +
            c.hydro_impact_weight * project.hydro_impact +
            c.safety_weight * project.safety_score +
            c.distance_cost_weight * project.distance_cost
        )

        return quality

    def advertise(self, project_id: int, scout_id: int):
        """
        Scout advertises (dances for) a project.

        This increases the project's recruitment signal.

        Args:
            project_id: ID of project to advertise
            scout_id: ID of advertising scout
        """
        if project_id not in self.active_projects:
            return

        project = self.active_projects[project_id]
        project.advertising_scouts.add(scout_id)

        # If first advertisement, activate project
        if project.status == ProjectStatus.PROPOSED:
            project.status = ProjectStatus.ACTIVE

    def stop_advertising(self, project_id: int, scout_id: int):
        """Scout stops advertising a project"""
        if project_id in self.active_projects:
            project = self.active_projects[project_id]
            project.advertising_scouts.discard(scout_id)

    def update_recruitment(self, advertising_scouts: List[int], dt: float):
        """
        Update recruitment signals for all active projects.

        dR_p/dt = γ_dance * N_advertising * f(Q_p) - κ * R_p - χ * Σ R_q

        Args:
            advertising_scouts: List of scout IDs currently advertising
            dt: Time step
        """
        # First pass: compute dance contributions
        dance_contributions = {}

        for project_id, project in self.active_projects.items():
            # Count active advertisers for this project
            n_advertising = sum(1 for sid in advertising_scouts
                               if sid in project.advertising_scouts)

            # f(Q) = exp(λQ) - quality sensitivity function
            f_q = np.exp(self.config.quality_sensitivity * project.quality)

            # Dance contribution
            dance_contributions[project_id] = (
                self.config.dance_gain * n_advertising * f_q
            )

        # Compute total recruitment for cross-inhibition
        total_R = sum(p.recruitment_signal for p in self.active_projects.values())

        # Second pass: update signals
        for project_id, project in self.active_projects.items():
            dance = dance_contributions[project_id]
            decay = self.config.recruitment_decay * project.recruitment_signal

            # Cross-inhibition: other projects' signals suppress this one
            other_R = total_R - project.recruitment_signal
            cross_inhibition = self.config.cross_inhibition * other_R

            # Update
            dR = dt * (dance - decay - cross_inhibition)
            project.recruitment_signal = max(0.0, project.recruitment_signal + dR)

            # Track history
            self.signal_history[project_id].append(project.recruitment_signal)
            if len(self.signal_history[project_id]) > self.signal_history_len:
                self.signal_history[project_id].pop(0)

    def get_recruitment_probability(self, project_id: int,
                                    agent_threshold: float) -> float:
        """
        Get probability that an agent joins a project.

        P(join) = sigmoid(R_p - θ_agent)

        Args:
            project_id: Project ID
            agent_threshold: Agent's personal threshold for this task type

        Returns:
            Probability in [0, 1]
        """
        if project_id not in self.active_projects:
            return 0.0

        R_p = self.active_projects[project_id].recruitment_signal

        # Sigmoid function
        x = R_p - agent_threshold
        return 1.0 / (1.0 + np.exp(-x))

    def assign_worker(self, project_id: int, worker_id: int):
        """Assign a worker to a project"""
        if project_id in self.active_projects:
            self.active_projects[project_id].assigned_workers.add(worker_id)

    def unassign_worker(self, project_id: int, worker_id: int):
        """Remove worker assignment from project"""
        if project_id in self.active_projects:
            self.active_projects[project_id].assigned_workers.discard(worker_id)

    def add_progress(self, project_id: int, wood_amount: int):
        """
        Add construction progress to a project.

        Args:
            project_id: Project ID
            wood_amount: Amount of wood deposited
        """
        if project_id not in self.active_projects:
            return

        project = self.active_projects[project_id]
        project.wood_deposited += wood_amount

    def check_completions(self, grid_state, current_step: int = 0) -> List[ProjectType]:
        """
        Check for completed projects and finalize them.

        Returns:
            List of completed project types
        """
        completed_types = []
        to_complete = []

        for project_id, project in self.active_projects.items():
            if project.wood_deposited >= project.wood_required:
                to_complete.append(project_id)

        for project_id in to_complete:
            project = self.active_projects.pop(project_id)
            project.status = ProjectStatus.COMPLETED
            project.completion_step = current_step

            self.completed_projects[project_id] = project
            completed_types.append(project.project_type)

            # Apply effects to grid
            self._apply_project_completion(project, grid_state)

        return completed_types

    def _apply_project_completion(self, project: Project, grid_state):
        """Apply effects of completed project to grid"""
        if project.project_type == ProjectType.DAM:
            # Reduce permeability in dam region
            for (y, x) in project.region_cells:
                if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                    grid_state.dam_permeability[y, x] = 0.1  # Very low permeability

        elif project.project_type == ProjectType.LODGE:
            # Mark lodge location
            for (y, x) in project.region_cells:
                if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                    grid_state.lodge_map[y, x] = True

    def abandon_project(self, project_id: int, reason: str = ""):
        """Abandon an active project"""
        if project_id in self.active_projects:
            project = self.active_projects.pop(project_id)
            project.status = ProjectStatus.ABANDONED
            # Could track reason if needed

    def get_recruitment_signals(self) -> np.ndarray:
        """
        Get recruitment signals for all active projects.

        Returns normalized array for observation features.
        """
        max_projects = 8  # Maximum projects to track in observations
        signals = np.zeros(max_projects)

        for i, project in enumerate(self.active_projects.values()):
            if i >= max_projects:
                break
            signals[i] = min(1.0, project.recruitment_signal / 5.0)  # Normalize

        return signals

    def get_recruitment_at_position(self, y: int, x: int, decay_radius: float = 10.0) -> float:
        """
        Get recruitment signal strength at a specific position.

        Combines recruitment signals from all active projects with
        distance-based falloff (Gaussian decay from project centers).

        This enables the unified coordination field Ψ from the framework:
        Ψ = α_τ·τ + α_D·(D/L) + α_R·R

        Args:
            y, x: Grid position
            decay_radius: Distance at which signal decays to 1/e

        Returns:
            Combined recruitment signal strength [0, 1]
        """
        if not self.active_projects:
            return 0.0

        total_recruitment = 0.0

        for project in self.active_projects.values():
            py, px = project.center
            # Euclidean distance to project center
            distance = np.sqrt((y - py)**2 + (x - px)**2)
            # Gaussian falloff
            falloff = np.exp(-distance**2 / (2 * decay_radius**2))
            # Weight by recruitment signal strength
            total_recruitment += project.recruitment_signal * falloff

        # Normalize to [0, 1]
        return min(1.0, total_recruitment / 5.0)

    def get_best_project(self, agent_position: Tuple[int, int],
                         agent_thresholds: np.ndarray,
                         max_distance: float = float('inf')) -> Optional[int]:
        """
        Get the best project for an agent to join.

        Considers recruitment signal, quality, and distance.

        Args:
            agent_position: (y, x) agent position
            agent_thresholds: Agent's task thresholds
            max_distance: Maximum acceptable distance

        Returns:
            Project ID or None
        """
        best_project_id = None
        best_score = -float('inf')

        ay, ax = agent_position

        for project_id, project in self.active_projects.items():
            py, px = project.center
            distance = abs(py - ay) + abs(px - ax)

            if distance > max_distance:
                continue

            # Compute attractiveness
            # Use first threshold for general task response
            threshold = agent_thresholds[0] if len(agent_thresholds) > 0 else 0.5

            recruitment_prob = self.get_recruitment_probability(project_id, threshold)
            distance_penalty = 1.0 - (distance / self.grid_size)

            score = recruitment_prob * project.quality * distance_penalty

            if score > best_score:
                best_score = score
                best_project_id = project_id

        return best_project_id

    def get_project_info(self, project_id: int) -> Optional[Dict]:
        """Get information about a project"""
        project = self.projects.get(project_id)
        if project is None:
            return None

        return {
            "id": project.id,
            "type": project.project_type.value,
            "status": project.status.value,
            "center": project.center,
            "quality": project.quality,
            "recruitment_signal": project.recruitment_signal,
            "progress": project.wood_deposited / project.wood_required,
            "n_workers": len(project.assigned_workers),
            "n_advertisers": len(project.advertising_scouts),
        }

    def has_quorum(self, project_id: int, total_agents: int) -> bool:
        """
        Check if project has achieved quorum for consensus decision.

        Args:
            project_id: Project ID
            total_agents: Total number of agents in colony

        Returns:
            True if quorum achieved
        """
        if project_id not in self.active_projects:
            return False

        project = self.active_projects[project_id]
        n_committed = len(project.assigned_workers) + len(project.advertising_scouts)

        return n_committed >= total_agents * self.config.quorum_threshold

    def get_statistics(self) -> Dict:
        """Get statistics about projects"""
        return {
            "n_active": len(self.active_projects),
            "n_completed": len(self.completed_projects),
            "total_proposed": len(self.projects),
            "avg_recruitment_signal": np.mean([
                p.recruitment_signal for p in self.active_projects.values()
            ]) if self.active_projects else 0.0,
            "avg_quality": np.mean([
                p.quality for p in self.active_projects.values()
            ]) if self.active_projects else 0.0,
        }

    def reset(self):
        """Reset project manager to initial state"""
        self.projects.clear()
        self.active_projects.clear()
        self.completed_projects.clear()
        self.signal_history.clear()
        self.next_project_id = 0
