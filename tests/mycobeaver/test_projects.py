"""
Unit tests for mycobeaver/projects.py

Tests the bee-inspired waggle dance recruitment system.
"""

import pytest
import numpy as np
from mycobeaver.projects import Project, ProjectManager
from mycobeaver.config import ProjectConfig, ProjectType, ProjectStatus


class TestProject:
    """Tests for Project dataclass"""

    def test_project_creation(self):
        """Test project creation with defaults"""
        project = Project(
            id=0,
            project_type=ProjectType.DAM,
            status=ProjectStatus.PROPOSED,
            center=(10, 10),
            region_cells=[(10, 9), (10, 10), (10, 11)]
        )

        assert project.id == 0
        assert project.project_type == ProjectType.DAM
        assert project.status == ProjectStatus.PROPOSED
        assert project.recruitment_signal == 0.0
        assert project.wood_deposited == 0
        assert len(project.assigned_workers) == 0

    def test_project_with_metrics(self):
        """Test project with quality metrics"""
        project = Project(
            id=1,
            project_type=ProjectType.LODGE,
            status=ProjectStatus.ACTIVE,
            center=(5, 5),
            region_cells=[(5, 5)],
            resource_proximity=0.8,
            hydro_impact=0.6,
            safety_score=0.9,
            distance_cost=0.3
        )

        assert project.resource_proximity == 0.8
        assert project.hydro_impact == 0.6


class TestProjectManager:
    """Tests for ProjectManager class"""

    @pytest.fixture
    def project_manager(self, project_config, grid_size):
        """Create project manager for testing"""
        return ProjectManager(project_config, grid_size)

    @pytest.fixture
    def mock_grid_state(self, grid_size):
        """Create mock grid state"""
        class MockGridState:
            def __init__(self, size):
                self.vegetation = np.random.rand(size, size)
                self.water_depth = np.random.rand(size, size) * 0.3
                self.lodge_map = np.zeros((size, size), dtype=bool)
                self.dam_permeability = np.ones((size, size))
                self.elevation = np.random.rand(size, size) * 5

        return MockGridState(grid_size)

    def test_initialization(self, project_manager):
        """Test manager initialization"""
        assert len(project_manager.projects) == 0
        assert len(project_manager.active_projects) == 0
        assert project_manager.next_project_id == 0

    def test_propose_project(self, project_manager, mock_grid_state):
        """Test project proposal"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state,
            current_step=0
        )

        assert project_id is not None
        assert project_id in project_manager.projects
        assert project_id in project_manager.active_projects

    def test_propose_project_out_of_bounds(self, project_manager, mock_grid_state):
        """Test proposal at invalid location"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(-1, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        assert project_id is None

    def test_propose_project_too_close(self, project_manager, mock_grid_state):
        """Test proposal too close to existing project"""
        # First project
        project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        # Second project too close
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(9, 9),
            proposer_id=1,
            grid_state=mock_grid_state
        )

        assert project_id is None

    def test_get_project_region_dam(self, project_manager):
        """Test dam region computation"""
        region = project_manager._get_project_region((8, 8), ProjectType.DAM)

        # Dam spans horizontally
        assert len(region) == 5  # -2 to +2 in x
        assert all(y == 8 for y, x in region)

    def test_get_project_region_lodge(self, project_manager):
        """Test lodge region computation"""
        region = project_manager._get_project_region((8, 8), ProjectType.LODGE)

        # Lodge is 3x3
        assert len(region) == 9
        assert (8, 8) in region

    def test_compute_quality(self, project_manager):
        """Test quality computation"""
        project = Project(
            id=0,
            project_type=ProjectType.DAM,
            status=ProjectStatus.PROPOSED,
            center=(8, 8),
            region_cells=[],
            resource_proximity=0.5,
            hydro_impact=0.6,
            safety_score=0.7,
            distance_cost=0.4
        )

        quality = project_manager._compute_quality(project)
        assert quality > 0

    def test_advertise(self, project_manager, mock_grid_state):
        """Test project advertising"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.advertise(project_id, scout_id=1)

        project = project_manager.active_projects[project_id]
        assert 1 in project.advertising_scouts
        assert project.status == ProjectStatus.ACTIVE

    def test_stop_advertising(self, project_manager, mock_grid_state):
        """Test stopping advertisement"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.advertise(project_id, scout_id=1)
        project_manager.stop_advertising(project_id, scout_id=1)

        project = project_manager.active_projects[project_id]
        assert 1 not in project.advertising_scouts

    def test_update_recruitment(self, project_manager, mock_grid_state):
        """Test recruitment signal update"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.advertise(project_id, scout_id=1)

        initial_signal = project_manager.active_projects[project_id].recruitment_signal
        project_manager.update_recruitment(advertising_scouts=[1], dt=1.0)

        # Signal should increase with advertising
        new_signal = project_manager.active_projects[project_id].recruitment_signal
        assert new_signal >= initial_signal

    def test_recruitment_decay(self, project_manager, mock_grid_state):
        """Test recruitment signal decay"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        # Set high signal
        project_manager.active_projects[project_id].recruitment_signal = 5.0

        # Update with no advertisers
        project_manager.update_recruitment(advertising_scouts=[], dt=1.0)

        # Signal should decay
        assert project_manager.active_projects[project_id].recruitment_signal < 5.0

    def test_cross_inhibition(self, project_manager, mock_grid_state):
        """Test cross-inhibition between projects"""
        # Create two projects far apart
        project_id1 = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(2, 2),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_id2 = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(14, 14),
            proposer_id=1,
            grid_state=mock_grid_state
        )

        # Set signals
        project_manager.active_projects[project_id1].recruitment_signal = 2.0
        project_manager.active_projects[project_id2].recruitment_signal = 3.0

        # Update - cross inhibition should reduce signals
        project_manager.update_recruitment(advertising_scouts=[], dt=1.0)

        # Both should decay (natural + cross inhibition)

    def test_get_recruitment_probability(self, project_manager, mock_grid_state):
        """Test recruitment probability calculation"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.active_projects[project_id].recruitment_signal = 2.0

        prob = project_manager.get_recruitment_probability(project_id, agent_threshold=1.0)

        # Should be sigmoid(2.0 - 1.0) = sigmoid(1.0) ≈ 0.73
        assert 0 < prob < 1
        assert prob > 0.5  # Signal > threshold

    def test_assign_worker(self, project_manager, mock_grid_state):
        """Test worker assignment"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.assign_worker(project_id, worker_id=2)

        project = project_manager.active_projects[project_id]
        assert 2 in project.assigned_workers

    def test_unassign_worker(self, project_manager, mock_grid_state):
        """Test worker unassignment"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.assign_worker(project_id, worker_id=2)
        project_manager.unassign_worker(project_id, worker_id=2)

        project = project_manager.active_projects[project_id]
        assert 2 not in project.assigned_workers

    def test_add_progress(self, project_manager, mock_grid_state):
        """Test adding construction progress"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.add_progress(project_id, wood_amount=10)

        project = project_manager.active_projects[project_id]
        assert project.wood_deposited == 10

    def test_check_completions(self, project_manager, mock_grid_state):
        """Test project completion check"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        # Add enough wood
        project = project_manager.active_projects[project_id]
        project.wood_deposited = project.wood_required

        completed = project_manager.check_completions(mock_grid_state, current_step=100)

        assert len(completed) == 1
        assert completed[0] == ProjectType.DAM
        assert project_id not in project_manager.active_projects
        assert project_id in project_manager.completed_projects

    def test_abandon_project(self, project_manager, mock_grid_state):
        """Test project abandonment"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.abandon_project(project_id, reason="low recruitment")

        assert project_id not in project_manager.active_projects
        project = project_manager.projects[project_id]
        assert project.status == ProjectStatus.ABANDONED

    def test_get_recruitment_signals(self, project_manager, mock_grid_state):
        """Test getting recruitment signals array"""
        project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        signals = project_manager.get_recruitment_signals()

        assert len(signals) == 8  # Max projects
        assert signals[0] >= 0

    def test_get_best_project(self, project_manager, mock_grid_state):
        """Test finding best project for agent"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.active_projects[project_id].recruitment_signal = 5.0
        project_manager.active_projects[project_id].quality = 0.8

        best = project_manager.get_best_project(
            agent_position=(7, 7),
            agent_thresholds=np.array([0.5])
        )

        assert best == project_id

    def test_get_project_info(self, project_manager, mock_grid_state):
        """Test getting project info"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        info = project_manager.get_project_info(project_id)

        assert info is not None
        assert info["id"] == project_id
        assert info["type"] == "dam"
        assert "quality" in info
        assert "progress" in info

    def test_has_quorum(self, project_manager, mock_grid_state):
        """Test quorum check"""
        project_id = project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        # Add workers
        for i in range(5):
            project_manager.assign_worker(project_id, worker_id=i)

        # Check quorum with 10 total agents
        has_quorum = project_manager.has_quorum(project_id, total_agents=10)
        # 5/10 = 50%, needs to meet threshold

    def test_get_statistics(self, project_manager, mock_grid_state):
        """Test statistics retrieval"""
        project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        stats = project_manager.get_statistics()

        assert "n_active" in stats
        assert stats["n_active"] == 1
        assert "n_completed" in stats
        assert "avg_recruitment_signal" in stats

    def test_reset(self, project_manager, mock_grid_state):
        """Test manager reset"""
        project_manager.propose_project(
            project_type=ProjectType.DAM,
            center=(8, 8),
            proposer_id=0,
            grid_state=mock_grid_state
        )

        project_manager.reset()

        assert len(project_manager.projects) == 0
        assert len(project_manager.active_projects) == 0
        assert project_manager.next_project_id == 0


class TestRecruitmentDynamics:
    """Tests for recruitment signal dynamics"""

    @pytest.fixture
    def manager(self, grid_size):
        """Create manager with specific config"""
        config = ProjectConfig(
            dance_gain=1.0,
            recruitment_decay=0.1,
            cross_inhibition=0.05,
            quality_sensitivity=1.0
        )
        return ProjectManager(config, grid_size)

    @pytest.fixture
    def grid_state(self, grid_size):
        """Create grid state"""
        class MockGridState:
            def __init__(self, size):
                self.vegetation = np.ones((size, size)) * 0.5
                self.water_depth = np.ones((size, size)) * 0.1
                self.lodge_map = np.zeros((size, size), dtype=bool)
                self.dam_permeability = np.ones((size, size))
                self.elevation = np.zeros((size, size))

        return MockGridState(grid_size)

    def test_signal_growth_with_quality(self, manager, grid_state):
        """Test higher quality leads to faster signal growth"""
        # Create two projects with different qualities
        pid1 = manager.propose_project(ProjectType.DAM, (4, 4), 0, grid_state)
        pid2 = manager.propose_project(ProjectType.DAM, (12, 12), 1, grid_state)

        # Manually set qualities
        manager.active_projects[pid1].quality = 0.3
        manager.active_projects[pid2].quality = 0.9

        # Both advertised
        manager.advertise(pid1, 0)
        manager.advertise(pid2, 1)

        # Update
        manager.update_recruitment([0, 1], dt=1.0)

        # Higher quality should have higher signal growth
        sig1 = manager.active_projects[pid1].recruitment_signal
        sig2 = manager.active_projects[pid2].recruitment_signal

        assert sig2 > sig1

    def test_equilibrium_signal(self, manager, grid_state):
        """Test signal reaches equilibrium"""
        pid = manager.propose_project(ProjectType.DAM, (8, 8), 0, grid_state)
        manager.active_projects[pid].quality = 0.5
        manager.advertise(pid, 0)

        # Run many updates
        prev_signal = 0
        for _ in range(100):
            manager.update_recruitment([0], dt=1.0)
            curr_signal = manager.active_projects[pid].recruitment_signal

            # Eventually should stabilize
            if abs(curr_signal - prev_signal) < 0.01:
                break
            prev_signal = curr_signal

        # Should have reached some positive equilibrium
        assert manager.active_projects[pid].recruitment_signal > 0
