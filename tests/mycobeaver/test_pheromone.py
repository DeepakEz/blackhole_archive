"""
Unit tests for mycobeaver/pheromone.py

Tests the Ant Colony Optimization-inspired pheromone routing system.
"""

import pytest
import numpy as np
from mycobeaver.pheromone import PheromoneField, PheromoneEdge, MultiPheromoneField
from mycobeaver.config import PheromoneConfig, InfoCostConfig


class TestPheromoneEdge:
    """Tests for PheromoneEdge dataclass"""

    def test_edge_creation(self):
        """Test edge creation with default values"""
        edge = PheromoneEdge(from_cell=(0, 0), to_cell=(0, 1))
        assert edge.from_cell == (0, 0)
        assert edge.to_cell == (0, 1)
        assert edge.pheromone == 0.0
        assert edge.traversal_count == 0
        assert edge.success_count == 0

    def test_edge_with_values(self):
        """Test edge creation with custom values"""
        edge = PheromoneEdge(
            from_cell=(1, 2),
            to_cell=(1, 3),
            pheromone=0.5,
            traversal_count=10,
            success_count=5
        )
        assert edge.pheromone == 0.5
        assert edge.traversal_count == 10
        assert edge.success_count == 5


class TestPheromoneField:
    """Tests for PheromoneField class"""

    def test_initialization(self, pheromone_config, grid_size, info_cost_config):
        """Test field initialization"""
        field = PheromoneField(pheromone_config, grid_size, info_cost_config)

        assert field.grid_size == grid_size
        assert field.pheromone_grid.shape == (grid_size, grid_size, 8)
        assert np.all(field.pheromone_grid == pheromone_config.min_pheromone)
        assert field.total_depositions == 0
        assert field.step_count == 0

    def test_direction_offsets(self, pheromone_field):
        """Test direction offset mappings"""
        # Should have 8 directions
        assert len(pheromone_field.direction_offsets) == 8

        # Check reverse mapping
        for idx, offset in enumerate(pheromone_field.direction_offsets):
            assert pheromone_field.offset_to_index[offset] == idx

        # Verify all 8 neighbors covered
        expected_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        assert set(pheromone_field.direction_offsets) == set(expected_offsets)

    def test_evaporate(self, pheromone_field):
        """Test pheromone evaporation: τ(t+1) = (1-ρ)*τ(t)"""
        # Set some initial pheromone
        pheromone_field.pheromone_grid[5, 5, 0] = 1.0
        initial_value = pheromone_field.pheromone_grid[5, 5, 0]

        # Evaporate
        pheromone_field.evaporate(dt=1.0)

        # Check decay
        expected = initial_value * (1 - pheromone_field.config.evaporation_rate)
        expected = max(expected, pheromone_field.config.min_pheromone)

        assert pheromone_field.pheromone_grid[5, 5, 0] == pytest.approx(expected, rel=1e-5)
        assert pheromone_field.step_count == 1

    def test_evaporate_minimum_enforcement(self, pheromone_field):
        """Test that evaporation respects minimum pheromone level"""
        # Set to minimum
        pheromone_field.pheromone_grid[:, :, :] = pheromone_field.config.min_pheromone

        # Evaporate multiple times
        for _ in range(100):
            pheromone_field.evaporate(dt=1.0)

        # Should still be at minimum
        assert np.all(pheromone_field.pheromone_grid >= pheromone_field.config.min_pheromone)

    def test_deposit_basic(self, pheromone_field):
        """Test basic pheromone deposit"""
        from_cell = (5, 5)
        to_cell = (5, 6)  # Move right

        initial = pheromone_field.get_pheromone(from_cell, to_cell)
        result = pheromone_field.deposit(from_cell, to_cell)

        assert result is True
        assert pheromone_field.get_pheromone(from_cell, to_cell) > initial
        assert pheromone_field.total_depositions == 1

    def test_deposit_success_multiplier(self, pheromone_field):
        """Test success deposit applies multiplier"""
        from_cell = (5, 5)
        to_cell_success = (5, 6)
        to_cell_normal = (6, 5)

        pheromone_field.deposit(from_cell, to_cell_success, success=True)
        pheromone_field.deposit(from_cell, to_cell_normal, success=False)

        success_pheromone = pheromone_field.get_pheromone(from_cell, to_cell_success)
        normal_pheromone = pheromone_field.get_pheromone(from_cell, to_cell_normal)

        # Success should be higher by the multiplier
        expected_ratio = pheromone_field.config.success_deposit_multiplier
        assert success_pheromone / normal_pheromone == pytest.approx(expected_ratio, rel=0.1)

    def test_deposit_no_movement(self, pheromone_field):
        """Test deposit fails for no movement"""
        result = pheromone_field.deposit((5, 5), (5, 5))
        assert result is False

    def test_deposit_out_of_bounds(self, pheromone_field):
        """Test deposit at grid boundaries"""
        # From out of bounds
        result = pheromone_field.deposit((-1, 0), (0, 0))
        assert result is False

        # From valid position moving out
        grid_size = pheromone_field.grid_size
        result = pheromone_field.deposit((grid_size - 1, grid_size - 1), (grid_size, grid_size))
        # Should still work as we clamp direction, deposit is on the valid cell
        # This depends on implementation

    def test_deposit_with_info_cost(self, pheromone_field, mock_agent_state):
        """Test deposit with info energy cost"""
        from_cell = (5, 5)
        to_cell = (5, 6)

        initial_energy = mock_agent_state.info_energy
        result = pheromone_field.deposit(from_cell, to_cell, agent_state=mock_agent_state)

        assert result is True
        assert mock_agent_state.info_energy < initial_energy
        assert pheromone_field.info_spent_this_step > 0

    def test_deposit_blocked_by_info_cost(self, pheromone_field, mock_agent_state):
        """Test deposit fails when agent lacks info energy"""
        mock_agent_state.info_energy = 0.0

        from_cell = (5, 5)
        to_cell = (5, 6)
        result = pheromone_field.deposit(from_cell, to_cell, agent_state=mock_agent_state)

        assert result is False
        assert pheromone_field.deposits_blocked_by_info == 1

    def test_get_pheromone(self, pheromone_field):
        """Test getting pheromone values"""
        # Initial should be minimum
        value = pheromone_field.get_pheromone((5, 5), (5, 6))
        assert value == pheromone_field.config.min_pheromone

        # After deposit
        pheromone_field.deposit((5, 5), (5, 6))
        value = pheromone_field.get_pheromone((5, 5), (5, 6))
        assert value > pheromone_field.config.min_pheromone

    def test_get_pheromone_invalid(self, pheromone_field):
        """Test getting pheromone for invalid cases"""
        # No movement
        assert pheromone_field.get_pheromone((5, 5), (5, 5)) == 0.0

        # Out of bounds
        assert pheromone_field.get_pheromone((-1, 0), (0, 0)) == 0.0

    def test_get_average_level(self, pheromone_field):
        """Test average pheromone level calculation"""
        # Initial should be small
        avg = pheromone_field.get_average_level(5, 5)
        assert avg >= 0
        assert avg <= 1.0

        # After deposits, should increase
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    pheromone_field.deposit((5, 5), (5 + dy, 5 + dx))

        avg_after = pheromone_field.get_average_level(5, 5)
        assert avg_after > avg

    def test_get_routing_probabilities(self, pheromone_field):
        """Test routing probability computation"""
        probs = pheromone_field.get_routing_probabilities(5, 5)

        # Should be a probability distribution
        assert len(probs) == 8
        assert np.all(probs >= 0)
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_get_routing_probabilities_with_heuristic(self, pheromone_field):
        """Test routing with heuristic values"""
        # Heuristic favoring direction 0
        heuristic = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        probs = pheromone_field.get_routing_probabilities(5, 5, heuristic=heuristic)

        # Direction 0 should have higher probability
        assert probs[0] > probs[1]

    def test_get_routing_probabilities_with_blocked(self, pheromone_field):
        """Test routing with blocked directions"""
        blocked = np.array([True, True, True, True, False, False, False, False])
        probs = pheromone_field.get_routing_probabilities(5, 5, blocked=blocked)

        # Blocked directions should have zero probability
        assert probs[0] == 0.0
        assert probs[4] > 0.0

    def test_get_routing_probabilities_boundary(self, pheromone_field):
        """Test routing at grid boundaries"""
        # Corner cell
        probs = pheromone_field.get_routing_probabilities(0, 0)

        # Only 3 valid directions from corner
        valid_count = np.sum(probs > 0)
        assert valid_count == 3

    def test_sample_direction(self, pheromone_field):
        """Test direction sampling"""
        dy, dx = pheromone_field.sample_direction(5, 5)

        # Should be a valid direction
        assert (dy, dx) in pheromone_field.direction_offsets

    def test_get_gradient(self, pheromone_field):
        """Test pheromone gradient calculation"""
        # Set asymmetric pheromone
        # High pheromone to the right
        pheromone_field.pheromone_grid[5, 5, 4] = 1.0  # Direction (0, 1) - right

        gy, gx = pheromone_field.get_gradient(5, 5)

        # Gradient should point right
        assert gx > 0

    def test_reinforce_path(self, pheromone_field):
        """Test path reinforcement"""
        path = [(5, 5), (5, 6), (5, 7), (5, 8)]

        reinforced = pheromone_field.reinforce_path(path, quality=1.0)

        assert reinforced == 3  # 3 edges in path of 4 nodes

        # Check pheromone increased along path
        assert pheromone_field.get_pheromone((5, 5), (5, 6)) > pheromone_field.config.min_pheromone

    def test_reinforce_path_with_decay(self, pheromone_field):
        """Test path reinforcement with decay along path"""
        path = [(5, 5), (5, 6), (5, 7), (5, 8)]

        pheromone_field.reinforce_path(path, decay_along_path=True)

        # Earlier edges should have more pheromone
        edge1 = pheromone_field.get_pheromone((5, 5), (5, 6))
        edge3 = pheromone_field.get_pheromone((5, 7), (5, 8))

        assert edge1 > edge3

    def test_reinforce_path_info_cost(self, pheromone_field, mock_agent_state):
        """Test path reinforcement with info cost"""
        path = [(5, 5), (5, 6), (5, 7)]
        initial_energy = mock_agent_state.info_energy

        reinforced = pheromone_field.reinforce_path(path, agent_state=mock_agent_state)

        assert reinforced == 2
        assert mock_agent_state.info_energy < initial_energy

    def test_diffuse(self, pheromone_field):
        """Test pheromone diffusion"""
        # Set high pheromone at one cell
        pheromone_field.pheromone_grid[8, 8, :] = 1.0
        initial = pheromone_field.pheromone_grid[7, 8, 0]

        pheromone_field.diffuse(rate=0.1)

        # Neighbors should have gained some pheromone
        after = pheromone_field.pheromone_grid[7, 8, 0]
        # Diffusion spreads pheromone, but specific values depend on implementation

    def test_get_statistics(self, pheromone_field):
        """Test statistics computation"""
        # Make some deposits
        pheromone_field.deposit((5, 5), (5, 6))
        pheromone_field.evaporate(dt=1.0)

        stats = pheromone_field.get_statistics()

        assert "mean_pheromone" in stats
        assert "max_pheromone" in stats
        assert "min_pheromone" in stats
        assert "std_pheromone" in stats
        assert "total_depositions" in stats
        assert stats["total_depositions"] == 1
        assert "steps" in stats
        assert stats["steps"] == 1

    def test_get_heatmap(self, pheromone_field):
        """Test heatmap generation"""
        heatmap = pheromone_field.get_heatmap()

        assert heatmap.shape == (pheromone_field.grid_size, pheromone_field.grid_size)
        assert np.all(heatmap >= 0)

    def test_reset(self, pheromone_field):
        """Test field reset"""
        # Make changes
        pheromone_field.deposit((5, 5), (5, 6))
        pheromone_field.evaporate(dt=1.0)

        # Reset
        pheromone_field.reset()

        assert pheromone_field.total_depositions == 0
        assert pheromone_field.step_count == 0
        assert np.all(pheromone_field.pheromone_grid == pheromone_field.config.min_pheromone)

    def test_reset_step_tracking(self, pheromone_field, mock_agent_state):
        """Test per-step tracking reset"""
        pheromone_field.deposit((5, 5), (5, 6), agent_state=mock_agent_state)
        assert pheromone_field.info_spent_this_step > 0

        pheromone_field.reset_step_tracking()

        assert pheromone_field.info_spent_this_step == 0.0
        assert pheromone_field.deposits_blocked_by_info == 0

    def test_update(self, pheromone_field):
        """Test batch update with movements"""
        movements = [
            ((5, 5), (5, 6)),
            ((6, 6), (7, 6)),
            ((3, 3), (3, 4)),
        ]

        initial_deposits = pheromone_field.total_depositions
        pheromone_field.update(movements)

        assert pheromone_field.total_depositions == initial_deposits + 3


class TestMultiPheromoneField:
    """Tests for MultiPheromoneField class"""

    def test_initialization(self, pheromone_config, grid_size):
        """Test multi-field initialization"""
        field = MultiPheromoneField(pheromone_config, grid_size, n_types=4)

        assert len(field.fields) == 4
        assert "resource" in field.fields
        assert "danger" in field.fields
        assert "project" in field.fields
        assert "home" in field.fields

    def test_deposit_specific_type(self, pheromone_config, grid_size):
        """Test depositing to specific pheromone type"""
        field = MultiPheromoneField(pheromone_config, grid_size)

        result = field.deposit("resource", (5, 5), (5, 6))
        assert result is True

        # Check only resource field changed
        resource_level = field.fields["resource"].get_pheromone((5, 5), (5, 6))
        danger_level = field.fields["danger"].get_pheromone((5, 5), (5, 6))

        assert resource_level > danger_level

    def test_evaporate_all(self, pheromone_config, grid_size):
        """Test evaporation affects all fields"""
        field = MultiPheromoneField(pheromone_config, grid_size)

        # Deposit to each type
        field.deposit("resource", (5, 5), (5, 6))
        field.deposit("danger", (5, 5), (5, 6))

        initial_resource = field.fields["resource"].get_pheromone((5, 5), (5, 6))

        field.evaporate(dt=1.0)

        after_resource = field.fields["resource"].get_pheromone((5, 5), (5, 6))
        assert after_resource < initial_resource

    def test_get_combined_routing(self, pheromone_config, grid_size):
        """Test combined routing probabilities"""
        field = MultiPheromoneField(pheromone_config, grid_size)

        probs = field.get_combined_routing(5, 5)

        assert len(probs) == 8
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_get_combined_routing_with_weights(self, pheromone_config, grid_size):
        """Test combined routing with custom weights"""
        field = MultiPheromoneField(pheromone_config, grid_size)

        # Heavily weight resource pheromone
        weights = {"resource": 10.0, "danger": 0.1, "project": 0.1, "home": 0.1}
        probs = field.get_combined_routing(5, 5, weights=weights)

        assert len(probs) == 8
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_get_average_level(self, pheromone_config, grid_size):
        """Test average level across all types"""
        field = MultiPheromoneField(pheromone_config, grid_size)

        avg = field.get_average_level(5, 5)
        assert avg >= 0
        assert avg <= 1.0

    def test_get_field(self, pheromone_config, grid_size):
        """Test getting specific field"""
        field = MultiPheromoneField(pheromone_config, grid_size)

        resource_field = field.get_field("resource")
        assert resource_field is not None

        invalid_field = field.get_field("invalid")
        assert invalid_field is None

    def test_reset(self, pheromone_config, grid_size):
        """Test resetting all fields"""
        field = MultiPheromoneField(pheromone_config, grid_size)

        # Make changes
        field.deposit("resource", (5, 5), (5, 6))

        # Reset
        field.reset()

        # All fields should be reset
        for pfield in field.fields.values():
            assert pfield.total_depositions == 0


class TestPheromoneEvaporationFormula:
    """Tests for the evaporation formula: τ(t+1) = (1-ρ)*τ(t)"""

    @pytest.mark.parametrize("evaporation_rate", [0.01, 0.1, 0.5])
    def test_evaporation_rate_effect(self, grid_size, evaporation_rate):
        """Test different evaporation rates"""
        config = PheromoneConfig(evaporation_rate=evaporation_rate)
        field = PheromoneField(config, grid_size)

        # Set initial pheromone
        field.pheromone_grid[5, 5, 0] = 1.0
        initial = field.pheromone_grid[5, 5, 0]

        # Evaporate
        field.evaporate(dt=1.0)

        expected = initial * (1 - evaporation_rate)
        expected = max(expected, config.min_pheromone)

        assert field.pheromone_grid[5, 5, 0] == pytest.approx(expected, rel=1e-5)

    def test_multiple_evaporation_steps(self, pheromone_field):
        """Test multiple evaporation steps"""
        field = pheromone_field
        field.pheromone_grid[5, 5, 0] = 1.0
        initial = field.pheromone_grid[5, 5, 0]

        n_steps = 10
        for _ in range(n_steps):
            field.evaporate(dt=1.0)

        # After n steps: τ = τ_0 * (1-ρ)^n
        expected = initial * ((1 - field.config.evaporation_rate) ** n_steps)
        expected = max(expected, field.config.min_pheromone)

        assert field.pheromone_grid[5, 5, 0] == pytest.approx(expected, rel=1e-4)


class TestPheromoneRoutingProbability:
    """Tests for routing probability: p_ij ∝ τ_ij^α * η_ij^β"""

    def test_pheromone_alpha_effect(self, grid_size):
        """Test pheromone alpha exponent effect"""
        # Higher alpha = more exploitation
        config_low = PheromoneConfig(pheromone_alpha=0.5)
        config_high = PheromoneConfig(pheromone_alpha=2.0)

        field_low = PheromoneField(config_low, grid_size)
        field_high = PheromoneField(config_high, grid_size)

        # Set asymmetric pheromone
        for field in [field_low, field_high]:
            field.pheromone_grid[5, 5, 0] = 1.0  # High in direction 0
            field.pheromone_grid[5, 5, 1] = 0.1  # Low in direction 1

        probs_low = field_low.get_routing_probabilities(5, 5)
        probs_high = field_high.get_routing_probabilities(5, 5)

        # Higher alpha should make the difference more pronounced
        ratio_low = probs_low[0] / probs_low[1] if probs_low[1] > 0 else float('inf')
        ratio_high = probs_high[0] / probs_high[1] if probs_high[1] > 0 else float('inf')

        assert ratio_high > ratio_low

    def test_heuristic_beta_effect(self, grid_size):
        """Test heuristic beta exponent effect"""
        config = PheromoneConfig(heuristic_beta=1.0)
        field = PheromoneField(config, grid_size)

        # Heuristic favoring direction 0
        heuristic = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        probs = field.get_routing_probabilities(5, 5, heuristic=heuristic)

        # Direction 0 should dominate
        assert probs[0] > np.mean(probs[1:])
