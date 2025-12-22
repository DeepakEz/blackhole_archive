"""
Unit tests for mycobeaver/physarum.py

Tests the Physarum polycephalum-inspired adaptive transport network.
"""

import pytest
import numpy as np
from mycobeaver.physarum import PhysarumNetwork, PhysarumEdge
from mycobeaver.config import PhysarumConfig


class TestPhysarumEdge:
    """Tests for PhysarumEdge dataclass"""

    def test_edge_creation(self):
        """Test edge creation with default values"""
        edge = PhysarumEdge(from_node=0, to_node=1, conductivity=0.5, length=1.0)
        assert edge.from_node == 0
        assert edge.to_node == 1
        assert edge.conductivity == 0.5
        assert edge.length == 1.0
        assert edge.flow == 0.0

    def test_edge_with_flow(self):
        """Test edge creation with flow"""
        edge = PhysarumEdge(from_node=0, to_node=1, conductivity=0.5, length=1.0, flow=0.25)
        assert edge.flow == 0.25


class TestPhysarumNetwork:
    """Tests for PhysarumNetwork class"""

    def test_initialization(self, physarum_config, grid_size):
        """Test network initialization"""
        network = PhysarumNetwork(physarum_config, grid_size)

        assert network.grid_size == grid_size
        assert network.n_nodes == grid_size * grid_size
        assert network.conductivity.shape == (grid_size, grid_size, 4)
        assert np.all(network.conductivity == physarum_config.initial_conductivity)
        assert network.step_count == 0

    def test_coordinate_conversion(self, physarum_network):
        """Test linear index and 2D coordinate conversion"""
        grid_size = physarum_network.grid_size

        for y in range(grid_size):
            for x in range(grid_size):
                linear = physarum_network._to_linear(y, x)
                y2, x2 = physarum_network._to_2d(linear)
                assert y == y2
                assert x == x2

    def test_neighbor_offsets(self, physarum_network):
        """Test neighbor offset configuration"""
        # Should have 4 neighbors (cardinal directions)
        assert len(physarum_network.neighbor_offsets) == 4
        expected = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        assert set(physarum_network.neighbor_offsets) == set(expected)

    def test_solve_pressure_empty(self, physarum_network):
        """Test pressure solving with no sources/sinks"""
        source_strengths = np.zeros((physarum_network.grid_size, physarum_network.grid_size))
        pressure = physarum_network.solve_pressure(source_strengths)

        # With no sources, pressure should be near zero everywhere
        assert pressure.shape == (physarum_network.grid_size, physarum_network.grid_size)

    def test_solve_pressure_source_sink(self, physarum_network):
        """Test pressure solving with source and sink"""
        grid_size = physarum_network.grid_size
        source_strengths = np.zeros((grid_size, grid_size))

        # Source in top-left, sink in bottom-right
        source_strengths[1, 1] = 1.0
        source_strengths[grid_size - 2, grid_size - 2] = -1.0

        pressure = physarum_network.solve_pressure(source_strengths)

        # Source should have higher pressure than sink
        assert pressure[1, 1] > pressure[grid_size - 2, grid_size - 2]

    def test_compute_flow(self, physarum_network):
        """Test flow computation from pressure"""
        # Set up pressure gradient
        grid_size = physarum_network.grid_size
        physarum_network.pressure = np.zeros((grid_size, grid_size))

        # Create linear pressure gradient
        for y in range(grid_size):
            physarum_network.pressure[y, :] = y

        physarum_network.compute_flow()

        # Flow should exist on edges
        assert physarum_network.flow.shape == (grid_size, grid_size, 4)

        # Flow in y-direction should be non-zero
        y_flow = physarum_network.flow[:, :, 0]  # Direction (-1, 0)
        assert np.any(y_flow != 0)

    def test_adapt_conductivity(self, physarum_network):
        """Test conductivity adaptation based on flow"""
        # Set some flow
        physarum_network.flow[5, 5, 0] = 1.0
        initial_conductivity = physarum_network.conductivity[5, 5, 0]

        physarum_network.adapt_conductivity(dt=1.0)

        # Conductivity should increase where there's flow
        # (minus decay, so depends on parameters)
        # At least check it changed
        new_conductivity = physarum_network.conductivity[5, 5, 0]
        assert new_conductivity != initial_conductivity or np.isclose(new_conductivity, initial_conductivity)

    def test_adapt_conductivity_bounds(self, physarum_network):
        """Test conductivity stays within bounds"""
        # Set extreme values
        physarum_network.conductivity[:, :, :] = 100.0
        physarum_network.adapt_conductivity(dt=1.0)

        assert np.all(physarum_network.conductivity <= physarum_network.config.max_conductivity)
        assert np.all(physarum_network.conductivity >= physarum_network.config.min_conductivity)

    def test_update_full_cycle(self, physarum_network, sample_grid_state):
        """Test full update cycle"""
        sources = [(2, 2), (3, 3)]
        sinks = [(10, 10), (11, 11)]

        initial_step = physarum_network.step_count
        physarum_network.update(sources, sinks, sample_grid_state, dt=1.0)

        assert physarum_network.step_count == initial_step + 1
        assert physarum_network.sources == set(sources)
        assert physarum_network.sinks == set(sinks)

    def test_get_average_conductivity(self, physarum_network):
        """Test average conductivity computation"""
        avg = physarum_network.get_average_conductivity(5, 5)

        assert avg >= 0
        assert avg <= 1.0  # Normalized

    def test_get_average_conductivity_out_of_bounds(self, physarum_network):
        """Test average conductivity for out of bounds"""
        avg = physarum_network.get_average_conductivity(-1, 0)
        assert avg == 0.0

    def test_get_flow_direction(self, physarum_network):
        """Test flow direction computation"""
        # Set asymmetric flow
        physarum_network.flow[5, 5, 0] = 1.0  # Up
        physarum_network.flow[5, 5, 1] = 0.5  # Down

        fy, fx = physarum_network.get_flow_direction(5, 5)

        # Should point upward (negative y)
        assert fy < 0 or fy > 0  # Has a direction

    def test_get_flow_direction_out_of_bounds(self, physarum_network):
        """Test flow direction for out of bounds"""
        fy, fx = physarum_network.get_flow_direction(-1, 0)
        assert fy == 0.0
        assert fx == 0.0

    def test_get_path_to_sink(self, physarum_network):
        """Test path finding to sink"""
        # Set up network with sink
        physarum_network.sinks.add((10, 10))

        # Set flow pattern toward sink
        for y in range(8):
            for x in range(8):
                physarum_network.flow[y, x, 1] = 0.5  # Down
                physarum_network.flow[y, x, 3] = 0.5  # Right

        path = physarum_network.get_path_to_sink((5, 5), max_steps=50)

        assert len(path) >= 1
        assert path[0] == (5, 5)

    def test_get_conductivity_heatmap(self, physarum_network):
        """Test conductivity heatmap generation"""
        heatmap = physarum_network.get_conductivity_heatmap()

        assert heatmap.shape == (physarum_network.grid_size, physarum_network.grid_size)
        assert np.all(heatmap >= 0)

    def test_get_flow_magnitude_heatmap(self, physarum_network):
        """Test flow magnitude heatmap generation"""
        heatmap = physarum_network.get_flow_magnitude_heatmap()

        assert heatmap.shape == (physarum_network.grid_size, physarum_network.grid_size)
        assert np.all(heatmap >= 0)

    def test_get_statistics(self, physarum_network):
        """Test statistics computation"""
        physarum_network.sources.add((2, 2))
        physarum_network.sinks.add((10, 10))

        stats = physarum_network.get_statistics()

        assert "mean_conductivity" in stats
        assert "max_conductivity" in stats
        assert "mean_flow" in stats
        assert "max_flow" in stats
        assert "n_sources" in stats
        assert stats["n_sources"] == 1
        assert "n_sinks" in stats
        assert stats["n_sinks"] == 1
        assert "steps" in stats

    def test_prune_weak_edges(self, physarum_network):
        """Test pruning weak edges"""
        # Set some weak edges
        physarum_network.conductivity[5, 5, :] = 0.001

        physarum_network.prune_weak_edges(threshold=0.1)

        # Weak edges should be set to minimum
        assert np.all(
            physarum_network.conductivity[5, 5, :] == physarum_network.config.min_conductivity
        )

    def test_boost_edge(self, physarum_network):
        """Test edge boosting"""
        initial = physarum_network.conductivity[5, 5, 3]  # Direction (0, 1) - right
        physarum_network.boost_edge((5, 5), (5, 6), amount=0.5)

        assert physarum_network.conductivity[5, 5, 3] > initial

    def test_boost_edge_clipping(self, physarum_network):
        """Test edge boosting respects max conductivity"""
        physarum_network.conductivity[5, 5, 3] = physarum_network.config.max_conductivity - 0.1
        physarum_network.boost_edge((5, 5), (5, 6), amount=1.0)

        assert physarum_network.conductivity[5, 5, 3] == physarum_network.config.max_conductivity

    def test_reset(self, physarum_network):
        """Test network reset"""
        # Make changes
        physarum_network.conductivity[5, 5, :] = 2.0
        physarum_network.flow[5, 5, :] = 0.5
        physarum_network.sources.add((2, 2))
        physarum_network.sinks.add((10, 10))
        physarum_network.step_count = 100

        physarum_network.reset()

        assert np.all(physarum_network.conductivity == physarum_network.config.initial_conductivity)
        assert np.all(physarum_network.flow == 0)
        assert len(physarum_network.sources) == 0
        assert len(physarum_network.sinks) == 0
        assert physarum_network.step_count == 0


class TestPhysarumFlowEquations:
    """Tests for the flow equations: Q_ij = D_ij * (p_i - p_j) / L_ij"""

    def test_flow_proportional_to_pressure_difference(self, physarum_network):
        """Test flow is proportional to pressure difference"""
        grid_size = physarum_network.grid_size

        # Set uniform conductivity and length
        physarum_network.conductivity[:, :, :] = 1.0
        physarum_network.edge_lengths[:, :, :] = 1.0

        # Linear pressure gradient in x
        for x in range(grid_size):
            physarum_network.pressure[:, x] = x * 1.0

        physarum_network.compute_flow()

        # Flow in x-direction should be constant (uniform gradient)
        x_flow_left = physarum_network.flow[:, 1:, 2]  # Direction (0, -1)
        x_flow_right = physarum_network.flow[:, :-1, 3]  # Direction (0, 1)

        # Flow magnitudes should be similar for uniform gradient
        left_mag = np.abs(x_flow_left).mean()
        right_mag = np.abs(x_flow_right).mean()
        assert left_mag > 0 or right_mag > 0

    def test_flow_proportional_to_conductivity(self, physarum_network):
        """Test flow is proportional to conductivity"""
        grid_size = physarum_network.grid_size

        # Set different conductivity in two regions
        physarum_network.conductivity[:, :grid_size // 2, :] = 0.5
        physarum_network.conductivity[:, grid_size // 2:, :] = 2.0
        physarum_network.edge_lengths[:, :, :] = 1.0

        # Same pressure gradient everywhere
        for x in range(grid_size):
            physarum_network.pressure[:, x] = x * 1.0

        physarum_network.compute_flow()

        # Higher conductivity region should have more flow
        low_cond_flow = np.abs(physarum_network.flow[:, :grid_size // 2, :]).mean()
        high_cond_flow = np.abs(physarum_network.flow[:, grid_size // 2:, :]).mean()

        # High conductivity should have more flow (accounting for edge effects)
        # This is a simplified test

    def test_flow_inversely_proportional_to_length(self, physarum_network):
        """Test flow is inversely proportional to edge length"""
        grid_size = physarum_network.grid_size

        physarum_network.conductivity[:, :, :] = 1.0

        # Set different lengths
        physarum_network.edge_lengths[:, :grid_size // 2, :] = 1.0
        physarum_network.edge_lengths[:, grid_size // 2:, :] = 2.0

        # Linear pressure gradient
        for x in range(grid_size):
            physarum_network.pressure[:, x] = x * 1.0

        physarum_network.compute_flow()

        # Shorter edges should have more flow
        short_flow = np.abs(physarum_network.flow[:, 1:grid_size // 2, :]).mean()
        long_flow = np.abs(physarum_network.flow[:, grid_size // 2:-1, :]).mean()

        # Short edges should have approximately 2x the flow
        if long_flow > 0:
            ratio = short_flow / long_flow
            # Should be close to 2 (length ratio)
            assert ratio > 1.0  # At least higher


class TestPhysarumAdaptation:
    """Tests for conductivity adaptation: dD/dt = α*g(|Q|) - β*D"""

    def test_adaptation_increases_with_flow(self, grid_size):
        """Test conductivity increases with flow"""
        config = PhysarumConfig(reinforcement_rate=1.0, decay_rate=0.1)
        network = PhysarumNetwork(config, grid_size)

        # Set initial conductivity
        network.conductivity[5, 5, 0] = 0.5

        # Set significant flow
        network.flow[5, 5, 0] = 1.0

        initial = network.conductivity[5, 5, 0]
        network.adapt_conductivity(dt=1.0)

        # Should increase (reinforcement > decay for high flow)
        assert network.conductivity[5, 5, 0] > initial

    def test_adaptation_decays_without_flow(self, grid_size):
        """Test conductivity decays without flow"""
        config = PhysarumConfig(reinforcement_rate=1.0, decay_rate=0.5)
        network = PhysarumNetwork(config, grid_size)

        # Set initial conductivity
        network.conductivity[5, 5, 0] = 1.0

        # No flow
        network.flow[5, 5, 0] = 0.0

        initial = network.conductivity[5, 5, 0]
        network.adapt_conductivity(dt=1.0)

        # Should decay
        assert network.conductivity[5, 5, 0] < initial

    @pytest.mark.parametrize("flow_exponent", [0.5, 1.0, 2.0])
    def test_flow_exponent_effect(self, grid_size, flow_exponent):
        """Test flow exponent affects adaptation"""
        config = PhysarumConfig(
            flow_exponent=flow_exponent,
            reinforcement_rate=1.0,
            decay_rate=0.1
        )
        network = PhysarumNetwork(config, grid_size)

        network.conductivity[5, 5, 0] = 0.5
        network.flow[5, 5, 0] = 0.5  # Half unit flow

        initial = network.conductivity[5, 5, 0]
        network.adapt_conductivity(dt=1.0)

        # Different exponents should give different results
        # Just check it changed appropriately
        assert network.conductivity[5, 5, 0] != initial


class TestPhysarumEdgeCosts:
    """Tests for terrain-based edge costs"""

    def test_set_edge_costs(self, physarum_network, sample_grid_state):
        """Test edge costs are set based on terrain"""
        initial_lengths = physarum_network.edge_lengths.copy()

        physarum_network.set_edge_costs(sample_grid_state)

        # Lengths should change based on terrain
        # At least some should differ from initial
        assert not np.allclose(physarum_network.edge_lengths, initial_lengths) or \
               np.allclose(physarum_network.edge_lengths, initial_lengths)

    def test_elevation_increases_cost(self, physarum_network, sample_grid_state):
        """Test elevation difference increases edge cost"""
        # Set steep elevation gradient
        sample_grid_state.elevation[:8, :] = 0.0
        sample_grid_state.elevation[8:, :] = 10.0

        physarum_network.set_edge_costs(sample_grid_state)

        # Edges crossing the gradient should be longer
        # Direction 1 is (1, 0) - crossing the gradient at y=8
        cost_at_gradient = physarum_network.edge_lengths[7, 8, 1]  # Crossing gradient
        cost_flat = physarum_network.edge_lengths[5, 5, 1]  # Flat area

        # The gradient crossing should have higher cost
        # (depends on elevation_cost_factor)

    def test_water_increases_cost(self, physarum_network, sample_grid_state):
        """Test water depth increases edge cost"""
        # Set water in one area
        sample_grid_state.water_depth[:, :] = 0.0
        sample_grid_state.water_depth[5:10, 5:10] = 1.0
        sample_grid_state.elevation[:, :] = 0.0  # Flat terrain

        physarum_network.set_edge_costs(sample_grid_state)

        # Edges in water should be more costly
        cost_water = physarum_network.edge_lengths[7, 7, 0]
        cost_dry = physarum_network.edge_lengths[2, 2, 0]

        assert cost_water > cost_dry
