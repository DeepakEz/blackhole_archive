"""
Clustered Resource Generator
============================
Gaussian Mixture Model-based resource distribution for exploration pressure.

Creates spatially clustered resources (vegetation, food, water features)
that force agents to navigate and explore rather than staying in one place.

Pattern reference: formal_lyapunov_stability.py - creating structured
distributions that encourage emergent behavior through constraints.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ClusterParams:
    """Parameters for a single resource cluster"""
    center: Tuple[float, float]  # (y, x) normalized [0, 1]
    std: float                    # Standard deviation (spread)
    weight: float                 # Peak amplitude [0, 1]


class ClusteredResourceGenerator:
    """
    Gaussian Mixture Model resource generator.

    Creates spatially clustered resource distributions that:
    1. Force exploration (resources aren't uniform)
    2. Create natural gathering points (cluster centers)
    3. Maintain ecological realism (patches of vegetation)

    The GMM approach ensures smooth, continuous distributions
    rather than discrete blobs.

    Usage:
        generator = ClusteredResourceGenerator(grid_size=64, n_clusters=3)
        vegetation = generator.generate_vegetation()
    """

    def __init__(self,
                 grid_size: int,
                 n_clusters: int = 3,
                 seed: Optional[int] = None):
        """
        Initialize clustered resource generator.

        Args:
            grid_size: Size of the grid (assumes square)
            n_clusters: Number of resource clusters (default 3)
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.n_clusters = n_clusters
        self.rng = np.random.default_rng(seed)

        # Generate cluster parameters
        self.clusters = self._generate_cluster_params()

    def _generate_cluster_params(self) -> List[ClusterParams]:
        """Generate random cluster parameters with spatial separation"""
        clusters = []

        # Minimum separation between cluster centers (as fraction of grid)
        min_separation = 0.25

        for i in range(self.n_clusters):
            # Try to find a well-separated center
            max_attempts = 50
            for attempt in range(max_attempts):
                # Random center in [0.1, 0.9] to avoid edges
                center_y = self.rng.uniform(0.15, 0.85)
                center_x = self.rng.uniform(0.15, 0.85)

                # Check separation from existing clusters
                well_separated = True
                for existing in clusters:
                    dist = np.sqrt(
                        (center_y - existing.center[0])**2 +
                        (center_x - existing.center[1])**2
                    )
                    if dist < min_separation:
                        well_separated = False
                        break

                if well_separated:
                    break

            # Cluster spread: 8-15% of grid size
            std = self.rng.uniform(0.08, 0.15)

            # Cluster intensity: 60-100% peak
            weight = self.rng.uniform(0.6, 1.0)

            clusters.append(ClusterParams(
                center=(center_y, center_x),
                std=std,
                weight=weight
            ))

        return clusters

    def _gaussian_2d(self,
                     y: np.ndarray,
                     x: np.ndarray,
                     center_y: float,
                     center_x: float,
                     std: float) -> np.ndarray:
        """
        Compute 2D Gaussian distribution.

        Args:
            y, x: Coordinate grids (normalized [0, 1])
            center_y, center_x: Center of Gaussian
            std: Standard deviation

        Returns:
            Gaussian values at each point
        """
        dist_sq = (y - center_y)**2 + (x - center_x)**2
        return np.exp(-dist_sq / (2 * std**2))

    def generate_vegetation(self,
                           base_level: float = 0.1,
                           water_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate clustered vegetation distribution.

        This is the main entry point for environment integration.

        Args:
            base_level: Minimum vegetation everywhere (sparse grass)
            water_mask: Optional boolean mask where water is deep (vegetation reduced)

        Returns:
            Vegetation array [0, 1] with clusters
        """
        size = self.grid_size

        # Create coordinate grids normalized to [0, 1]
        y_coords = np.linspace(0, 1, size)
        x_coords = np.linspace(0, 1, size)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

        # Start with base level
        vegetation = np.full((size, size), base_level)

        # Add each cluster as a Gaussian component
        for cluster in self.clusters:
            gaussian = self._gaussian_2d(
                Y, X,
                cluster.center[0],
                cluster.center[1],
                cluster.std
            )
            # Weight by cluster intensity
            vegetation += cluster.weight * gaussian

        # Add noise for realism (small patches)
        noise = self.rng.uniform(-0.05, 0.1, (size, size))
        vegetation += noise

        # Apply water mask if provided (reduce vegetation in water)
        if water_mask is not None:
            vegetation = np.where(water_mask, 0.1 * vegetation, vegetation)

        # Clamp to [0, 1]
        vegetation = np.clip(vegetation, 0.0, 1.0)

        return vegetation

    def generate_food_sources(self, n_sources: int = 5) -> np.ndarray:
        """
        Generate discrete food source locations.

        Unlike vegetation (continuous), food sources are discrete
        high-value points that agents must travel to.

        Args:
            n_sources: Number of food source locations

        Returns:
            Binary mask with food source locations
        """
        size = self.grid_size
        food_map = np.zeros((size, size), dtype=np.float32)

        for _ in range(n_sources):
            # Random location
            y = self.rng.integers(5, size - 5)
            x = self.rng.integers(5, size - 5)

            # Small cluster around source
            radius = self.rng.integers(2, 5)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < size and 0 <= nx < size:
                        dist = np.sqrt(dy**2 + dx**2)
                        if dist <= radius:
                            food_map[ny, nx] = max(
                                food_map[ny, nx],
                                1.0 - dist / radius
                            )

        return food_map

    def get_cluster_centers(self) -> List[Tuple[int, int]]:
        """
        Get cluster center positions in grid coordinates.

        Useful for spawning, pathfinding, or visualization.

        Returns:
            List of (y, x) positions in grid coordinates
        """
        centers = []
        for cluster in self.clusters:
            y = int(cluster.center[0] * self.grid_size)
            x = int(cluster.center[1] * self.grid_size)
            # Clamp to valid range
            y = max(0, min(self.grid_size - 1, y))
            x = max(0, min(self.grid_size - 1, x))
            centers.append((y, x))
        return centers

    def regenerate_clusters(self, seed: Optional[int] = None):
        """
        Regenerate cluster positions with new random seed.

        Call this in environment reset() to get new cluster layouts.

        Args:
            seed: Optional new random seed
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.clusters = self._generate_cluster_params()


# ============================================================================
# CONVENIENCE FUNCTION FOR ENVIRONMENT INTEGRATION
# ============================================================================

def create_clustered_vegetation(grid_size: int,
                                n_clusters: int = 3,
                                seed: Optional[int] = None,
                                water_depth: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convenience function to generate clustered vegetation.

    Drop-in replacement for uniform vegetation initialization.

    Usage in environment reset():
        # Old:
        self.vegetation = 0.5 * np.ones((grid_size, grid_size))

        # New:
        from mycobeaver.clustered_resources import create_clustered_vegetation
        self.vegetation = create_clustered_vegetation(grid_size, n_clusters=3)

    Args:
        grid_size: Size of the grid
        n_clusters: Number of vegetation clusters (default 3)
        seed: Random seed for reproducibility
        water_depth: Optional water depth array to reduce vegetation in water

    Returns:
        Vegetation array with clusters
    """
    # Create water mask if water_depth provided
    water_mask = None
    if water_depth is not None:
        water_mask = water_depth > 0.5

    generator = ClusteredResourceGenerator(
        grid_size=grid_size,
        n_clusters=n_clusters,
        seed=seed
    )

    return generator.generate_vegetation(
        base_level=0.1,
        water_mask=water_mask
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Clustered Resource Generator Test")
    print("=" * 60)

    # Test basic generation
    generator = ClusteredResourceGenerator(
        grid_size=64,
        n_clusters=3,
        seed=42
    )

    print(f"\nGenerated {len(generator.clusters)} clusters:")
    for i, cluster in enumerate(generator.clusters):
        print(f"  Cluster {i}: center=({cluster.center[0]:.2f}, {cluster.center[1]:.2f}), "
              f"std={cluster.std:.2f}, weight={cluster.weight:.2f}")

    # Generate vegetation
    vegetation = generator.generate_vegetation()
    print(f"\nVegetation stats:")
    print(f"  Shape: {vegetation.shape}")
    print(f"  Min: {vegetation.min():.3f}")
    print(f"  Max: {vegetation.max():.3f}")
    print(f"  Mean: {vegetation.mean():.3f}")

    # Check cluster centers
    centers = generator.get_cluster_centers()
    print(f"\nCluster centers (grid coords):")
    for i, (y, x) in enumerate(centers):
        print(f"  Cluster {i}: ({y}, {x}), vegetation={vegetation[y, x]:.2f}")

    # Test convenience function
    veg2 = create_clustered_vegetation(64, n_clusters=5, seed=123)
    print(f"\nConvenience function result:")
    print(f"  Shape: {veg2.shape}")
    print(f"  Mean: {veg2.mean():.3f}")

    print("\nTest complete!")
