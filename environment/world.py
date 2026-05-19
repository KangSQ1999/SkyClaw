"""
SkyClaw World Module - Physical simulation environment with tidal physics engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class WorldState:
    timestep: int
    gu_positions: npt.NDArray[np.float64]
    haps_positions: npt.NDArray[np.float64]
    boundaries: tuple[float, float]
    movement_flags: npt.NDArray[np.int32]
    is_tidal_active: bool = False
    tidal_affected_count: int = 0
    emergency_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization/logging."""
        return {
            "timestep": self.timestep,
            "gu_count": len(self.gu_positions),
            "haps_count": len(self.haps_positions),
            "mobile_users": int(np.sum(self.movement_flags)),
            "stationary_users": int(len(self.movement_flags) - np.sum(self.movement_flags)),
            "is_tidal_active": self.is_tidal_active,
            "tidal_affected_count": self.tidal_affected_count,
            "emergency_active": self.is_emergency_active() if self.emergency_config else False,
        }

    def is_emergency_active(self, current_time_sec: int | None = None) -> bool:
        """Check if emergency event is active at given time."""
        if not self.emergency_config or not self.emergency_config.get("ENABLE_EMERGENCY", 0):
            return False

        start_time = self.emergency_config.get("EMERGENCY_START_TIME", 0)
        end_time = self.emergency_config.get("EMERGENCY_END_TIME", 0)

        if current_time_sec is None:
            # Use timestep as fallback (assume 5-min steps)
            current_time_sec = self.timestep * 300

        return start_time <= current_time_sec < end_time

    def get_emergency_position(self) -> list[float] | None:
        """Get emergency position if emergency is enabled."""
        if not self.emergency_config or not self.emergency_config.get("ENABLE_EMERGENCY", 0):
            return None
        return self.emergency_config.get("EMERGENCY_POS", None)


class World:


    # User distribution constants
    X_MIN: float = 300.0  # km
    X_MAX: float = 900.0  # km
    Y_MIN: float = 0.0    # km
    Y_MAX: float = 1000.0 # km

    # Movement model constants
    SPEED_MEAN: float = 5.0        # km/h (normal movement)
    SPEED_STD: float = 2.0         # km/h
    SPEED_MIN: float = 2.0         # km/h
    SPEED_MAX: float = 8.0         # km/h

    # Tidal event constants - Legacy defaults (will be overridden by config)
    TIDAL_SPEED: float = 15.0      # km/h (legacy fallback)

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize World with simulation configuration."""
        self.config = config

        # Extract spatial configuration
        space_cfg = config["simulation"]["space"]
        self.space_width: float = float(space_cfg["width"])
        self.space_height: float = float(space_cfg["height"])

        # Extract user configuration
        gu_cfg = config["ground_users"]
        self.gu_count: int = int(gu_cfg["count"])

        # Mobile user ratio from config (default 0.3)
        self.mobile_ratio: float = float(gu_cfg.get("MOBILE_USER_RATIO", 0.3))
        self.mobile_count: int = int(self.gu_count * self.mobile_ratio)
        self.stationary_count: int = self.gu_count - self.mobile_count

        # Multi-cluster Gaussian distribution configuration
        self.num_clusters: int = int(gu_cfg.get("NUM_CLUSTERS", 30))
        self.cluster_std: float = float(gu_cfg.get("CLUSTER_STD", 50.0))

        # Cluster parameters (initialized in initialize())
        self.cluster_centers: npt.NDArray[np.float64] | None = None

        # Tidal event configuration
        self.enable_tidal: bool = bool(gu_cfg.get("ENABLE_TIDAL_EVENT", 0))
        self.tidal_start: int = int(gu_cfg.get("TIDAL_START_TIME", 18))
        self.tidal_end: int = int(gu_cfg.get("TIDAL_END_TIME", 24))
        self.tidal_target: npt.NDArray[np.float64] = np.array(
            gu_cfg.get("TIDAL_TARGET", [850, 850]), dtype=np.float64
        )

        # Tidal physics engine parameters (from config)
        self.tidal_influence_radius: float = float(gu_cfg.get("TIDAL_INFLUENCE_RADIUS", 300.0))
        self.tidal_traction_speed: float = float(gu_cfg.get("TIDAL_TRACTION_SPEED", 50.0))
        self.tidal_scatter_radius: float = float(gu_cfg.get("TIDAL_SCATTER_RADIUS", 20.0))

        # Emergency event configuration
        self.emergency_config: dict[str, Any] = {
            "ENABLE_EMERGENCY": int(gu_cfg.get("ENABLE_EMERGENCY", 0)),
            "EMERGENCY_START_TIME": int(gu_cfg.get("EMERGENCY_START_TIME", 43200)),
            "EMERGENCY_END_TIME": int(gu_cfg.get("EMERGENCY_END_TIME", 50400)),
            "EMERGENCY_POS": gu_cfg.get("EMERGENCY_POS", [300.0, 300.0]),
        }

        # Initialize state variables
        self.current_timestep: int = 0
        self.time_per_step: float = 3600.0  # 1 hour in seconds

        # User positions and movement flags
        self.gu_positions: npt.NDArray[np.float64] | None = None
        self.movement_flags: npt.NDArray[np.bool_] | None = None  # True=mobile, False=stationary

        # HAPS positions
        self.haps_positions: npt.NDArray[np.float64] | None = None

        # Random number generator with configurable seed
        simulation_cfg = config.get("simulation", {})
        seed_value = simulation_cfg.get("RANDOM_SEED", 42)
        # Handle None or "random" string for truly random initialization
        if seed_value is None or seed_value == "random":
            self._rng = np.random.default_rng()
            logger.info("Using random seed (non-reproducible)")
        else:
            self._rng = np.random.default_rng(seed=int(seed_value))
            logger.info(f"Using fixed random seed: {seed_value}")

        logger.info(
            f"World initialized: {self.space_width}x{self.space_height} km, "
            f"{self.gu_count} users ({self.mobile_count} mobile, {self.stationary_count} stationary), "
            f"Tidal={'ON' if self.enable_tidal else 'OFF'}"
        )
        if self.enable_tidal:
            logger.info(
                f"  Tidal event: {self.tidal_start}:00-{self.tidal_end}:00, "
                f"target=({self.tidal_target[0]}, {self.tidal_target[1]}), "
                f"influence_radius={self.tidal_influence_radius}km, "
                f"traction_speed={self.tidal_traction_speed}km/h, "
                f"scatter_radius={self.tidal_scatter_radius}km"
            )

    def initialize(self) -> None:

        # Generate cluster centers uniformly within spatial bounds
        cluster_x = self._rng.uniform(
            low=self.X_MIN,
            high=self.X_MAX,
            size=self.num_clusters
        )
        cluster_y = self._rng.uniform(
            low=self.Y_MIN,
            high=self.Y_MAX,
            size=self.num_clusters
        )
        self.cluster_centers = np.column_stack([cluster_x, cluster_y]).astype(np.float64)

        # Assign each user to a random cluster using vectorized operation
        user_cluster_indices = self._rng.integers(
            low=0,
            high=self.num_clusters,
            size=self.gu_count
        )

        # Get center coordinates for each user via advanced indexing
        user_centers = self.cluster_centers[user_cluster_indices]

        # Vectorized 2D Gaussian sampling: N(center, CLUSTER_STD^2)
        user_x = self._rng.standard_normal(size=self.gu_count) * self.cluster_std + user_centers[:, 0]
        user_y = self._rng.standard_normal(size=self.gu_count) * self.cluster_std + user_centers[:, 1]

        # Clip coordinates to physical boundaries
        user_x = np.clip(user_x, self.X_MIN, self.X_MAX)
        user_y = np.clip(user_y, self.Y_MIN, self.Y_MAX)

        self.gu_positions = np.column_stack([user_x, user_y]).astype(np.float64)

        # Assign movement flags (50% mobile, 50% stationary)
        self.movement_flags = np.zeros(self.gu_count, dtype=np.bool_)
        mobile_indices = self._rng.choice(
            self.gu_count,
            size=self.mobile_count,
            replace=False
        )
        self.movement_flags[mobile_indices] = True

        # HAPS positions placeholder
        self.haps_positions = np.zeros((0, 2), dtype=np.float64)

        self.current_timestep = 0

        # Calculate statistics for logging
        actual_mobile = np.sum(self.movement_flags)
        x_mean = np.mean(self.gu_positions[:, 0])
        y_mean = np.mean(self.gu_positions[:, 1])
        x_std = np.std(self.gu_positions[:, 0])
        y_std = np.std(self.gu_positions[:, 1])

        logger.info(
            f"World generated: {self.gu_count} users at timestep {self.current_timestep}, "
            f"mobile={actual_mobile}, stationary={self.gu_count - actual_mobile}, "
            f"center=({x_mean:.1f}, {y_mean:.1f}), spread=({x_std:.1f}, {y_std:.1f})"
        )
        logger.info(
            f"  Cluster distribution: {self.num_clusters} clusters, "
            f"uniform_std={self.cluster_std:.0f}km"
        )

    def step(self, current_time: int | None = None, time_seconds: float | None = None) -> None:
        
        if self.gu_positions is None or self.movement_flags is None:
            raise RuntimeError("World not initialized. Call initialize() first.")

        # Use provided time or default to 1 hour
        dt_seconds = time_seconds if time_seconds is not None else self.time_per_step
        dt_hours = dt_seconds / 3600.0

        # Get current time for tidal check
        t = current_time if current_time is not None else self.current_timestep

        # Check if tidal event is active
        is_tidal = (
            self.enable_tidal and
            self.tidal_start <= t <= self.tidal_end
        )

        # Identify mobile users
        mobile_mask = self.movement_flags
        mobile_indices = np.where(mobile_mask)[0]

        tidal_affected_count = 0

        if len(mobile_indices) > 0:
            # Normal random walk for all mobile users
            angles = self._rng.uniform(0, 2 * np.pi, size=len(mobile_indices))
            speeds = self._rng.normal(self.SPEED_MEAN, self.SPEED_STD, size=len(mobile_indices))
            speeds = np.clip(speeds, self.SPEED_MIN, self.SPEED_MAX)

            dx = speeds * np.cos(angles) * dt_hours
            dy = speeds * np.sin(angles) * dt_hours

            # Tidal Physics Engine: Range-based traction with scatter
            if is_tidal:
                # Get current positions of all mobile users
                mobile_positions = self.gu_positions[mobile_indices]

                # Calculate Euclidean distance from each mobile user to tidal target
                direction_to_target = self.tidal_target - mobile_positions
                distances_to_target = np.linalg.norm(direction_to_target, axis=1)

                # Filter: Only users within TIDAL_INFLUENCE_RADIUS are affected
                within_radius_mask = distances_to_target <= self.tidal_influence_radius
                affected_local_indices = np.where(within_radius_mask)[0]

                if len(affected_local_indices) > 0:
                    # Get affected user positions
                    affected_positions = mobile_positions[affected_local_indices]
                    n_affected = len(affected_local_indices)

                    # Generate scatter destinations for each affected user
                    # Polar coordinate generation: theta in [0, 2*pi), r in [0, scatter_radius]
                    theta = self._rng.uniform(0, 2 * np.pi, size=n_affected)
                    r = self._rng.uniform(0, self.tidal_scatter_radius, size=n_affected)

                    # Convert polar to Cartesian offset
                    offset_x = r * np.cos(theta)
                    offset_y = r * np.sin(theta)

                    # Calculate absolute destination coordinates for each user
                    dest_x = self.tidal_target[0] + offset_x
                    dest_y = self.tidal_target[1] + offset_y

                    # Calculate direction vector from current position to destination
                    direction_to_dest = np.column_stack([dest_x - affected_positions[:, 0],
                                                          dest_y - affected_positions[:, 1]])
                    dist_to_dest = np.linalg.norm(direction_to_dest, axis=1, keepdims=True)

                    # Normalize direction vectors (add epsilon to avoid division by zero)
                    unit_direction = direction_to_dest / (dist_to_dest + 1e-6)

                    # Calculate step displacement based on configured traction speed
                    step_displacement = self.tidal_traction_speed * dt_hours

                    # If distance < step_displacement, snap to destination to avoid oscillation
                    actual_displacement = np.minimum(dist_to_dest.flatten(), step_displacement)

                    # Compute final displacement vectors
                    displacement_x = unit_direction[:, 0] * actual_displacement
                    displacement_y = unit_direction[:, 1] * actual_displacement

                    # Update displacement arrays for affected users
                    global_affected_indices = mobile_indices[affected_local_indices]

                    # Override normal random walk for affected users
                    dx[affected_local_indices] = displacement_x
                    dy[affected_local_indices] = displacement_y

                    tidal_affected_count = n_affected
                    logger.debug(
                        f"Tidal physics: {tidal_affected_count} users within {self.tidal_influence_radius}km "
                        f"moving toward scattered destinations around {self.tidal_target}"
                    )

            # Update positions for all mobile users
            self.gu_positions[mobile_indices, 0] += dx
            self.gu_positions[mobile_indices, 1] += dy

        # Apply hard boundary constraints to ALL users
        self.gu_positions[:, 0] = np.clip(self.gu_positions[:, 0], 0, self.space_width)
        self.gu_positions[:, 1] = np.clip(self.gu_positions[:, 1], 0, self.space_height)

        # Advance timestep
        self.current_timestep += 1

        # Log statistics
        logger.debug(
            f"Step {self.current_timestep} (t={t}): {len(mobile_indices)} mobile users moved, "
            f"tidal_active={is_tidal}, tidal_affected={tidal_affected_count}"
        )

    def get_state(self) -> WorldState:

        if self.gu_positions is None:
            raise RuntimeError("World not initialized. Call initialize() first.")

        # Check current tidal status
        t = self.current_timestep
        is_tidal = (
            self.enable_tidal and
            self.tidal_start <= t <= self.tidal_end
        )

        # Count affected users if tidal is active (range-based)
        tidal_affected = 0
        if is_tidal and self.gu_positions is not None and self.movement_flags is not None:
            # Get mobile user positions
            mobile_mask = self.movement_flags
            mobile_positions = self.gu_positions[mobile_mask]

            if len(mobile_positions) > 0:
                # Calculate distances to tidal target using vectorized operation
                direction_to_target = self.tidal_target - mobile_positions
                distances = np.linalg.norm(direction_to_target, axis=1)

                # Count users within influence radius
                tidal_affected = int(np.sum(distances <= self.tidal_influence_radius))

        return WorldState(
            timestep=self.current_timestep,
            gu_positions=self.gu_positions.copy(),
            haps_positions=self.haps_positions.copy() if self.haps_positions is not None else np.zeros((0, 2)),
            boundaries=(self.space_width, self.space_height),
            movement_flags=self.movement_flags.copy().astype(np.int32),
            is_tidal_active=is_tidal,
            tidal_affected_count=tidal_affected,
            emergency_config=self.emergency_config
        )

    def get_coverage_stats(
        self,
        haps_positions: npt.NDArray[np.float64] | None = None,
        coverage_radius: float = 210.0
    ) -> dict[str, float]:
        """Calculate coverage statistics."""
        if self.gu_positions is None:
            raise RuntimeError("World not initialized")

        haps = haps_positions if haps_positions is not None else self.haps_positions

        if haps is None or len(haps) == 0:
            return {
                "coverage_ratio": 0.0,
                "covered_users": 0,
                "total_users": self.gu_count,
                "covered_mobile": 0,
                "covered_stationary": 0,
            }

        distances = np.min(
            np.linalg.norm(
                self.gu_positions[:, np.newaxis, :] - haps[np.newaxis, :, :],
                axis=2
            ),
            axis=1
        )

        covered_mask = distances <= coverage_radius
        covered = np.sum(covered_mask)

        mobile_mask = self.movement_flags
        covered_mobile = np.sum(covered_mask & mobile_mask)
        covered_stationary = np.sum(covered_mask & ~mobile_mask)

        return {
            "coverage_ratio": float(covered / self.gu_count),
            "covered_users": int(covered),
            "total_users": self.gu_count,
            "covered_mobile": int(covered_mobile),
            "covered_stationary": int(covered_stationary),
        }

    def register_haps_positions(self, positions: npt.NDArray[np.float64]) -> None:
        """Register HAPS agent positions."""
        self.haps_positions = positions.astype(np.float64).copy()
        logger.debug(f"Registered {len(positions)} HAPS positions")

    def update_haps_positions(self, positions: npt.NDArray[np.float64]) -> None:
        """Update HAPS agent positions with boundary checking."""
        positions = np.asarray(positions, dtype=np.float64)
        positions[:, 0] = np.clip(positions[:, 0], 0, self.space_width)
        positions[:, 1] = np.clip(positions[:, 1], 0, self.space_height)
        self.haps_positions = positions.copy()
        logger.debug(f"Updated {len(positions)} HAPS positions")

    def get_user_summary(self) -> dict[str, Any]:
        """Get summary statistics of current user distribution."""
        if self.gu_positions is None:
            return {"error": "World not initialized"}

        mobile_mask = self.movement_flags

        return {
            "total_users": self.gu_count,
            "mobile_users": int(np.sum(mobile_mask)),
            "stationary_users": int(np.sum(~mobile_mask)),
            "mobile_ratio_config": self.mobile_ratio,
            "tidal_enabled": self.enable_tidal,
            "x_range": {
                "min": float(np.min(self.gu_positions[:, 0])),
                "max": float(np.max(self.gu_positions[:, 0])),
                "mean": float(np.mean(self.gu_positions[:, 0]))
            },
            "y_range": {
                "min": float(np.min(self.gu_positions[:, 1])),
                "max": float(np.max(self.gu_positions[:, 1])),
                "mean": float(np.mean(self.gu_positions[:, 1]))
            },
        }
