"""SkyClaw APF Optimizer - macro movement via Artificial Potential Field method."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class APFOptimizer:
    """APF optimizer: computes net force from attraction (users) and repulsion (other HAPS) to guide movement.

    F_total = F_att + F_rep
    """

    def __init__(
        self,
        coverage_radius: float = 210.0,
        max_step: float = 40.0,
        safety_margin: float = 1.5,
        space_bounds: tuple[float, float, float, float] = (0, 1000, 0, 1000)
    ) -> None:
        """Initialize APF optimizer.

        Args:
            coverage_radius: Coverage radius in km.
            max_step: Maximum single-step movement in km.
            safety_margin: Safety distance multiplier (default 1.5x coverage radius).
            space_bounds: Space boundaries (x_min, x_max, y_min, y_max).
        """
        self.coverage_radius = coverage_radius
        self.max_step = max_step
        self.safety_margin = safety_margin
        self.space_bounds = space_bounds

        self.x_min, self.x_max, self.y_min, self.y_max = space_bounds
        self.safe_distance = coverage_radius * safety_margin

        logger.debug(
            f"APFOptimizer initialized: coverage={coverage_radius}km, "
            f"max_step={max_step}km, safe_distance={self.safe_distance:.1f}km"
        )

    def calculate_move(
        self,
        assigned_users: npt.NDArray[np.float64],
        my_pos: npt.NDArray[np.float64],
        other_haps_positions: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]],
        coverage_radius: float | None = None
    ) -> dict[str, Any]:
        """Calculate next move using APF.

        Steps: attractive force -> repulsive force -> net force -> clamp step -> clip bounds.
        """
        my_pos = np.asarray(my_pos, dtype=np.float64)
        safe_dist = coverage_radius * self.safety_margin if coverage_radius else self.safe_distance

        # Step 1: Compute attractive force (F_att)
        f_att = self._calculate_attractive_force(assigned_users, my_pos)

        # Step 2: Compute repulsive force (F_rep)
        if other_haps_positions is not None and len(other_haps_positions) > 0:
            other_pos = np.asarray(other_haps_positions, dtype=np.float64)
        else:
            other_pos = np.zeros((0, 2))
        f_rep = self._calculate_repulsive_force(my_pos, other_pos, safe_dist)

        # Step 3: Compute net force
        f_total = f_att + f_rep

        # Check convergence (net force is small)
        force_magnitude = np.linalg.norm(f_total)
        convergence_threshold = 5.0  # km
        converged = force_magnitude < convergence_threshold

        if converged:
            logger.debug(f"APF converged at ({my_pos[0]:.1f}, {my_pos[1]:.1f})")
            return {
                "new_position": my_pos.copy(),
                "move_vector": np.array([0.0, 0.0]),
                "move_distance": 0.0,
                "f_attractive": f_att,
                "f_repulsive": f_rep,
                "converged": True
            }

        # Step 4: Clamp step size
        if force_magnitude > 1e-6:
            direction = f_total / force_magnitude
            step_size = min(force_magnitude * 0.5, self.max_step)
            move_vector = direction * step_size
        else:
            move_vector = np.array([0.0, 0.0])

        # Step 5: Compute new position and clip to bounds
        new_position = my_pos + move_vector
        new_position[0] = np.clip(new_position[0], self.x_min, self.x_max)
        new_position[1] = np.clip(new_position[1], self.y_min, self.y_max)

        actual_move = new_position - my_pos
        actual_distance = np.linalg.norm(actual_move)

        logger.debug(
            f"APF move: F_att={np.linalg.norm(f_att):.1f}, "
            f"F_rep={np.linalg.norm(f_rep):.1f}, "
            f"step={actual_distance:.1f}km"
        )

        return {
            "new_position": new_position,
            "move_vector": actual_move,
            "move_distance": float(actual_distance),
            "f_attractive": f_att,
            "f_repulsive": f_rep,
            "converged": False
        }

    def _calculate_attractive_force(
        self,
        assigned_users: npt.NDArray[np.float64],
        my_pos: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute attractive force toward user centroid. Magnitude proportional to distance."""
        if len(assigned_users) == 0:
            return np.array([0.0, 0.0])

        centroid = np.mean(assigned_users, axis=0)
        direction = centroid - my_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return np.array([0.0, 0.0])

        force_magnitude = distance * 0.5

        return direction / distance * force_magnitude

    def _calculate_repulsive_force(
        self,
        my_pos: npt.NDArray[np.float64],
        other_haps_positions: npt.NDArray[np.float64],
        safe_distance: float
    ) -> npt.NDArray[np.float64]:
        """Compute repulsive force from other HAPS. Inverse-distance within safe_distance."""
        if len(other_haps_positions) == 0:
            return np.array([0.0, 0.0])

        total_repulsive = np.array([0.0, 0.0])

        diff = my_pos - other_haps_positions
        distances = np.linalg.norm(diff, axis=1)

        mask = distances < safe_distance

        if not np.any(mask):
            return total_repulsive

        close_haps = other_haps_positions[mask]
        close_distances = distances[mask]
        close_diff = diff[mask]

        for dist, direction_vec in zip(close_distances, close_diff):
            if dist < 1e-6:
                continue

            force_mag = safe_distance * (1.0 / dist - 1.0 / safe_distance)
            force_mag = max(0, force_mag)

            unit_direction = direction_vec / dist
            total_repulsive += unit_direction * force_mag

        return total_repulsive
