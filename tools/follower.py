"""SkyClaw Heartbeat Follower - local fine-tuning triggered by coverage drop."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class HeartbeatFollower:
    """Heartbeat follower: performs local fine-tuning when coverage drops below threshold.

    Moves toward the densest remaining user cluster within the current coverage disk.
    """

    def __init__(
        self,
        coverage_drop_threshold: float = 0.05,
        fine_tune_step: float = 5.0,
        coverage_radius: float = 210.0
    ) -> None:
        """Initialize heartbeat follower.

        Args:
            coverage_drop_threshold: Coverage drop trigger threshold (default 0.05 = 5%).
            fine_tune_step: Fine-tuning step size in km (default 5 km).
            coverage_radius: Coverage radius in km.
        """
        self.coverage_drop_threshold = coverage_drop_threshold
        self.fine_tune_step = fine_tune_step
        self.coverage_radius = coverage_radius

        logger.debug(
            f"HeartbeatFollower initialized: threshold={coverage_drop_threshold:.1%}, "
            f"step={fine_tune_step}km"
        )

    def fine_tune(
        self,
        my_pos: npt.NDArray[np.float64],
        assigned_users: npt.NDArray[np.float64],
        current_coverage: float,
        prev_coverage: float
    ) -> dict[str, Any]:
        """Perform heartbeat-based local fine-tuning.

        Decision: if coverage drop > threshold, move toward centroid of covered users.
        """
        my_pos = np.asarray(my_pos, dtype=np.float64)
        coverage_drop = prev_coverage - current_coverage

        if coverage_drop <= self.coverage_drop_threshold:
            return {
                "new_position": my_pos.copy(),
                "move_vector": np.array([0.0, 0.0]),
                "move_distance": 0.0,
                "triggered": False,
                "coverage_drop": coverage_drop,
                "strategy": "no_action"
            }

        logger.info(
            f"Heartbeat triggered! Coverage dropped by {coverage_drop:.1%} "
            f"({prev_coverage:.1%} -> {current_coverage:.1%})"
        )

        if len(assigned_users) == 0:
            return {
                "new_position": my_pos.copy(),
                "move_vector": np.array([0.0, 0.0]),
                "move_distance": 0.0,
                "triggered": True,
                "coverage_drop": coverage_drop,
                "strategy": "no_users_to_follow"
            }

        distances = np.linalg.norm(assigned_users - my_pos, axis=1)
        covered_mask = distances <= self.coverage_radius
        covered_users = assigned_users[covered_mask]

        if len(covered_users) == 0:
            nearest_user = assigned_users[np.argmin(distances)]
            direction = nearest_user - my_pos
            dist_to_nearest = np.linalg.norm(direction)

            if dist_to_nearest > 1e-6:
                move_vector = direction / dist_to_nearest * self.fine_tune_step
                new_position = my_pos + move_vector
            else:
                new_position = my_pos.copy()
                move_vector = np.array([0.0, 0.0])

            return {
                "new_position": new_position,
                "move_vector": move_vector,
                "move_distance": np.linalg.norm(move_vector),
                "triggered": True,
                "coverage_drop": coverage_drop,
                "strategy": "move_to_nearest_user"
            }

        centroid = np.mean(covered_users, axis=0)
        direction = centroid - my_pos
        distance_to_centroid = np.linalg.norm(direction)

        if distance_to_centroid < 1e-6:
            return {
                "new_position": my_pos.copy(),
                "move_vector": np.array([0.0, 0.0]),
                "move_distance": 0.0,
                "triggered": True,
                "coverage_drop": coverage_drop,
                "strategy": "already_at_centroid"
            }

        unit_direction = direction / distance_to_centroid
        move_vector = unit_direction * self.fine_tune_step
        new_position = my_pos + move_vector

        logger.info(
            f"Fine-tuning toward centroid: step={self.fine_tune_step}km, "
            f"covered_users={len(covered_users)}"
        )

        return {
            "new_position": new_position,
            "move_vector": move_vector,
            "move_distance": self.fine_tune_step,
            "triggered": True,
            "coverage_drop": coverage_drop,
            "strategy": "move_toward_covered_centroid"
        }

    def calculate_coverage_ratio(
        self,
        my_pos: npt.NDArray[np.float64],
        assigned_users: npt.NDArray[np.float64],
        total_users: int
    ) -> float:
        """Calculate current coverage ratio (0-1)."""
        if total_users == 0:
            return 0.0

        my_pos = np.asarray(my_pos, dtype=np.float64)
        distances = np.linalg.norm(assigned_users - my_pos, axis=1)
        covered_count = np.sum(distances <= self.coverage_radius)

        return covered_count / total_users
