"""SkyClaw Space Partitioner - dynamic Voronoi-based user assignment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class SpacePartitioner:
    """Space partitioner: assigns ground users to the nearest HAPS based on Euclidean distance.

    Approximates Voronoi diagram partitioning.
    """

    def __init__(self, map_size: float = 1000.0) -> None:
        """Initialize space partitioner.

        Args:
            map_size: Map boundary size, default 1000 km.
        """
        self.map_size = map_size
        logger.debug(f"SpacePartitioner initialized with map_size={map_size}")

    def partition(
        self,
        all_users: npt.NDArray[np.float64],
        my_pos: npt.NDArray[np.float64],
        other_haps_positions: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Execute space partition, return users assigned to current HAPS.

        Steps: build HAPS position matrix -> compute distance matrix -> find nearest HAPS per user -> filter.
        """
        if len(all_users) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        my_pos = np.asarray(my_pos, dtype=np.float64).reshape(1, 2)

        if len(other_haps_positions) == 0:
            all_haps = my_pos
        else:
            other_pos = np.asarray(other_haps_positions, dtype=np.float64)
            if other_pos.ndim == 1:
                other_pos = other_pos.reshape(1, 2)
            all_haps = np.vstack([my_pos, other_pos])

        distances = np.linalg.norm(
            all_users[:, np.newaxis, :] - all_haps[np.newaxis, :, :],
            axis=2
        )

        nearest_haps_idx = np.argmin(distances, axis=1)
        my_users_mask = nearest_haps_idx == 0
        assigned_users = all_users[my_users_mask]

        logger.debug(
            f"[{len(assigned_users)}/{len(all_users)}] users assigned to this HAPS"
        )

        return assigned_users

    def partition_with_info(
        self,
        all_users: npt.NDArray[np.float64],
        my_pos: npt.NDArray[np.float64],
        other_haps_positions: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]
    ) -> dict[str, Any]:
        """Execute space partition and return detailed metadata."""
        assigned_users = self.partition(all_users, my_pos, other_haps_positions)

        user_count = len(assigned_users)
        total_users = len(all_users)
        coverage_ratio = user_count / total_users if total_users > 0 else 0.0

        if user_count > 0:
            my_pos_arr = np.asarray(my_pos, dtype=np.float64)
            distances = np.linalg.norm(assigned_users - my_pos_arr, axis=1)
            avg_distance = float(np.mean(distances))
            territory_centroid = np.mean(assigned_users, axis=0)
        else:
            avg_distance = 0.0
            territory_centroid = np.array(my_pos).copy()

        return {
            "assigned_users": assigned_users,
            "user_count": user_count,
            "coverage_ratio": coverage_ratio,
            "avg_distance": avg_distance,
            "territory_centroid": territory_centroid
        }
