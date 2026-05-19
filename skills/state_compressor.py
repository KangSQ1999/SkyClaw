"""SkyClaw State Compressor - 8-sector occupancy grid perception."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class StateCompressor:
    """Compresses raw perception data into semantic summaries using 8 fixed-direction sectors.

    Replaces clustering with absolute-stable sector identities for cross-timestep tracking.
    """

    SECTOR_DEFS = [
        ("East", -22.5, 22.5),
        ("Northeast", 22.5, 67.5),
        ("North", 67.5, 112.5),
        ("Northwest", 112.5, 157.5),
        ("West", 157.5, 180.0),
        ("West", -180.0, -157.5),
        ("Southwest", -157.5, -112.5),
        ("South", -112.5, -67.5),
        ("Southeast", -67.5, -22.5),
    ]

    def __init__(self, max_radius: float = 250.0) -> None:
        """Initialize compressor with max perception radius (default 250.0 km)."""
        self.max_radius = max_radius
        logger.debug(f"StateCompressor initialized: max_radius={max_radius}")

    def compress(
        self,
        agent_pos: npt.NDArray[np.float64],
        users_positions: npt.NDArray[np.float64],
        movement_flags: npt.NDArray[np.bool_]
    ) -> dict[str, Any]:
        """Compress perception into semantic summary (8-sector version).

        Returns dict with semantic_summary, total_local_users, and sectors.
        """
        if len(users_positions) == 0:
            return {
                "semantic_summary": "No users within perception range.",
                "total_local_users": 0,
                "sectors": []
            }

        # Step 1: spatial truncation - keep only users within max_radius
        distances_to_agent = np.linalg.norm(users_positions - agent_pos, axis=1)
        within_radius_mask = distances_to_agent <= self.max_radius

        local_users = users_positions[within_radius_mask]
        local_movement = movement_flags[within_radius_mask]
        local_distances = distances_to_agent[within_radius_mask]

        total_local = len(local_users)

        if total_local == 0:
            return {
                "semantic_summary": f"No users within perception range (radius {self.max_radius}km).",
                "total_local_users": 0,
                "sectors": []
            }

        # Step 2: compute angle of each user relative to agent
        dx = local_users[:, 0] - agent_pos[0]
        dy = local_users[:, 1] - agent_pos[1]
        angles = np.arctan2(dy, dx) * 180 / np.pi

        # Step 3: 8-sector statistics
        sectors_info: list[dict[str, Any]] = []

        # East, Northeast, North, Northwest, West (crosses 180/-180), Southwest, South, Southeast
        sector_bounds = [
            ("East", -22.5, 22.5),
            ("Northeast", 22.5, 67.5),
            ("North", 67.5, 112.5),
            ("Northwest", 112.5, 157.5),
            ("West", 157.5, 180.0, -180.0, -157.5),
            ("Southwest", -157.5, -112.5),
            ("South", -112.5, -67.5),
            ("Southeast", -67.5, -22.5),
        ]

        for bounds in sector_bounds:
            if len(bounds) == 5:
                name, low1, high1, low2, high2 = bounds
                mask = ((angles >= low1) & (angles < high1)) | ((angles >= low2) & (angles < high2))
            else:
                name, low, high = bounds
                mask = (angles >= low) & (angles < high)

            user_count = int(np.sum(mask))
            if user_count == 0:
                continue

            sector_users = local_users[mask]
            sector_movement = local_movement[mask]
            sector_distances = local_distances[mask]

            centroid = np.mean(sector_users, axis=0)

            mobile_count = int(np.sum(sector_movement))
            mobile_ratio = mobile_count / user_count if user_count > 0 else 0.0
            avg_distance = float(np.mean(sector_distances)) if len(sector_distances) > 0 else 0.0

            sectors_info.append({
                "direction": name,
                "centroid": [float(centroid[0]), float(centroid[1])],
                "user_count": user_count,
                "mobile_count": mobile_count,
                "mobile_ratio": float(mobile_ratio),
                "avg_distance": avg_distance
            })

        # Step 4: sort by user_count descending
        sectors_info.sort(key=lambda s: s["user_count"], reverse=True)

        # Step 5: generate objective semantic summary
        semantic_summary = self._generate_objective_summary(
            sectors_info, total_local, self.max_radius
        )

        result = {
            "semantic_summary": semantic_summary,
            "total_local_users": total_local,
            "sectors": sectors_info
        }

        logger.debug(
            f"Compressed state: {total_local} local users, "
            f"{len(sectors_info)} sectors"
        )

        return result

    def _generate_objective_summary(
        self,
        sectors_info: list[dict[str, Any]],
        total_local: int,
        max_radius: float
    ) -> str:
        """Generate objective radar-style report (facts only, no recommendations)."""
        if not sectors_info:
            return f"No users within perception range (radius {max_radius}km)."

        sector_descriptions: list[str] = []
        for sector in sectors_info:
            direction = sector["direction"]
            distance = sector["avg_distance"]
            user_count = sector["user_count"]
            mobile_ratio = sector["mobile_ratio"] * 100

            centroid = sector["centroid"]
            desc = (
                f"{direction} sector(centroid[{centroid[0]:.1f},{centroid[1]:.1f}], "
                f"~{distance:.1f}km, {user_count} users, mobile ratio {mobile_ratio:.0f}%)"
            )
            sector_descriptions.append(desc)

        summary = (
            f"Found {total_local} users within perception range. "
            f"Main distribution: {'; '.join(sector_descriptions)}."
        )

        return summary
