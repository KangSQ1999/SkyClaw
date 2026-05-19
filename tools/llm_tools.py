"""SkyClaw LLM Tools - standardized tool functions for the LLM controller."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from tools.partitioner import SpacePartitioner
from tools.optimizer import APFOptimizer

logger = logging.getLogger(__name__)


class LLMTools:
    """Collection of tools wrapping low-level algorithms behind LLM-callable interfaces.

    All tools return human-readable string descriptions.
    """

    def __init__(
        self,
        agent,
        world_state: Any,
        other_haps_positions: npt.NDArray[np.float64]
    ) -> None:
        """Initialize toolset.

        Args:
            agent: Current Agent instance.
            world_state: Current world state.
            other_haps_positions: Positions of other HAPS.
        """
        self.agent = agent
        self.world_state = world_state
        self.other_haps = other_haps_positions

        self.partitioner = SpacePartitioner(map_size=1000.0)
        self.optimizer = APFOptimizer(
            coverage_radius=210.0,
            max_step=40.0,
            safety_margin=1.5
        )

        self._last_partition_result: dict[str, Any] | None = None

        logger.debug(f"LLMTools initialized for {agent.agent_id}")

    def emergency_move_tool(
        self,
        target_x: float,
        target_y: float,
        coverage_radius: float = 210.0
    ) -> str:
        """Emergency support move tool with distance arbitration and edge braking.

        Layer 1: Only the nearest airship can support.
        Layer 2: Stop approaching at 85% coverage radius; use disk edge for coverage.
        """
        target = np.array([target_x, target_y], dtype=np.float64)
        my_pos = self.agent.position

        # Layer 1: Distance arbitration
        my_distance = np.linalg.norm(target - my_pos)

        all_haps_distances = [my_distance]
        for other_pos in self.other_haps:
            dist = np.linalg.norm(target - other_pos)
            all_haps_distances.append(dist)

        min_distance = min(all_haps_distances)
        if my_distance > min_distance + 1e-6:
            logger.info(
                f"[{self.agent.agent_id}] Emergency move REJECTED: "
                f"distance={my_distance:.1f}km, min={min_distance:.1f}km"
            )
            return (
                "Call failed: system determined another airship is closer to the target. "
                "Abort support and immediately call partition_space_tool and optimize_move_tool for regular patrol."
            )

        # Layer 2: Edge coverage braking
        coverage_threshold = coverage_radius * 0.85

        if my_distance <= coverage_threshold:
            self.agent.current_destination = my_pos.copy()

            logger.info(
                f"[{self.agent.agent_id}] Emergency BRAKE: target within coverage "
                f"(distance={my_distance:.1f}km <= {coverage_threshold:.1f}km)"
            )
            return (
                f"Call succeeded: hot spot already within safe coverage range (distance {my_distance:.1f} km). "
                "Airship has braked and is now hovering on station; no further approach needed."
            )
        else:
            direction = target - my_pos
            direction_unit = direction / my_distance

            optimal_distance = coverage_threshold * 0.95
            new_destination = target - direction_unit * optimal_distance

            new_destination[0] = np.clip(new_destination[0], 0.0, 1000.0)
            new_destination[1] = np.clip(new_destination[1], 0.0, 1000.0)

            self.agent.current_destination = new_destination.copy()

            logger.info(
                f"[{self.agent.agent_id}] Emergency move APPROVED: "
                f"approaching target, distance={my_distance:.1f}km, "
                f"optimal_distance={optimal_distance:.1f}km"
            )
            return (
                f"Call succeeded: distance to target is {my_distance:.1f} km. Approaching at full speed."
            )

    def partition_space_tool(self) -> str:
        """Space partition tool: assigns ground users to nearest HAPS via Voronoi approximation.

        Underlying state is read automatically from world state.
        """
        all_users = self.world_state.gu_positions

        if len(all_users) == 0:
            return "Partition failed: no ground users in current field of view."

        partition_info = self.partitioner.partition_with_info(
            all_users=all_users,
            my_pos=self.agent.position,
            other_haps_positions=self.other_haps
        )

        self._last_partition_result = partition_info
        self.agent._last_assigned_users = partition_info["assigned_users"]

        user_count = partition_info["user_count"]
        avg_dist = partition_info["avg_distance"]
        coverage_ratio = partition_info["coverage_ratio"]
        centroid = partition_info["territory_centroid"]

        logger.info(
            f"[{self.agent.agent_id}] Space partitioned: "
            f"{user_count} users assigned"
        )

        return (
            f"Partition successful! Locked {user_count} exclusive users "
            f"({coverage_ratio:.1%} of total). "
            f"Your territory centroid is at [{centroid[0]:.1f}, {centroid[1]:.1f}], "
            f"average distance {avg_dist:.1f} km."
        )

    def optimize_move_tool(self) -> str:
        """APF-based move optimization tool.

        Computes attractive force (user centroid), repulsive force (other HAPS collision avoidance),
        and returns the optimal move direction.

        Must call partition_space_tool first.
        """
        if self._last_partition_result is None:
            assigned_users = getattr(self.agent, '_last_assigned_users', None)
            if assigned_users is None:
                return (
                    "Optimization failed: no partition result found. "
                    "Please call partition_space_tool first to obtain exclusive territory."
                )
        else:
            assigned_users = self._last_partition_result["assigned_users"]

        if len(assigned_users) == 0:
            self.agent.current_destination = self.agent.position.copy()
            return "Optimization complete: no users in current territory, maintaining hover."

        move_result = self.optimizer.calculate_move(
            assigned_users=assigned_users,
            my_pos=self.agent.position,
            other_haps_positions=self.other_haps,
            coverage_radius=210.0
        )

        new_position = move_result["new_position"]
        self.agent.current_destination = new_position.copy()

        move_distance = move_result["move_distance"]
        f_att = move_result.get("f_attractive", np.array([0, 0]))
        f_rep = move_result.get("f_repulsive", np.array([0, 0]))

        rep_magnitude = np.linalg.norm(f_rep)
        if rep_magnitude > 1.0:
            avoidance_note = " (auto-avoiding nearby allied airships)"
        else:
            avoidance_note = ""

        logger.info(
            f"[{self.agent.agent_id}] APF optimization: "
            f"move {move_distance:.1f}km {avoidance_note}"
        )

        return (
            f"Move optimization complete. Moved {move_distance:.1f} km toward user centroid"
            f"{avoidance_note}. New target coordinates [{new_position[0]:.1f}, {new_position[1]:.1f}]."
        )

    def predictive_move_tool(
        self,
        target_x: float,
        target_y: float,
        reason: str = ""
    ) -> str:
        """Predictive move tool: pre-deploy to predicted hotspot based on historical memory.

        Differs from emergency_move_tool: no distance arbitration, uses edge coverage strategy.
        Call when memory records past tidal events near the current time window.
        """
        target = np.array([target_x, target_y], dtype=np.float64)
        my_pos = self.agent.position
        coverage_radius = 210.0

        my_distance = np.linalg.norm(target - my_pos)
        coverage_threshold = coverage_radius * 0.85

        if my_distance <= coverage_threshold:
            self.agent.current_destination = my_pos.copy()

            logger.info(
                f"[{self.agent.agent_id}] Predictive BRAKE: target within coverage "
                f"(distance={my_distance:.1f}km <= {coverage_threshold:.1f}km)"
            )
            return (
                f"Predictive move complete: target hotspot already within coverage range (distance {my_distance:.1f} km). "
                f"Airship has braked and is now hovering on station at edge-coverage position."
            )
        else:
            direction = target - my_pos
            direction_unit = direction / my_distance

            optimal_distance = coverage_threshold * 0.95
            new_destination = target - direction_unit * optimal_distance

            new_destination[0] = np.clip(new_destination[0], 0.0, 1000.0)
            new_destination[1] = np.clip(new_destination[1], 0.0, 1000.0)

            self.agent.current_destination = new_destination.copy()

            logger.info(
                f"[{self.agent.agent_id}] Predictive move APPROVED: "
                f"approaching hotspot {target}, distance={my_distance:.1f}km, "
                f"optimal_distance={optimal_distance:.1f}km, reason={reason[:50]}"
            )
            return (
                f"Predictive move initiated: {my_distance:.1f} km from target hotspot. "
                f"Heading to edge-coverage position [{new_destination[0]:.1f}, {new_destination[1]:.1f}]. "
                f"Reason: {reason if reason else 'Periodic hotspot prediction based on historical memory'}"
            )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM function calling (OpenAI/Gemini format)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "emergency_move_tool",
                    "description": "Called upon receiving emergency HELP_REQUEST or system alert. Sets emergency support target coordinates. Includes distance arbitration (only the nearest airship may respond) and edge braking (hover once within coverage instead of flying directly overhead).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_x": {
                                "type": "number",
                                "description": "X coordinate of the target requesting support"
                            },
                            "target_y": {
                                "type": "number",
                                "description": "Y coordinate of the target requesting support"
                            }
                        },
                        "required": ["target_x", "target_y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "partition_space_tool",
                    "description": "Space partition tool: assigns ground users to nearest HAPS by distance and returns exclusive territory info.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_move_tool",
                    "description": "Computes optimal move direction via Artificial Potential Field (APF), considering current user coverage, collision avoidance, and warm-up attraction toward historically predicted hotspots.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "predictive_move_tool",
                    "description": "Predictive move tool: pre-deploy to predicted hotspot based on historical memory. Unlike emergency move, this tool does NOT perform distance arbitration and uses edge-coverage strategy (stops at coverage disk edge rather than flying to the centroid). Call when long-term memory records past tidal events near the current time window.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_x": {
                                "type": "number",
                                "description": "X coordinate of the predicted hotspot centroid (e.g. 850.0)"
                            },
                            "target_y": {
                                "type": "number",
                                "description": "Y coordinate of the predicted hotspot centroid (e.g. 850.0)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for the move (e.g. 'Based on yesterday's memory, tidal event expected in northeast at 18:00')"
                            }
                        },
                        "required": ["target_x", "target_y"]
                    }
                }
            }
        ]
