"""HAPS Agent core: observe-compress-memorize-calculate-act-reflect workflow with LLM-as-Controller."""

from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

# Import Skill, Tool, and MemoryManager
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from skills.state_compressor import StateCompressor
from skills.memory_consolidator import MemoryConsolidator
from tools.optimizer import APFOptimizer
from tools.partitioner import SpacePartitioner
from tools.follower import HeartbeatFollower
from tools.llm_tools import LLMTools
from core_agent.workspace.memory_manager import MemoryManager
from core_agent.llm_logger import LLMInteractionLogger

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Encapsulates the complete output of an agent's single-step decision."""

    agent_id: str
    target_position: npt.NDArray[np.float64]
    reasoning: str
    confidence: float
    memory_snapshot: dict[str, Any] = field(default_factory=dict)


class HapsAgent:
    """SkyClaw HAPS autonomous agent with full observe-compress-memorize-calculate-act-reflect workflow."""

    def __init__(
        self,
        agent_id: str,
        initial_position: list[float] | npt.NDArray[np.float64],
        memory_capacity: int = 5,
        max_radius: float = 250.0,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        move_ratio: float = 0.2,
        gateway: Any | None = None,
        llm_client: Any | None = None,
        novelty_threshold: float = 0.5,
        ib_beta: float = 1.0,
        ib_utility_threshold: float = 0.2,
        use_ib_optimizer: bool = True
    ) -> None:
        """Initialize the HAPS Agent with Lagrangian Information Bottleneck support."""
        self.agent_id = agent_id
        self.position = np.array(initial_position, dtype=np.float64)
        self.memory_capacity = memory_capacity

        # Initialize Skill: StateCompressor (8-sector fixed perception)
        self.compressor = StateCompressor(max_radius=max_radius)

        # Phase 3: Initialize MemoryManager (isolated workspace)
        self.memory_manager = MemoryManager(agent_id)

        # Phase 4/6: Initialize MemoryConsolidator
        self.memory_consolidator = MemoryConsolidator(
            llm_client=llm_client,
            use_ib_optimizer=use_ib_optimizer
        )

        # Cognitive Information Bottleneck parameters
        self.novelty_threshold = novelty_threshold
        self.ib_beta = ib_beta
        self.ib_utility_threshold = ib_utility_threshold
        self.use_ib_optimizer = use_ib_optimizer

        # Phase 5: Communication gateway (minimal inbox-based)
        self.gateway = gateway

        # Phase 6 Fix: Use ThreadPoolExecutor for background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_future: Any | None = None

        # Phase 3 Movement Control: advanced movement control tools (cerebellum layer)
        # Dynamic space partitioning: Voronoi approximation, assign exclusive users
        self.partitioner = SpacePartitioner(map_size=1000.0)
        # Artificial Potential Field optimizer: attraction and repulsion guide macro movement
        self.apf_optimizer = APFOptimizer(
            coverage_radius=210.0,
            max_step=40.0,
            safety_margin=1.5
        )
        # Heartbeat follower: local fine-tuning based on coverage change
        self.follower = HeartbeatFollower(
            coverage_drop_threshold=0.05,
            fine_tune_step=5.0,
            coverage_radius=210.0
        )
        # Previous step coverage, used for heartbeat detection
        self.prev_coverage = 0.0

        # Time-axis Refactoring: separate decision target from current position
        # current_destination stores the LLM/Tool decision target (updated hourly)
        # position stores the current actual position (updated every 5 minutes)
        self.current_destination: np.ndarray | None = None

        # LLM Interaction Logger: records all LLM interactions
        self.interaction_logger = LLMInteractionLogger(agent_id)

        # Note: do not set llm_client.logger here, pass logger parameter at call time instead
        # Avoids singleton pattern causing logger to be overwritten

        # In-memory short-term cache (as file system cache layer)
        self.workspace_memory: list[dict[str, Any]] = []

        # Load Soul (system prompt)
        self.soul_prompt = self._load_soul()

        logger.info(
            f"HapsAgent '{agent_id}' initialized at "
            f"({self.position[0]:.1f}, {self.position[1]:.1f}) with isolated workspace"
        )

    def _load_soul(self) -> str:
        """Load identity definition from each HAPS's own soul.md."""
        soul_path = Path(__file__).parent / "workspace" / self.agent_id / "soul.md"
        if soul_path.exists():
            return soul_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"soul.md not found at {soul_path}")
            return f"# Default Soul\nYou are {self.agent_id}."

    def _load_skill(self) -> str:
        """Load SOP manual from skill.md (LLM-as-Controller version)."""
        skill_path = Path(__file__).parent / "workspace" / "skill.md"
        if skill_path.exists():
            return skill_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"skill.md not found at {skill_path}")
            return "# Default Skill\nFollow the workflow and use tools."

    def _extract_memory_records(self, memory_text: str) -> str:
        """Extract records from memory.md, stripping header and truncating to avoid token overflow."""
        if not memory_text or not memory_text.strip():
            return ""

        # Find content after "## Memory Records"
        marker = "## Memory Records"
        if marker in memory_text:
            records = memory_text.split(marker, 1)[1].strip()
        else:
            records = memory_text.strip()

        # Skip empty content
        if not records.strip():
            return ""

        # Limit length: keep at most the last 2000 characters (~3-4 AI Reflections)
        max_len = 2000
        if len(records) > max_len:
            records = "...(earlier memories truncated)\n" + records[-max_len:]

        return records

    def observe_and_act(
        self,
        world_state: Any,
        current_time: int | None = None
    ) -> tuple[npt.NDArray[np.float64], Any]:
        """Core workflow: LLM-as-Controller with Tool Calling via ReAct loop (max 3 rounds)."""
        logger.info(f"[{self.agent_id}] LLM-as-Controller decision at T={current_time}")

        # ========== Step 1: Preparation ==========
        # Get other HAPS positions
        all_haps = world_state.haps_positions
        other_haps = []
        for i, pos in enumerate(all_haps):
            if f"haps_{i}" != self.agent_id:
                other_haps.append(pos)
        other_haps_arr = np.array(other_haps) if other_haps else np.zeros((0, 2))

        # Initialize LLM toolset
        llm_tools = LLMTools(self, world_state, other_haps_arr)

        # Load SOP manual
        skill_prompt = self._load_skill()

        # Load long-term memory and inject into system prompt
        memory_text = self.memory_manager.read_memory()
        memory_records = self._extract_memory_records(memory_text)
        if memory_records:
            skill_prompt += (
                "\n\n=== Your Long-Term Memory (patterns learned from past runs) ===\n"
                f"{memory_records}\n"
                "\nNote: The above is your historical experience records. "
                "Please decide whether to call predictive_move_tool for preemptive deployment based on the SOP rules."
            )

        # ========== Step 2: Receive messages and build status brief ==========
        received_messages = []
        if self.gateway is not None:
            received_messages = self.gateway.receive(self.agent_id)

        # Build minimal status brief (highly semantic, token-efficient)
        help_request_msg = None
        mailbox_status = "no messages"
        for msg in received_messages:
            if msg.get("type") == "HELP_REQUEST":
                help_request_msg = msg
                mailbox_status = f"received HELP_REQUEST from {msg.get('sender')} at {msg.get('target_pos')}"
                break

        # Calculate coverage trend
        all_users = world_state.gu_positions
        current_coverage = self.follower.calculate_coverage_ratio(
            my_pos=self.position,
            assigned_users=all_users,  # simplified: use all users
            total_users=len(all_users)
        )
        coverage_trend = "stable"
        if self.prev_coverage > 0:
            diff = current_coverage - self.prev_coverage
            if diff > 0.02:
                coverage_trend = f"up {diff:.1%}"
            elif diff < -0.02:
                coverage_trend = f"down {abs(diff):.1%}"

        # ========== Step 2.5: State compression (8-sector fixed perception) ==========
        # Get movement flags (ensure bool type)
        if world_state.movement_flags is not None and len(world_state.movement_flags) == len(all_users):
            movement_flags: npt.NDArray[np.bool_] = world_state.movement_flags.astype(np.bool_)
        else:
            # Default: all non-moving users
            movement_flags: npt.NDArray[np.bool_] = np.zeros(len(all_users), dtype=np.bool_)

        # Call StateCompressor to compress perception state
        compressed_observation = self.compressor.compress(
            agent_pos=self.position,
            users_positions=all_users,
            movement_flags=movement_flags
        )

        # Extract core info for status brief
        semantic_summary = compressed_observation.get("semantic_summary", "Perception unavailable")
        sector_count = len(compressed_observation.get("sectors", []))
        total_local = compressed_observation.get("total_local_users", 0)

        # ========== Emergency alert injection ==========
        # Check if currently within an emergency event window
        emergency_alert = ""
        if world_state.emergency_config and world_state.emergency_config.get("ENABLE_EMERGENCY", 0):
            # current_time is in hours (0-23), convert to seconds for comparison
            current_time_sec = (current_time if current_time is not None else 0) * 3600
            start_time = world_state.emergency_config.get("EMERGENCY_START_TIME", 0)
            end_time = world_state.emergency_config.get("EMERGENCY_END_TIME", 0)
            emergency_pos = world_state.emergency_config.get("EMERGENCY_POS", [300.0, 300.0])

            if start_time <= current_time_sec < end_time:
                emergency_alert = f"\n[SYSTEM CRITICAL ALERT] Emergency event at [{emergency_pos[0]}, {emergency_pos[1]}], urgent coverage support needed!"

        status_brief = f"""
Current Time: T={current_time} ({current_time:02d}:00)
Current Position: [{self.position[0]:.1f}, {self.position[1]:.1f}]
Mailbox Status: {mailbox_status}
Last Strategy: {getattr(self, '_last_strategy', 'initialize')}
Coverage Trend: {coverage_trend} (current {current_coverage:.1%})
Perception Summary: {semantic_summary}{emergency_alert}
""".strip()

        logger.info(f"[{self.agent_id}] Status brief:\n{status_brief}")

        # ========== Step 3: ReAct loop (LLM Tool Calling - OpenAI standard protocol) ==========
        # Initialize conversation history
        conversation_history: list[dict[str, Any]] = [{"role": "user", "content": status_brief}]

        # Get tool definitions
        tool_schemas = llm_tools.get_tool_schemas()

        # ReAct loop (max 3 rounds)
        max_iterations = 3
        final_destination = self.position.copy()
        strategy_name = "llm_fallback"

        for iteration in range(max_iterations):
            logger.debug(f"[{self.agent_id}] ReAct iteration {iteration + 1}/{max_iterations}")

            try:
                # Check if LLM client is available
                if self.memory_consolidator.llm_client is None:
                    logger.warning(f"[{self.agent_id}] No LLM client, falling back to hardcoded logic")
                    # Fallback: use original hardcoded logic
                    final_destination, strategy_name = self._fallback_decision(
                        help_request_msg, other_haps_arr, all_users
                    )
                    break

                # Call LLM (with tools)
                llm_client = self.memory_consolidator.llm_client
                from core_agent.llm_client import LLMResponse
                result: LLMResponse = llm_client.call_with_tools(
                    system_prompt=skill_prompt,
                    messages=conversation_history,
                    tools=tool_schemas,
                    temperature=0.3,
                    interaction_logger=self.interaction_logger  # pass logger parameter
                )

                # Check if it's a tool call request
                if result.is_tool_call:
                    tool_name = result.tool_name
                    tool_args = result.tool_args
                    tool_call_id = result.tool_call_id

                    logger.info(f"[{self.agent_id}] LLM requested: {tool_name}({tool_args})")

                    # Add assistant message (including tool_calls) to conversation history
                    if result.assistant_message:
                        conversation_history.append(result.assistant_message)

                    # Execute tool
                    tool_result = self._execute_tool(llm_tools, tool_name, tool_args or {})

                    logger.info(f"[{self.agent_id}] Tool result: {tool_result[:80]}...")

                    # Add tool result message (required! OpenAI protocol)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result
                    }
                    conversation_history.append(tool_message)

                    # Record strategy
                    if "emergency" in tool_name:
                        strategy_name = "llm_emergency"
                    elif "partition" in tool_name:
                        strategy_name = "llm_partition"
                    elif "optimize" in tool_name:
                        strategy_name = "llm_optimize"

                    # If a movement tool was called, can end the loop
                    if "move" in tool_name:
                        final_destination = self.current_destination.copy()
                        break

                    continue  # Continue next round of conversation

                else:
                    # LLM gave a final answer
                    content = result.content
                    logger.info(f"[{self.agent_id}] LLM final response: {content[:100]}...")

                    # Add assistant message to conversation history
                    conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })

                    final_destination = self.current_destination.copy() if self.current_destination is not None else self.position.copy()
                    break

            except Exception as e:
                logger.error(f"[{self.agent_id}] ReAct error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to hardcoded
                final_destination, strategy_name = self._fallback_decision(
                    help_request_msg, other_haps_arr, all_users
                )
                break

        # If loop ended without a target, use current position
        if self.current_destination is None:
            self.current_destination = final_destination.copy()

        # Update strategy record
        self._last_strategy = strategy_name

        # ========== Step 4: Memory persistence ==========
        move_distance = np.linalg.norm(final_destination - self.position)

        # Convert numpy arrays in observation to JSON-serializable lists
        def convert_to_serializable(obj: Any) -> Any:
            """Recursively convert numpy arrays to Python lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        serializable_observation = convert_to_serializable(compressed_observation)

        history_record = {
            "time": current_time if current_time is not None else -1,
            "agent_position": self.position.tolist(),
            "destination": final_destination.tolist(),
            "llm_strategy": strategy_name,
            "coverage": {
                "current": current_coverage,
                "previous": self.prev_coverage
            },
            "observation": serializable_observation  # JSON-serializable sector perception data
        }

        if received_messages:
            history_record["communication"] = {
                "received": len(received_messages),
                "messages": [{"sender": m.get("sender"), "type": m.get("type")} for m in received_messages]
            }

        self.memory_manager.append_history(history_record)

        # Update state
        self.prev_coverage = current_coverage

        logger.info(
            f"[{self.agent_id}] Decision complete: destination=({final_destination[0]:.1f}, {final_destination[1]:.1f}), "
            f"strategy={strategy_name}"
        )

        # ========== Step 5: REFLECT (Phase 4/6) ==========
        thinking_future = None
        if current_time is not None and current_time > 0 and (current_time % 6 == 0 or current_time == 23):
            thinking_future = self._trigger_consolidation(current_time)

        # Phase 6 Sync: return target position and Future for main loop synchronization
        return self.current_destination, thinking_future

    def _execute_tool(
        self,
        llm_tools: Any,
        tool_name: str,
        tool_args: dict[str, Any]
    ) -> str:
        """Execute the LLM-requested tool call and log the result."""
        try:
            if tool_name == "emergency_move_tool":
                result = llm_tools.emergency_move_tool(
                    target_x=tool_args.get("target_x", 0),
                    target_y=tool_args.get("target_y", 0)
                )
            elif tool_name == "partition_space_tool":
                result = llm_tools.partition_space_tool()
            elif tool_name == "optimize_move_tool":
                result = llm_tools.optimize_move_tool()
            elif tool_name == "predictive_move_tool":
                result = llm_tools.predictive_move_tool(
                    target_x=tool_args.get("target_x", 0),
                    target_y=tool_args.get("target_y", 0),
                    reason=tool_args.get("reason", "")
                )
            else:
                result = f"Error: unknown tool '{tool_name}'"

            # Record tool execution result
            if hasattr(self, 'interaction_logger') and self.interaction_logger:
                self.interaction_logger.log_tool_result(tool_name, result)

            return result

        except Exception as e:
            error_msg = f"Tool execution error: {type(e).__name__}: {str(e)}"
            logger.error(f"Tool execution error: {type(e).__name__}: {e}")

            # Record error
            if hasattr(self, 'interaction_logger') and self.interaction_logger:
                self.interaction_logger.log_error(error_msg)

            return error_msg

    def _fallback_decision(
        self,
        help_request: Any,
        other_haps: npt.NDArray[np.float64],
        all_users: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], str]:
        """Fallback decision logic when LLM is unavailable. Uses partitioning + APF optimizer."""
        # Get exclusive users via Voronoi partitioning
        partition_info = self.partitioner.partition_with_info(
            all_users=all_users,
            my_pos=self.position,
            other_haps_positions=other_haps
        )
        assigned_users = partition_info["assigned_users"]

        # Emergency support priority
        if help_request is not None:
            target = np.array(help_request["target_pos"], dtype=np.float64)
            self.current_destination = target
            return target, "fallback_emergency"

        # Use APF optimization
        move_result = self.apf_optimizer.calculate_move(
            assigned_users=assigned_users,
            my_pos=self.position,
            other_haps_positions=other_haps,
            coverage_radius=210.0
        )

        new_pos = move_result["new_position"]
        self.current_destination = new_pos.copy()
        return new_pos, "fallback_apf"

    def _trigger_consolidation(self, current_time: int) -> concurrent.futures.Future | None:
        """Trigger memory reflection and pattern extraction via ThreadPoolExecutor (non-blocking)."""
        if self.memory_consolidator.llm_client is None:
            logger.debug(f"[{self.agent_id}] No LLM client, skipping consolidation")
            return None

        # Phase 6 Fix: check for running task (ThreadPoolExecutor mode)
        if self._pending_future is not None and not self._pending_future.done():
            logger.debug(f"[{self.agent_id}] Previous consolidation still running, skipping")
            return None

        # Get soul.md content as LLM system prompt
        soul_context = self.soul_prompt

        try:
            # Step 6a: Get last 6 hours of history
            recent_history = self.memory_manager.get_recent_history(k=6)

            if len(recent_history) < 2:
                logger.debug(f"[{self.agent_id}] Insufficient history for consolidation")
                return None

            logger.info(f"[{self.agent_id}] Submitting consolidation task to thread pool at t={current_time}")

            # Step 6b: Define sync worker function (executed in thread pool)
            def do_consolidation():
                """Thread pool worker: execute LLM call and result processing."""
                try:
                    # Snowball memory: read previous insight as prior knowledge
                    previous_insight = self.memory_manager.read_memory()

                    # Select optimization mode based on config
                    if self.use_ib_optimizer:
                        # Mode A: Lagrangian Information Bottleneck optimization (recommended)
                        # Optimization target: max L_Utility = D_JS_Proxy + beta * E[log P(Y|M)]
                        result = self.memory_consolidator.consolidate_with_ib(
                            soul_context=soul_context,
                            previous_insight=previous_insight,
                            recent_history_records=recent_history,
                            beta=self.ib_beta,
                            utility_threshold=self.ib_utility_threshold
                        )

                        insight = result.insight if result.should_update else ""
                        should_update = result.should_update
                        metrics = f"L={result.lagrangian_utility:.3f},D_JS={result.d_js_proxy:.3f},E_logP={result.expected_log_p:.3f}"

                    else:
                        # Mode B: Traditional cross-entropy threshold (compatibility mode)
                        insight, cross_entropy = self.memory_consolidator.consolidate(
                            soul_context=soul_context,
                            previous_insight=previous_insight,
                            recent_history_records=recent_history,
                            novelty_threshold=self.novelty_threshold,
                            enable_logprobs=True
                        )
                        should_update = bool(insight)
                        metrics = f"CE={cross_entropy:.3f}"

                    # Step 6c: IB-based decision: only write insights meeting utility threshold
                    if insight and should_update:
                        # Add timestamp and formatting
                        timestamp_str = f"\n## AI Reflection [{current_time}:00]\n\n"
                        consolidated_content = timestamp_str + insight + "\n"

                        # Append mode write (thread-safe)
                        self.memory_manager.write_memory(consolidated_content, mode="append")

                        logger.info(f"[{self.agent_id}] AI insight persisted [{metrics}]: {insight[:60]}...")
                    else:
                        logger.debug(f"[{self.agent_id}] LLM found no significant patterns")

                except Exception as e:
                    logger.error(f"[{self.agent_id}] Consolidation worker error: {e}")

            # Step 6b: Submit to ThreadPoolExecutor (FIXED: no asyncio!)
            self._pending_future = self.executor.submit(do_consolidation)
            logger.debug(f"[{self.agent_id}] Consolidation task submitted to thread pool")

            # Return Future for main loop synchronization
            return self._pending_future

        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to submit consolidation: {e}")
            return None

    def get_next_gui_move(self, max_step: float) -> npt.NDArray[np.float64]:
        """Get next GUI micro-move (called every 5 minutes). Smooths movement toward current_destination."""
        if self.current_destination is None:
            # No decision target yet, stay at current position
            return self.position.copy()

        # Calculate direction and distance to target
        direction = self.current_destination - self.position
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # Already close to target
            return self.position.copy()

        if distance <= max_step:
            # Can reach target directly
            new_position = self.current_destination.copy()
        else:
            # Move one small step toward target
            unit_direction = direction / distance
            new_position = self.position + unit_direction * max_step

        # Boundary clip [0, 1000]
        new_position[0] = np.clip(new_position[0], 0.0, 1000.0)
        new_position[1] = np.clip(new_position[1], 0.0, 1000.0)

        # Update current position
        self.position = new_position.copy()

        return new_position

    def _update_memory_cache(self, compressed_state: dict[str, Any]) -> None:
        """Update in-memory short-term cache as a fast-access layer (FIFO queue)."""
        self.workspace_memory.append(compressed_state)

        if len(self.workspace_memory) > self.memory_capacity:
            self.workspace_memory.pop(0)

        logger.debug(f"[{self.agent_id}] Memory cache updated: {len(self.workspace_memory)}/{self.memory_capacity} entries")

    def _generate_reasoning(
        self,
        compressed_state: dict[str, Any],
        optimization_result: dict[str, Any]
    ) -> str:
        """Generate human-readable decision reasoning."""
        summary = compressed_state["semantic_summary"]
        move_distance = optimization_result["move_distance"]
        converged = optimization_result["converged"]

        if converged:
            return f"Converged near optimal position, maintaining current location. Based on: {summary}"
        else:
            return (
                f"Moving {move_distance:.1f}km toward high-density user area to improve coverage."
                f"Decision basis: {summary}"
            )

    def get_memory_history(self, k: int = 5) -> list[dict[str, Any]]:
        """Get recent history records (reads from persistent file)."""
        return self.memory_manager.get_recent_history(k)

    def get_memory_info(self) -> dict[str, Any]:
        """Get memory system info summary."""
        info = self.memory_manager.get_workspace_info()
        info["memory_cache_size"] = len(self.workspace_memory)
        return info

    def get_position(self) -> npt.NDArray[np.float64]:
        """Get current position as a copy."""
        return self.position.copy()
