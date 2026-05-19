"""Per-agent workspace manager for isolated file I/O and persistent memory."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryManager:
    """Agent memory manager: handles per-agent local file I/O for history, memory, and soul."""

    def __init__(self, agent_id: str) -> None:
        """Initialize memory manager: create workspace directory and init files if needed."""
        self.agent_id = agent_id

        # Determine workspace directory path
        # Derive from current file location: memory_manager.py -> workspace/ -> core_agent/ -> project_root/
        current_file = Path(__file__).resolve()
        workspace_root = current_file.parent  # core_agent/workspace/
        self.workspace_dir = workspace_root / agent_id

        # Determine file paths
        self.history_file = self.workspace_dir / "history.jsonl"
        self.memory_file = self.workspace_dir / "memory.md"
        self.soul_file = self.workspace_dir / "soul.md"

        # Initialize workspace
        self._initialize_workspace()

        logger.debug(f"MemoryManager initialized for '{agent_id}' at {self.workspace_dir}")

    def _initialize_workspace(self) -> None:
        """Initialize workspace directory and files (called by __init__)."""
        # 1. Create workspace directory (exist_ok=True avoids race conditions)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # 2. Initialize history.jsonl if it doesn't exist
        if not self.history_file.exists():
            self.history_file.touch()
            logger.debug(f"Created empty history file: {self.history_file}")

        # 3. Initialize memory.md if it doesn't exist
        if not self.memory_file.exists():
            initial_content = f"# {self.agent_id} Long-Term Memory\n\n"
            initial_content += f"This is {self.agent_id}'s persistent memory storage.\n"
            initial_content += "Significant events, patterns, and experiences accumulate here.\n\n"
            initial_content += "## Memory Records\n\n"

            with open(self.memory_file, "w", encoding="utf-8") as f:
                f.write(initial_content)

            logger.debug(f"Created initial memory file: {self.memory_file}")

        # 4. Initialize soul.md if it doesn't exist
        if not self.soul_file.exists():
            soul_content = f"""# Role
You are a High Altitude Platform Station (HAPS) in the SkyClaw simulation system, designated: {self.agent_id}.

# Objective
Your core mission is to move to optimal 2D spatial coordinates to maximize coverage of dynamic ground users, while minimizing coverage overlap with other HAPS nodes.

# Constraints
* You must rely on locally mounted Tools for precise geometric coordinate calculations.
* Your communication bandwidth is limited; you must express your spatial intent with maximum conciseness.
"""
            with open(self.soul_file, "w", encoding="utf-8") as f:
                f.write(soul_content)

            logger.debug(f"Created soul file: {self.soul_file}")

    def append_history(self, record_dict: dict[str, Any]) -> None:
        """Append a record to history.jsonl using JSON Lines format (O(1) write)."""
        # Serialize dict to JSON string
        json_line = json.dumps(record_dict, ensure_ascii=False, separators=(',', ':'))

        # Append mode write ('a' = append), each record on its own line
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        logger.debug(f"[{self.agent_id}] Appended history record at t={record_dict.get('time', '?')}")

    def get_recent_history(self, k: int = 5) -> list[dict[str, Any]]:
        """Read the last k records from history.jsonl. Returns empty list if file is empty."""
        # Check if file exists and is non-empty
        if not self.history_file.exists() or self.history_file.stat().st_size == 0:
            return []

        # Read all lines
        with open(self.history_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Take the last k lines
        recent_lines = lines[-k:] if len(lines) >= k else lines

        # Parse JSON
        records = []
        for line in recent_lines:
            line = line.strip()
            if line:  # Skip blank lines
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"[{self.agent_id}] Failed to parse history line: {e}")
                    continue

        return records

    def read_memory(self) -> str:
        """Read the full content of memory.md. Returns empty string if file doesn't exist."""
        if not self.memory_file.exists():
            return ""

        with open(self.memory_file, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    def write_memory(self, content: str, mode: str = "append") -> None:
        """Write content to memory.md. Mode: "append" (default) or "overwrite"."""
        if mode == "overwrite":
            with open(self.memory_file, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(content)

        logger.debug(f"[{self.agent_id}] Wrote to memory.md (mode={mode})")

    def get_workspace_info(self) -> dict[str, Any]:
        """Get workspace info summary for debugging."""
        info = {
            "agent_id": self.agent_id,
            "workspace_dir": str(self.workspace_dir),
            "history_file": str(self.history_file),
            "memory_file": str(self.memory_file),
        }

        # Count history.jsonl lines
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            info["history_count"] = line_count
        else:
            info["history_count"] = 0

        # Memory.md file size
        if self.memory_file.exists():
            info["memory_size_bytes"] = self.memory_file.stat().st_size
        else:
            info["memory_size_bytes"] = 0

        return info
