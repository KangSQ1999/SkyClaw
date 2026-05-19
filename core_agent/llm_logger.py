"""LLM interaction logger for debugging and analyzing LLM decision-making."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LLMInteractionLogger:
    """Logs complete LLM interactions: requests, responses, tool calls, and errors."""

    def __init__(self, agent_id: str, output_dir: Path | None = None) -> None:
        """Initialize the interaction logger."""
        self.agent_id = agent_id
        self.output_dir = output_dir
        self.log_file: Path | None = None
        self.interactions: list[dict[str, Any]] = []

        if output_dir is not None:
            self._set_log_file(output_dir)

    def set_output_dir(self, output_dir: Path) -> None:
        """Set output directory for deferred initialization."""
        self.output_dir = output_dir
        self._set_log_file(output_dir)

    def _set_log_file(self, output_dir: Path) -> None:
        """Set log file path and create directory."""
        agent_log_dir = output_dir / "llm_logs"
        agent_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = agent_log_dir / f"{self.agent_id}_interactions.json"
        logger.info(f"[{self.agent_id}] LLM interaction log: {self.log_file}")

    def log_interaction(
        self,
        interaction_type: str,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        response: str | None = None,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        tool_result: str | None = None,
        error: str | None = None
    ) -> None:
        """Record a single LLM interaction entry and flush to file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "type": interaction_type
        }

        if system_prompt is not None:
            entry["system_prompt"] = system_prompt

        if messages is not None:
            entry["messages"] = messages

        if tools is not None:
            entry["tools"] = tools

        if response is not None:
            entry["response"] = response

        if tool_name is not None:
            entry["tool_name"] = tool_name

        if tool_args is not None:
            entry["tool_args"] = tool_args

        if tool_result is not None:
            entry["tool_result"] = tool_result

        if error is not None:
            entry["error"] = error

        self.interactions.append(entry)

        # Write to file immediately (append mode)
        self._flush_to_file()

    def log_request(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None
    ) -> None:
        """Log a request sent to the LLM."""
        self.log_interaction(
            interaction_type="request",
            system_prompt=system_prompt,
            messages=messages.copy() if messages else None,
            tools=tools.copy() if tools else None
        )
        logger.debug(f"[{self.agent_id}] Logged LLM request ({len(messages)} messages)")

    def log_response(self, response: str) -> None:
        """Log an LLM response."""
        self.log_interaction(
            interaction_type="response",
            response=response
        )
        logger.debug(f"[{self.agent_id}] Logged LLM response ({len(response)} chars)")

    def log_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any]
    ) -> None:
        """Log an LLM-requested tool call."""
        self.log_interaction(
            interaction_type="tool_call",
            tool_name=tool_name,
            tool_args=tool_args.copy()
        )
        logger.info(f"[{self.agent_id}] Logged tool call: {tool_name}")

    def log_tool_result(self, tool_name: str, result: str) -> None:
        """Log a tool execution result."""
        self.log_interaction(
            interaction_type="tool_result",
            tool_name=tool_name,
            tool_result=result
        )
        logger.debug(f"[{self.agent_id}] Logged tool result: {tool_name}")

    def log_error(self, error: str) -> None:
        """Log an error message."""
        self.log_interaction(
            interaction_type="error",
            error=error
        )
        logger.error(f"[{self.agent_id}] Logged error: {error[:100]}...")

    def _flush_to_file(self) -> None:
        """Write interaction records to file."""
        if self.log_file is None:
            return

        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.interactions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write LLM log: {e}")

    def get_summary(self) -> dict[str, Any]:
        """Get interaction summary statistics."""
        total = len(self.interactions)
        requests = sum(1 for i in self.interactions if i.get("type") == "request")
        responses = sum(1 for i in self.interactions if i.get("type") == "response")
        tool_calls = sum(1 for i in self.interactions if i.get("type") == "tool_call")
        errors = sum(1 for i in self.interactions if i.get("type") == "error")

        return {
            "agent_id": self.agent_id,
            "total_interactions": total,
            "requests": requests,
            "responses": responses,
            "tool_calls": tool_calls,
            "errors": errors,
            "log_file": str(self.log_file) if self.log_file else None
        }
