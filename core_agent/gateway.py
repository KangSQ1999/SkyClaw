"""Minimal inbox-based Pub/Sub communication gateway for multi-agent HAPS systems."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Gateway:
    """Minimal Pub/Sub communication gateway using in-memory mailboxes."""

    def __init__(self, agent_ids: list[str]) -> None:
        """Initialize gateway with an empty inbox for each agent."""
        self.agent_ids: list[str] = agent_ids.copy()

        # Initialize inbox dictionary: one empty list per agent
        self.inboxes: dict[str, list[dict[str, Any]]] = {
            aid: [] for aid in agent_ids
        }

        self.message_count: int = 0

        logger.info(f"Gateway initialized with {len(agent_ids)} agents: {agent_ids}")

    def broadcast(
        self,
        sender_id: str,
        message_dict: dict[str, Any]
    ) -> int:
        """Broadcast a message to all agents except the sender. Returns count of recipients."""
        # Validate sender identity
        if sender_id not in self.agent_ids:
            logger.warning(f"Unknown sender '{sender_id}', message dropped")
            return 0

        # Ensure message contains sender field
        if "sender" not in message_dict:
            message_dict["sender"] = sender_id

        # Broadcast to all agents except sender
        sent_count = 0
        for aid in self.agent_ids:
            if aid != sender_id:  # Don't send to self
                self.inboxes[aid].append(message_dict.copy())
                sent_count += 1

        self.message_count += sent_count

        logger.debug(
            f"[{sender_id}] Broadcast message to {sent_count} agents: "
            f"type={message_dict.get('type', 'UNKNOWN')}"
        )

        return sent_count

    def receive(self, agent_id: str) -> list[dict[str, Any]]:

        if agent_id not in self.agent_ids:
            logger.warning(f"Unknown agent '{agent_id}', returning empty list")
            return []

        # Retrieve all messages
        messages = self.inboxes[agent_id].copy()

        # Clear inbox (read-and-burn)
        self.inboxes[agent_id] = []

        if messages:
            logger.debug(f"[{agent_id}] Received {len(messages)} messages, inbox cleared")

        return messages

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics for debugging."""
        inbox_sizes = {aid: len(msgs) for aid, msgs in self.inboxes.items()}

        return {
            "registered_agents": len(self.agent_ids),
            "agent_ids": self.agent_ids.copy(),
            "total_messages_processed": self.message_count,
            "current_inbox_sizes": inbox_sizes,
            "total_pending_messages": sum(inbox_sizes.values()),
        }
