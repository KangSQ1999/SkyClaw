"""DeepSeek API client (OpenAI-compatible) with Tool Calling and Embedding support."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from core_agent.llm_logger import LLMInteractionLogger

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM response result structure."""

    content: str  # Text content (if not a tool call)
    is_tool_call: bool  # Whether this is a tool call
    tool_name: str | None = None  # Tool name
    tool_args: dict[str, Any] | None = None  # Tool arguments
    tool_call_id: str | None = None  # Tool call ID
    assistant_message: dict[str, Any] | None = None  # Assistant message with tool_calls
    error: str | None = None  # Error message when API call fails


@dataclass
class LLMResponseWithLogprobs:
    """LLM response with logprobs for cross-entropy-based novelty evaluation."""

    content: str  # Generated text
    cross_entropy: float  # Cross-entropy H = -mean(logprobs)
    logprobs: list[float]  # Log probabilities for all tokens
    avg_logprob: float  # Average log probability


class DeepSeekClient:
    """DeepSeek API client (OpenAI-compatible format) with Function Calling support."""

    _instance: DeepSeekClient | None = None

    def __new__(cls) -> DeepSeekClient:
        """Singleton: ensure only one global client instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize DeepSeek client (runs only once due to singleton pattern)."""
        if self._initialized:
            return

        # Read API Key from environment variable
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found in environment variables. "
                "Please set it before running: export DEEPSEEK_API_KEY='your-key'"
            )

        try:
            from openai import OpenAI

            # Configure DeepSeek client (LLM chat)
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            # Model name: deepseek-v4-flash (default, latest version)
            # Legacy deepseek-chat / deepseek-reasoner deprecated 2026/07/24
            self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

            # Configure Embedding client (SiliconFlow)
            # DeepSeek does not provide Embedding API; use SiliconFlow as separate backend
            # Supported models: BAAI/bge-m3 (multilingual, 1024-dim, recommended)
            sf_api_key = os.environ.get("SILICONFLOW_API_KEY")
            if sf_api_key:
                self.embedding_client = OpenAI(
                    api_key=sf_api_key,
                    base_url="https://api.siliconflow.cn/v1"
                )
                self.embedding_model = os.environ.get(
                    "SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"
                )
                logger.info(
                    f"Embedding client initialized (SiliconFlow, "
                    f"model: {self.embedding_model})"
                )
            else:
                self.embedding_client = None
                logger.warning(
                    "SILICONFLOW_API_KEY not found. Embedding will use "
                    "fallback hash-based vectors. Set SILICONFLOW_API_KEY "
                    "to enable real semantic divergence."
                )

            logger.info(f"DeepSeekClient initialized (model: {self.model})")
            self._initialized = True

        except ImportError as e:
            raise ImportError(
                "openai package not installed. "
                "Run: pip install openai>=1.0.0"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DeepSeek client: {e}") from e

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        interaction_logger: LLMInteractionLogger | None = None
    ) -> str:
        """Generate text (basic interface). Returns clean text or empty string on error."""
        if not self._initialized:
            logger.error("DeepSeekClient not initialized")
            return ""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Log request
            if interaction_logger:
                interaction_logger.log_request(
                    system_prompt=system_prompt,
                    messages=messages
                )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                extra_body={"thinking": {"type": "disabled"}}
            )

            result = response.choices[0].message.content.strip()

            # Log response
            if interaction_logger:
                interaction_logger.log_response(result)

            logger.debug(f"LLM generated {len(result)} chars")
            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"LLM API error: {error_msg}")

            # Log error
            if interaction_logger:
                interaction_logger.log_error(error_msg)

            return ""

    # Alias: keep method name consistent with callers
    generate_sync = generate

    @staticmethod
    def compute_cross_entropy(logprobs: list[float] | None) -> tuple[float, float]:
        """Compute cross-entropy for semantic novelty evaluation.

        Formula: H = -(1/N) * sum(log(p_i))
        Higher values indicate greater novelty.
        """
        if not logprobs or len(logprobs) == 0:
            logger.warning("Empty logprobs provided, returning default values")
            return 0.0, 0.0

        try:
            # Filter out invalid values (None, NaN)
            valid_logprobs = [lp for lp in logprobs if lp is not None and not (isinstance(lp, float) and (lp != lp))]  # lp != lp checks for NaN

            if len(valid_logprobs) == 0:
                logger.warning("No valid logprobs after filtering")
                return 0.0, 0.0

            # Compute average log probability
            avg_logprob = sum(valid_logprobs) / len(valid_logprobs)

            # Compute cross-entropy: H = -mean(logprobs)
            cross_entropy = -avg_logprob

            logger.debug(f"Cross-Entropy computed: {cross_entropy:.4f}, avg_logprob: {avg_logprob:.4f}, tokens: {len(valid_logprobs)}")

            return cross_entropy, avg_logprob

        except Exception as e:
            logger.error(f"Error computing cross-entropy: {type(e).__name__}: {e}")
            return 0.0, 0.0

    def generate_with_logprobs(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        interaction_logger: LLMInteractionLogger | None = None
    ) -> LLMResponseWithLogprobs | None:
        """Generate text with logprobs for cross-entropy computation (cognitive IB).

        Used in the Reflection phase for semantic novelty evaluation.
        """
        if not self._initialized:
            logger.error("DeepSeekClient not initialized")
            return None

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Log request
            if interaction_logger:
                interaction_logger.log_request(
                    system_prompt=system_prompt,
                    messages=messages
                )

            # Call LLM API with logprobs enabled
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                logprobs=True,
                top_logprobs=5,
                extra_body={"thinking": {"type": "disabled"}}
            )

            # Extract generated text
            content = response.choices[0].message.content.strip()

            # Extract logprobs
            logprobs_list: list[float] = []
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                logprobs_content = response.choices[0].logprobs.content
                if logprobs_content:
                    for token_logprob in logprobs_content:
                        if hasattr(token_logprob, 'logprob'):
                            logprobs_list.append(token_logprob.logprob)

            # Compute cross-entropy
            cross_entropy, avg_logprob = self.compute_cross_entropy(logprobs_list)

            # Log response
            if interaction_logger:
                interaction_logger.log_response(
                    f"{content}\n[Cross-Entropy: {cross_entropy:.4f}]"
                )

            logger.info(f"LLM generated {len(content)} chars, Cross-Entropy: {cross_entropy:.4f}")

            return LLMResponseWithLogprobs(
                content=content,
                cross_entropy=cross_entropy,
                logprobs=logprobs_list,
                avg_logprob=avg_logprob
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"LLM API error with logprobs: {error_msg}")

            if interaction_logger:
                interaction_logger.log_error(error_msg)

            return None

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> list[float]:
        """Get text embedding vector for semantic divergence computation.

        Multi-backend support: SiliconFlow (BAAI/bge-m3), OpenAI, or hash fallback.
        Used for Lagrangian IB: D_JS_Proxy = 1 - cos(Emb(M), Emb(X_t))
        """
        if not self._initialized:
            logger.error("DeepSeekClient not initialized")
            # Return zero vector as fallback
            return [0.0] * 1536

        try:
            # Ensure text is not empty
            if not text or not text.strip():
                text = "empty"

            # Use SiliconFlow client first; fallback if not configured
            if self.embedding_client is None:
                logger.warning("Embedding client not available, using fallback")
                return self._generate_fallback_embedding(text)

            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )

            # Extract embedding vector
            embedding = response.data[0].embedding

            logger.debug(f"Embedding generated: {len(embedding)} dims, text_len={len(text)}")

            return embedding

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"Embedding API error: {error_msg}")

            # Fallback: generate pseudo-random but deterministic vector based on text hash
            # Same text always produces the same embedding, maintaining some consistency
            logger.warning("Using fallback hash-based embedding")
            return self._generate_fallback_embedding(text)

    def _generate_fallback_embedding(self, text: str, dim: int = 1536) -> list[float]:
        """Generate hash-based fallback embedding when the Embedding API is unavailable.

        Same input text always produces the same deterministic vector.
        """
        import hashlib

        # Use MD5 hash as random seed
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash, 16)

        # Generate pseudo-random vector using linear congruential generator
        vector = []
        state = seed % (2**31)

        for _ in range(dim):
            # Linear congruential generator
            state = (state * 1103515245 + 12345) % (2**31)
            # Normalize to [-1, 1]
            value = (state / (2**31)) * 2 - 1
            vector.append(value)

        # L2 normalization (makes it more like a real embedding)
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def call_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        interaction_logger: LLMInteractionLogger | None = None
    ) -> LLMResponse:
        """Single LLM call with Tool Calling support. Caller manages multi-turn conversation."""
        if not self._initialized:
            logger.error("DeepSeekClient not initialized")
            return LLMResponse(
                content="",
                is_tool_call=False,
                error="Client not initialized"
            )

        try:
            # Build full conversation history
            full_messages = [{"role": "system", "content": system_prompt}]
            full_messages.extend(messages)

            # Log request
            if interaction_logger:
                interaction_logger.log_request(
                    system_prompt=system_prompt,
                    messages=messages.copy(),
                    tools=tools.copy() if tools else None
                )

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature,
                tools=tools if tools else None,
                extra_body={"thinking": {"type": "disabled"}}
            )

            message = response.choices[0].message

            # Check for tool calls
            if message.tool_calls:
                tool_call = message.tool_calls[0]  # Process the first tool call
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info(f"LLM requested tool call: {tool_name}({tool_args})")

                # Log tool call
                if interaction_logger:
                    interaction_logger.log_tool_call(tool_name, tool_args)

                # Build assistant message for conversation history
                assistant_message = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                    ]
                }

                # DeepSeek v4 thinking mode: must pass back reasoning_content
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    assistant_message["reasoning_content"] = message.reasoning_content

                return LLMResponse(
                    content="",
                    is_tool_call=True,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_call_id=tool_call.id,
                    assistant_message=assistant_message
                )

            else:
                # LLM gave a final answer
                content = message.content.strip() if message.content else ""
                logger.info(f"[{self.model}] LLM response: {content[:100]}...")

                # Log response
                if interaction_logger:
                    interaction_logger.log_response(content)

                return LLMResponse(
                    content=content,
                    is_tool_call=False
                )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"LLM API error: {error_msg}")

            # Log error
            if interaction_logger:
                interaction_logger.log_error(error_msg)

            return LLMResponse(
                content="",
                is_tool_call=False,
                error=error_msg
            )

    # Keep legacy method name for backward compatibility
    generate_with_tools = call_with_tools

    def consolidate(
        self,
        soul_context: str,
        recent_history_records: list[dict[str, Any]]
    ) -> str:
        """Extract patterns from historical records using LLM (legacy interface)."""
        # Check record count
        if len(recent_history_records) < 2:
            logger.debug("Insufficient history for consolidation (< 2 records)")
            return ""

        try:
            # Format historical records
            history_text = self._format_history(recent_history_records)

            # Build user prompt
            user_prompt = self._build_analysis_prompt(history_text)

            # Call LLM
            logger.info(f"Calling DeepSeek API for pattern extraction ({len(recent_history_records)} records)")

            insight = self.generate(
                system_prompt=soul_context,
                user_prompt=user_prompt,
                temperature=0.3
            )

            # Process response
            if not insight or "NO_PATTERN" in insight.upper():
                logger.debug("LLM found no significant patterns")
                return ""

            # Clean and return
            insight = insight.strip()
            logger.info(f"LLM extracted insight: {insight[:80]}...")
            return insight

        except Exception as e:
            logger.error(f"LLM consolidation failed: {type(e).__name__}: {e}")
            return ""

    def _format_history(self, records: list[dict[str, Any]]) -> str:
        """Format JSON history records as human-readable text."""
        lines = []

        for record in records:
            time = record.get("time", "?")
            obs = record.get("observation", {})
            centroid = obs.get("centroid", ["?", "?"])
            density = obs.get("density_score", 0)
            user_count = obs.get("user_count", 0)
            summary = obs.get("semantic_summary", "N/A")[:80]

            line = (
                f"[T{time:02d}] Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}) | "
                f"Density: {density:.2f} | Users: {user_count} | {summary}..."
            )
            lines.append(line)

        return "\n".join(lines)

    def _build_analysis_prompt(self, history_text: str) -> str:
        """Build the analysis task prompt for the LLM."""
        prompt = f"""As a senior HAPS (High Altitude Platform Station) operations analyst, analyze the following 6-hour operational history.

Your task is to identify patterns in user movement and density that could inform coverage optimization strategies.

=== HISTORICAL RECORDS (Past 6 Hours) ===
{history_text}

=== ANALYSIS REQUIREMENTS ===
Focus on identifying:
1. **User Movement Patterns**: Are users moving in a consistent direction (especially toward the northeast corner around coordinates [850, 850])?
2. **Tidal Gathering Events**: During evening hours (18:00-24:00), do you observe significant gatherings or density increases?
3. **Temporal Trends**: How does user density evolve over the 6-hour period?
4. **Actionable Insights**: What specific recommendations would you make for HAPS positioning?

=== RESPONSE FORMAT ===
* If you detect SIGNIFICANT patterns (especially user gatherings toward the northeast during evening), provide a concise 2-3 sentence summary.
* If no clear patterns exist or data is inconclusive, respond with exactly: NO_PATTERN
* Be specific about time periods and spatial coordinates when possible.

Your Analysis:"""

        return prompt


# Backward compatibility: GeminiClient alias points to DeepSeekClient
GeminiClient = DeepSeekClient
