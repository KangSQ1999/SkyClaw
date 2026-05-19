"""SkyClaw Memory Consolidator - Lagrangian Information Bottleneck integration for memory reflection."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Memory consolidator supporting two cognitive optimization modes.

    Mode A: Lagrangian Information Bottleneck (recommended, default)
    Mode B: Cross-entropy threshold (legacy compatibility)
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        use_ib_optimizer: bool = True
    ) -> None:
        """Initialize memory consolidator.

        Args:
            llm_client: DeepSeekClient instance.
            use_ib_optimizer: Whether to default to Lagrangian IB optimizer.
        """
        self.llm_client = llm_client
        self.use_ib_optimizer = use_ib_optimizer
        self._ib_optimizer: Any | None = None

        if llm_client:
            logger.info(
                f"MemoryConsolidator initialized "
                f"({'IB Optimizer' if use_ib_optimizer else 'Legacy CE mode'})"
            )
        else:
            logger.warning(
                "MemoryConsolidator initialized WITHOUT LLM client (mock mode)"
            )

    def _get_ib_optimizer(
        self,
        beta: float = 1.0,
        utility_threshold: float = 0.2
    ) -> Any:
        """Get or create LagrangianMemoryOptimizer instance (lazy init)."""
        if self._ib_optimizer is None:
            from skills.lagrangian_memory_optimizer import LagrangianMemoryOptimizer

            self._ib_optimizer = LagrangianMemoryOptimizer(
                llm_client=self.llm_client,
                beta=beta,
                utility_threshold=utility_threshold
            )
        else:
            self._ib_optimizer.beta = beta
            self._ib_optimizer.utility_threshold = utility_threshold

        return self._ib_optimizer

    def consolidate(
        self,
        soul_context: str,
        previous_insight: str,
        recent_history_records: list[dict[str, Any]],
        **kwargs: Any
    ) -> tuple[str, float] | Any:
        """Unified memory consolidation entry point (auto-routes to appropriate mode)."""
        use_ib = (
            self.use_ib_optimizer or
            'ib_beta' in kwargs or
            'ib_utility_threshold' in kwargs
        )

        if use_ib:
            ib_beta = kwargs.get('ib_beta', 1.0)
            ib_utility_threshold = kwargs.get('ib_utility_threshold', 0.2)

            result = self.consolidate_with_ib(
                soul_context=soul_context,
                previous_insight=previous_insight,
                recent_history_records=recent_history_records,
                beta=ib_beta,
                utility_threshold=ib_utility_threshold
            )
            return result
        else:
            novelty_threshold = kwargs.get('novelty_threshold', 0.5)
            enable_logprobs = kwargs.get('enable_logprobs', True)

            return self._consolidate_legacy(
                soul_context=soul_context,
                previous_insight=previous_insight,
                recent_history_records=recent_history_records,
                novelty_threshold=novelty_threshold,
                enable_logprobs=enable_logprobs
            )

    def consolidate_with_ib(
        self,
        soul_context: str,
        previous_insight: str,
        recent_history_records: list[dict[str, Any]],
        beta: float = 1.0,
        utility_threshold: float = 0.2
    ) -> Any:
        """Consolidate memory using Lagrangian Information Bottleneck optimizer.

        Maximizes L = D_JS(M_t || X_t) + beta * E[log P(Y_{t+1} | M_{t+1})]
        Update if L > theta.
        """
        if self.llm_client is None:
            logger.debug("No LLM client available, skipping IB consolidation")

            from skills.lagrangian_memory_optimizer import IBOptimizationResult
            return IBOptimizationResult(
                insight="",
                d_js_proxy=0.0,
                expected_log_p=0.0,
                lagrangian_utility=0.0,
                should_update=False,
                beta=beta,
                utility_threshold=utility_threshold
            )

        if len(recent_history_records) < 2:
            logger.debug("Insufficient history for consolidation (< 2 records)")

            from skills.lagrangian_memory_optimizer import IBOptimizationResult
            return IBOptimizationResult(
                insight="",
                d_js_proxy=0.0,
                expected_log_p=0.0,
                lagrangian_utility=0.0,
                should_update=False,
                beta=beta,
                utility_threshold=utility_threshold
            )

        try:
            history_text = self._format_history(recent_history_records)
            optimizer = self._get_ib_optimizer(beta, utility_threshold)

            logger.info(
                f"Running Lagrangian IB optimization "
                f"(beta={beta}, theta={utility_threshold}, {len(recent_history_records)} records)"
            )

            result = optimizer.optimize(
                prior_memory=previous_insight,
                new_observations=history_text,
                system_prompt=soul_context,
                temperature=0.3
            )

            logger.info(
                f"IB optimization complete: "
                f"L={result.lagrangian_utility:.4f}, "
                f"update={result.should_update}"
            )

            return result

        except Exception as e:
            logger.error(f"IB consolidation failed: {type(e).__name__}: {e}")

            from skills.lagrangian_memory_optimizer import IBOptimizationResult
            return IBOptimizationResult(
                insight="",
                d_js_proxy=0.0,
                expected_log_p=0.0,
                lagrangian_utility=0.0,
                should_update=False,
                beta=beta,
                utility_threshold=utility_threshold
            )

    def _consolidate_legacy(
        self,
        soul_context: str,
        previous_insight: str,
        recent_history_records: list[dict[str, Any]],
        novelty_threshold: float = 0.5,
        enable_logprobs: bool = True
    ) -> tuple[str, float]:
        """Traditional cross-entropy memory consolidation (compatibility mode).

        Decision: if cross_entropy >= novelty_threshold, update memory.
        """
        if self.llm_client is None:
            logger.debug("No LLM client available, skipping legacy consolidation")
            return "", 0.0

        if len(recent_history_records) < 2:
            logger.debug("Insufficient history for consolidation (< 2 records)")
            return "", 0.0

        try:
            history_text = self._format_history(recent_history_records)
            user_prompt = self._build_analysis_prompt(previous_insight, history_text)

            logger.info(
                f"Running legacy CE consolidation "
                f"({len(recent_history_records)} records)"
            )

            if enable_logprobs and hasattr(self.llm_client, 'generate_with_logprobs'):
                response = self.llm_client.generate_with_logprobs(
                    system_prompt=soul_context,
                    user_prompt=user_prompt,
                    temperature=0.3
                )

                if response is None:
                    logger.error("LLM API call with logprobs failed")
                    return "", 0.0

                insight = response.content
                cross_entropy = response.cross_entropy

                logger.info(f"Cross-Entropy computed: {cross_entropy:.4f}")
            else:
                insight = self.llm_client.generate_sync(
                    system_prompt=soul_context,
                    user_prompt=user_prompt,
                    temperature=0.3
                )
                cross_entropy = 0.5
                logger.warning("Using legacy mode without logprobs")

            if not insight or "NO_PATTERN" in insight.upper():
                logger.debug("LLM found no significant patterns")
                return "", cross_entropy

            insight = insight.strip()

            if cross_entropy < novelty_threshold:
                logger.info(
                    f"Low novelty (CE={cross_entropy:.4f} < {novelty_threshold:.4f}). "
                    f"Insight discarded."
                )
                return "", cross_entropy
            else:
                logger.info(
                    f"High novelty (CE={cross_entropy:.4f} >= {novelty_threshold:.4f}). "
                    f"Insight accepted: {insight[:80]}..."
                )
                return insight, cross_entropy

        except Exception as e:
            logger.error(f"Legacy consolidation failed: {type(e).__name__}: {e}")
            return "", 0.0

    def _format_history(self, records: list[dict[str, Any]]) -> str:
        """Format JSON history records into human-readable text."""
        lines = []

        for record in records:
            time_val = record.get("time", -1)
            try:
                time_str = f"{int(time_val):02d}"
            except (ValueError, TypeError):
                time_str = str(time_val)

            agent_pos = record.get("agent_position", [None, None])
            pos_str = f"[{agent_pos[0]:.1f},{agent_pos[1]:.1f}]" if agent_pos[0] is not None else "[?,?]"

            obs = record.get("observation", {})

            sectors = obs.get("sectors", [])
            sector_count = len(sectors)
            total_local_users = obs.get("total_local_users", 0)
            semantic_summary = obs.get("semantic_summary", "N/A")

            line = (
                f"[T{time_str}] HAPS_POS:{pos_str} | "
                f"Users:{total_local_users} | Sectors:{sector_count} | {semantic_summary}"
            )
            lines.append(line)

        return "\n".join(lines)

    def _build_analysis_prompt(self, previous_insight: str, history_text: str) -> str:
        """Build analysis prompt for the LLM."""
        prompt = f"""=== HISTORICAL MEMORY ===
{previous_insight if previous_insight else 'No historical memory.'}

=== LAST 6 HOURS OF OBSERVATIONS ===
{history_text}

=== ANALYSIS REQUIREMENTS ===
Compare historical memory with new observations and update your pattern summary. You must answer these two core questions:

1. **Who is moving?** Based on each record's HAPS_POS, determine:
   - Is HAPS itself actively moving toward user clusters?
   - Are user groups migrating toward HAPS' location?
   - Or are both moving?

2. **Tidal Event?** If records contain [Tidal Period] markers, determine:
   - Are users clearly converging toward the northeast corner [900,900]?
   - Has the mobile user ratio significantly increased?

=== OUTPUT FORMAT ===
* Summarize patterns in 1-2 sentences.
* If a tidal-period northeast convergence is detected, **first sentence MUST be**: "Detected tidal-period northeast convergence."
* If no clear pattern, output: NO_PATTERN
* If large-scale unidirectional migration is detected, append at end: [CRITICAL_SHIFT: direction_description]

Your analysis:"""

        return prompt
