"""SkyClaw Lagrangian Information Bottleneck Memory Optimizer - cognitive memory evolution via IB theory."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IBOptimizationResult:
    """Result of Lagrangian IB optimization with all key metrics for memory update decisions."""

    insight: str
    d_js_proxy: float
    expected_log_p: float
    lagrangian_utility: float
    should_update: bool
    beta: float
    utility_threshold: float

    def __repr__(self) -> str:
        return (
            f"IBOptimizationResult("
            f"utility={self.lagrangian_utility:.4f}, "
            f"d_js={self.d_js_proxy:.4f}, "
            f"E_log_p={self.expected_log_p:.4f}, "
            f"update={self.should_update}"
            f")"
        )


class LagrangianMemoryOptimizer:

    def __init__(
        self,
        llm_client: Any | None = None,
        beta: float = 0.5,
        utility_threshold: float = 0.2,
        embedding_model: str = "text-embedding-3-small"
    ) -> None:
        """Initialize Lagrangian memory optimizer.

        Args:
            llm_client: LLM client instance (DeepSeekClient or compatible).
            beta: Lagrangian multiplier (controls compression vs prediction tradeoff).
            utility_threshold: Utility threshold for memory update decisions.
            embedding_model: Embedding model for semantic divergence.
        """
        self.llm_client = llm_client
        self.beta = beta
        self.utility_threshold = utility_threshold
        self.embedding_model = embedding_model

        if llm_client:
            logger.info(
                f"LagrangianMemoryOptimizer initialized "
                f"(beta={beta}, threshold={utility_threshold})"
            )
        else:
            logger.warning(
                "LagrangianMemoryOptimizer initialized WITHOUT LLM client (mock mode)"
            )

    def optimize(
        self,
        prior_memory: str,
        new_observations: str,
        system_prompt: str,
        temperature: float = 0.3
    ) -> IBOptimizationResult:
        """Perform Lagrangian IB optimization.

        Step 1: Compute semantic divergence D_JS_Proxy = 1 - cos(Emb(M), Emb(X_t))
        Step 2: Generate insight with logprobs to get E[log P(Y|M)]
        Step 3: Compute L = D_JS_Proxy + beta * E[logP]
        Step 4: Decision: update if L > theta
        """
        if self.llm_client is None:
            logger.debug("No LLM client available, skipping IB optimization")
            return IBOptimizationResult(
                insight="",
                d_js_proxy=0.0,
                expected_log_p=0.0,
                lagrangian_utility=0.0,
                should_update=False,
                beta=self.beta,
                utility_threshold=self.utility_threshold
            )

        try:
            # Step 1: Compute semantic divergence D_JS_Proxy
            logger.debug("Step 1: Computing semantic divergence (D_JS_Proxy)")
            d_js_proxy = self._compute_semantic_divergence(
                prior_memory=prior_memory,
                new_observations=new_observations
            )

            # Step 2 & 3: Build prompt and get prediction expectation E_log_P
            logger.debug("Step 2-3: Generating insight with logprobs")

            analysis_prompt = self._build_analysis_prompt(
                prior_memory=prior_memory,
                new_observations=new_observations
            )

            insight, expected_log_p = self._generate_insight_with_likelihood(
                system_prompt=system_prompt,
                user_prompt=analysis_prompt,
                temperature=temperature
            )

            if not insight or "NO_PATTERN" in insight.upper():
                logger.debug("LLM found no significant patterns")
                return IBOptimizationResult(
                    insight="",
                    d_js_proxy=d_js_proxy,
                    expected_log_p=expected_log_p,
                    lagrangian_utility=d_js_proxy + self.beta * expected_log_p,
                    should_update=False,
                    beta=self.beta,
                    utility_threshold=self.utility_threshold
                )

            insight = insight.strip()

            # Step 4: Compute Lagrangian utility
            # L_Utility = D_JS_Proxy + beta * E_log_P
            lagrangian_utility = d_js_proxy + self.beta * expected_log_p

            logger.info(
                f"IB Optimization: D_JS={d_js_proxy:.4f}, "
                f"E[logP]={expected_log_p:.4f}, "
                f"L={lagrangian_utility:.4f}"
            )

            # Step 5: Decision based on utility threshold
            should_update = lagrangian_utility > self.utility_threshold

            if should_update:
                logger.info(
                    f"Memory update APPROVED (L={lagrangian_utility:.4f} > "
                    f"theta={self.utility_threshold:.4f}): {insight[:80]}..."
                )
            else:
                logger.info(
                    f"Memory update REJECTED (L={lagrangian_utility:.4f} <= "
                    f"theta={self.utility_threshold:.4f}): Insight discarded"
                )

            return IBOptimizationResult(
                insight=insight if should_update else "",
                d_js_proxy=d_js_proxy,
                expected_log_p=expected_log_p,
                lagrangian_utility=lagrangian_utility,
                should_update=should_update,
                beta=self.beta,
                utility_threshold=self.utility_threshold
            )

        except Exception as e:
            logger.error(f"IB optimization failed: {type(e).__name__}: {e}")
            return IBOptimizationResult(
                insight="",
                d_js_proxy=0.0,
                expected_log_p=0.0,
                lagrangian_utility=0.0,
                should_update=False,
                beta=self.beta,
                utility_threshold=self.utility_threshold
            )

    def _compute_semantic_divergence(
        self,
        prior_memory: str,
        new_observations: str
    ) -> float:
        """Compute semantic divergence proxy D_JS_Proxy = 1 - cosine(Emb(M), Emb(X_t)).

        Uses embedding cosine distance as a heuristic proxy for JS divergence.
        Range: [0, 2] where 0 = identical semantics, 1 = orthogonal, 2 = opposite.
        """
        try:
            if not hasattr(self.llm_client, 'get_embedding'):
                logger.warning(
                    "LLM client does not support get_embedding, "
                    "using fallback divergence=0.5"
                )
                return 0.5

            embedding_m = self.llm_client.get_embedding(
                prior_memory, model=self.embedding_model
            )
            embedding_x = self.llm_client.get_embedding(
                new_observations, model=self.embedding_model
            )

            cosine_sim = self._cosine_similarity(embedding_m, embedding_x)
            d_js_proxy = 1.0 - cosine_sim

            logger.debug(
                f"Semantic divergence computed: {d_js_proxy:.4f} "
                f"(cosine_sim={cosine_sim:.4f})"
            )

            return d_js_proxy

        except Exception as e:
            logger.error(f"Failed to compute semantic divergence: {e}")
            return 0.5

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors. Returns value in [-1, 1]."""
        if len(vec_a) != len(vec_b):
            logger.error(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _generate_insight_with_likelihood(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> tuple[str, float]:
        """Generate insight and compute E[log P(Y|M)] via token-averaged logprobs.

        Returns (insight_text, expected_log_p).
        """
        try:
            if not hasattr(self.llm_client, 'generate_with_logprobs'):
                logger.warning(
                    "LLM client does not support generate_with_logprobs, "
                    "falling back to standard generation"
                )
                insight = self.llm_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature
                )
                return insight, -1.0

            response = self.llm_client.generate_with_logprobs(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature
            )

            if response is None:
                logger.error("LLM API call with logprobs failed")
                return "", 0.0

            insight = response.content
            expected_log_p = response.avg_logprob

            logger.debug(
                f"Insight generated ({len(insight)} chars), "
                f"E[logP]={expected_log_p:.4f}"
            )

            return insight, expected_log_p

        except Exception as e:
            logger.error(f"Failed to generate insight with likelihood: {e}")
            return "", 0.0

    def _build_analysis_prompt(
        self,
        prior_memory: str,
        new_observations: str
    ) -> str:
        """Build analysis prompt combining prior memory and new observations."""

        prompt = f"""=== PRIOR KNOWLEDGE (Historical Memory M) ===
{prior_memory if prior_memory else 'No prior insights available. The population was previously assumed to be scattered.'}

=== NEW OBSERVATIONS (Recent States X_t) ===
{new_observations}

=== ANALYSIS TASK ===
You are a HAPS data analyst. Analyze user movement patterns based on [Historical Memory] and [Last 6 Hours of Observations].

**Analysis Requirements (internal reasoning, do NOT output thought process):**

1. **Determine if HAPS itself is moving**: Compare HAPS_POS; change >50km = large movement, <10km = hovering.
2. **Determine who is moving**: If HAPS is moving but sector direction is unchanged -> airship approaches users; if HAPS is stationary but user count increases -> users actively converge.
3. **Detect Tidal Events** (ALL three conditions must be met; missing any = no tidal event):
   - User count in a sector increases monotonically over multiple hours (at least doubling, e.g. 30->60)
   - That sector is highly concentrated (single sector users >30% of total within perception range)
   - Mobile user ratio significantly elevated (>75%)
4. **Predict next step**: Based on above judgments, predict trends.

**Output Requirements (strict):**
- Give conclusions directly; do NOT list analysis steps.
- Summarize core findings in 1-2 sentences, including key numbers (user count changes, direction, mobile ratio).
- If changes are minimal, output: NO_PATTERN
- If a tidal event is detected, append at end: [Tidal_Event: direction_keyNumbers_centroid[x,y]_mobileRatio%_startHH:00]
  Centroid coordinates MUST be extracted from observation records (e.g. "Northeast sector(centroid[570.8,569.1]..."), do NOT fabricate.
  Start time is the hour the tidal event first became significantly apparent (e.g. if occurring 18:00-23:00, append _start18:00).
Output example 1:
As HAPS moved significantly 65km northward, observed users in North and Northeast sectors rose to 120. This is an observation artifact from the airship approaching static populations; distribution is expected to remain stable as the airship cruises.
Output example 2:
Over the past 6 hours, Southeast sector users surged from 28 to 77 with mobile ratio at 85%, showing clear active group convergence trend. [Tidal_Event: Southeast_surgedTo77_centroid[850.0,850.0]_mobileRatio85%_start18:00]
Your Analysis:"""

        return prompt
