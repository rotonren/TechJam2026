"""Cross-session policy memory: the agent's own guidance logic, refined.

`QuestionPolicy` decides which attribute to ask about partly from a table of
hand-written response likelihoods - our guess at how often a shopper can
actually answer a question about material, size, budget and so on. A guess is
all it was, and nothing ever checked it.

Every turn supplies the evidence to check it. The agent asks about one
attribute, and the next message either discloses a new requirement or says
there is nothing more to give. That is a clean per-turn observation of the
quantity the table is trying to estimate, and it costs nothing to collect.

`PolicyMemory` treats the hand-written value as a Beta prior and updates a
posterior from those observations. The first session therefore behaves exactly
as the hand-written table did, and the estimate moves only as far as the
evidence carries it. A deployment that constructs a fresh agent per session
simply never accumulates evidence and keeps the prior, so the memory can
improve behaviour but can never be required for it.

Observations never touch ground truth. The agent learns whether its own
question was productive, not whether its recommendations were right - the same
implicit signal a production recommender reads off user behaviour.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# The hand-written starting point, previously private to `question_policy`.
DEFAULT_RESPONSE_LIKELIHOOD: Mapping[str, float] = {
    "category": 0.95,
    "material": 0.90,
    "color": 0.90,
    "size": 0.85,
    "style": 0.80,
    "brand": 0.65,
    "budget": 0.90,
    "feature": 0.70,
    "use_case": 0.85,
    "other": 0.75,
}


@dataclass(frozen=True)
class YieldStat:
    """How often asking about an attribute produced a new requirement."""

    asked: int = 0
    disclosed: int = 0

    def observe(self, disclosed: bool) -> YieldStat:
        return YieldStat(self.asked + 1, self.disclosed + int(disclosed))

    @property
    def rate(self) -> float:
        return self.disclosed / self.asked if self.asked else 0.0


def profile_segment(user_profile: Mapping[str, object] | None) -> str:
    """Derive a stable shopper segment from the safe aggregate profile.

    Segments exist so the memory can hold "shoppers who talk about fit answer
    size questions" without ever identifying a person. Only the organizer's
    controlled preference tags are used, sorted so the signature is stable.
    """
    if not user_profile:
        return ""
    tags = user_profile.get("preference_tags")
    if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes)):
        return ""
    cleaned = sorted({str(tag).strip().lower() for tag in tags if str(tag).strip()})
    return "|".join(cleaned[:4])


class PolicyMemory:
    """Posterior estimate of per-attribute question yield."""

    def __init__(
        self,
        prior: Mapping[str, float] | None = None,
        *,
        prior_strength: float = 8.0,
        segment_floor: int = 12,
        max_segments: int = 256,
        enabled: bool = True,
    ) -> None:
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        if segment_floor < 1:
            raise ValueError("segment_floor must be positive")
        if max_segments < 1:
            raise ValueError("max_segments must be positive")
        self.prior = dict(prior or DEFAULT_RESPONSE_LIKELIHOOD)
        self.prior_strength = prior_strength
        self.segment_floor = segment_floor
        self.max_segments = max_segments
        self.enabled = enabled and os.environ.get(
            "COMPASSCART_DISABLE_EVOLUTION"
        ) != "1"
        self._global: dict[str, YieldStat] = {}
        self._contexts: dict[str, dict[str, YieldStat]] = {}
        self._segments: dict[str, dict[str, YieldStat]] = {}

    def observe(
        self,
        attribute: str,
        disclosed: bool,
        *,
        context: str = "",
        segment: str = "",
    ) -> None:
        """Record whether asking `attribute` produced a new requirement.

        `context` is where the question was asked - the retrieval route, in
        practice. A Buying turn and a Browsing turn are not asking the same
        question even when they name the same attribute, so their evidence is
        kept apart as well as pooled.
        """
        if not self.enabled or not attribute:
            return
        self._global[attribute] = self._global.get(attribute, YieldStat()).observe(
            disclosed
        )
        # Routes are a closed set, so the context table needs no bound; segment
        # signatures come from data and do.
        self._record(self._contexts, context, attribute, disclosed, bound=None)
        self._record(
            self._segments, segment, attribute, disclosed, bound=self.max_segments
        )

    @staticmethod
    def _record(
        table: dict[str, dict[str, YieldStat]],
        key: str,
        attribute: str,
        disclosed: bool,
        *,
        bound: int | None,
    ) -> None:
        if not key:
            return
        bucket = table.get(key)
        if bucket is None:
            if bound is not None and len(table) >= bound:
                return
            bucket = {}
            table[key] = bucket
        bucket[attribute] = bucket.get(attribute, YieldStat()).observe(disclosed)

    def likelihood(
        self, attribute: str, *, context: str = "", segment: str = ""
    ) -> float:
        """Posterior probability that asking `attribute` discloses something.

        The estimate is refined in order of how much evidence stands behind it:
        the pooled figure first, then the route, then the shopper segment. Each
        refinement only applies once its own bucket clears the floor, so a thin
        slice never overrides a well-supported one.
        """
        estimate = self.prior.get(attribute, 0.70)
        if not self.enabled:
            return estimate
        estimate = self._posterior(self._global.get(attribute), estimate)
        for table, key in ((self._contexts, context), (self._segments, segment)):
            bucket = table.get(key) if key else None
            stat = bucket.get(attribute) if bucket else None
            if stat is not None and stat.asked >= self.segment_floor:
                estimate = self._posterior(stat, estimate)
        return estimate

    def _posterior(self, stat: YieldStat | None, prior: float) -> float:
        if stat is None or stat.asked == 0:
            return prior
        strength = self.prior_strength
        return (prior * strength + stat.disclosed) / (strength + stat.asked)

    def snapshot(self) -> dict[str, object]:
        """Aggregate evidence for the ablation report; carries no session data."""
        return {
            "enabled": self.enabled,
            "prior_strength": self.prior_strength,
            "observations": sum(stat.asked for stat in self._global.values()),
            "attributes": {
                attribute: {
                    "asked": stat.asked,
                    "disclosed": stat.disclosed,
                    "observed_rate": round(stat.rate, 6),
                    "prior": self.prior.get(attribute, 0.70),
                    "posterior": round(self.likelihood(attribute), 6),
                }
                for attribute, stat in sorted(self._global.items())
            },
            "contexts": {
                key: {
                    attribute: {
                        "asked": stat.asked,
                        "disclosed": stat.disclosed,
                        "observed_rate": round(stat.rate, 6),
                    }
                    for attribute, stat in sorted(bucket.items())
                }
                for key, bucket in sorted(self._contexts.items())
            },
            "segment_count": len(self._segments),
        }
