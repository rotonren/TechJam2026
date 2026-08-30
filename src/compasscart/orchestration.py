"""Per-turn strategy selection: what the pipeline should do differently now.

The pipeline runs the same shape every turn - route, retrieve, rank, rerank,
ask - regardless of whether the previous turn made progress. That is fine while
the conversation is converging and wasteful when it is not: a shopper who says
"those options are not quite right yet" has told us the approach is wrong, and
running the identical plan again cannot use that.

`StrategySelector` reads the distilled negative evidence on the session and
names what this turn should do instead:

- `open_probe` asks an open question when the structured question policy has
  nothing worth asking, so a turn that would otherwise return no question and
  collect no information still collects some.
- `exploit` stops asking once the candidate pool is small enough that a further
  question cannot pay for its turn.

A third strategy, `relax`, was implemented and removed. It demoted the weakest
hard constraint to a preference whenever the conversation stalled, on the
theory that a stall means the filters are too tight. Every configuration
measured worse, monotonically in how often it fired: `-0.003` at 72 firings,
`-0.010` at 146. A stall in this benchmark does not mean the filters are wrong,
it means the target is hard to find, and the hard constraints are the most
reliable evidence there is - they are quoted from the product itself. Removing
`relax` and keeping the other two moved the score from `-0.003` to `+0.018`,
partly because `relax` had been intercepting turns that `open_probe` should
have had: open questions rose from 2 to 11.

The lesson generalizes past this module. Adaptation conditioned on *what kind
of conversation this is* pays - route-conditioned reranking and
route-conditioned question priors are both worth about `+0.02`. Adaptation
conditioned on *whether the last turn failed* did not.

Selection is deterministic and ordered; there is no exploration and no
randomness, so a session replays identically. The selector is advisory - the
orchestrator applies the decision, and every strategy degrades to `probe`,
which is the pipeline's existing behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import SessionState

Strategy = Literal["probe", "open_probe", "exploit"]


@dataclass(frozen=True)
class StrategyDecision:
    """What this turn should do differently, and why."""

    name: Strategy = "probe"
    reason: str = "default"
    open_question: bool = False


class StrategySelector:
    def __init__(
        self,
        *,
        exploit_candidates: int = 10,
        enabled: bool = True,
    ) -> None:
        if exploit_candidates < 1:
            raise ValueError("exploit_candidates must be positive")
        self.exploit_candidates = exploit_candidates
        self.enabled = enabled

    def select(
        self, state: SessionState, *, structured_question: str | None
    ) -> StrategyDecision:
        """Name this turn's strategy from what previous turns achieved."""
        if not self.enabled:
            return StrategyDecision()

        # The structured policy declined to ask, which normally costs the whole
        # turn: the simulated shopper answers "ask me about one attribute" and
        # discloses nothing. An open question still collects something.
        if (
            structured_question is None
            and state.candidate_count > self.exploit_candidates
            and "other" not in state.asked_attributes
            and "other" not in state.no_preference_attributes
        ):
            return StrategyDecision(
                "open_probe",
                "no structured question had positive value",
                open_question=True,
            )

        if state.candidate_count and state.candidate_count <= self.exploit_candidates:
            return StrategyDecision("exploit", "candidate pool is already small")

        return StrategyDecision()
