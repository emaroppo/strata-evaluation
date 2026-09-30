"""One document's answer and guess, laid over each other character by character.

Computed once per document and only read by failure modes, so every mode
agrees on what overlapping, covering and spilling mean, and a new mode
rarely needs new geometry. See ``docs/adr/0042``.

"Characters" are the non-whitespace characters of the text: the space
between "John" and "Smith" is never a leak, a spill or a bridge. Two spans
overlap when they share one. A region is one item whatever labels it
carries, and labels agree when they share one.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from strata.contracts import Span

#: What a character a prediction covers outside every gold entity is, by
#: where it sits. Ranked: a character two predictions disagree about takes
#: the one attached to more entities.
Attribution = Literal["bridge", "spill", "spurious"]
_RANK: dict[str, int] = {"bridge": 2, "spill": 1, "spurious": 0}


@dataclass(frozen=True)
class Item:
    """A span as a set of characters, with the labels it carries."""

    labels: frozenset[str]
    start: int
    end: int
    chars: frozenset[int]


@dataclass(frozen=True)
class Alignment:
    text: str
    gold: list[Item]
    predicted: list[Item]
    #: For each gold entity, the predictions sharing a character with it.
    gold_to_pred: list[frozenset[int]]
    #: For each prediction, the gold entities it shares a character with.
    pred_to_gold: list[frozenset[int]]
    #: For each prediction, the other predictions it shares a character with.
    pred_to_pred: list[frozenset[int]]
    #: Every character some prediction covers.
    covered: frozenset[int]
    #: Every character inside some gold entity.
    gold_chars: frozenset[int]
    #: Each covered character outside every gold entity, by where it sits.
    attribution: dict[int, Attribution] = field(default_factory=dict)

    def pieces(self, predictions: Iterable[int]) -> list[set[int]]:
        """The predictions grouped by overlap: each group reads as one placeholder."""
        remaining = set(predictions)
        groups = []
        while remaining:
            seed = remaining.pop()
            group, frontier = {seed}, [seed]
            while frontier:
                for other in self.pred_to_pred[frontier.pop()] & remaining:
                    remaining.discard(other)
                    group.add(other)
                    frontier.append(other)
            groups.append(group)
        return groups

    def spill(self, p: int) -> set[int]:
        """Characters of prediction ``p`` beyond the outermost gold entities it overlaps."""
        touched = self.pred_to_gold[p]
        if not touched:
            return set()
        lo = min(min(self.gold[g].chars) for g in touched)
        hi = max(max(self.gold[g].chars) for g in touched)
        return {c for c in self.predicted[p].chars - self.gold_chars if c < lo or c > hi}


def items(text: str, spans: Sequence[Span], classes: Iterable[str] | None = None) -> list[Item]:
    """``spans`` as items, cut to ``classes`` when given.

    A span over whitespace alone keeps its whole range as its characters,
    so it still takes part rather than vanishing from every count.
    """
    wanted = set(classes) if classes is not None else None
    out = []
    for span in spans:
        labels = frozenset(span.labels if wanted is None else set(span.labels) & wanted)
        if not labels:
            continue
        chars = frozenset(
            i for i in range(span.start, min(span.end, len(text))) if not text[i].isspace()
        ) or frozenset(range(span.start, span.end))
        out.append(Item(labels=labels, start=span.start, end=span.end, chars=chars))
    return out


def align(
    text: str,
    gold: Sequence[Span],
    predicted: Sequence[Span],
    classes: Iterable[str] | None = None,
) -> Alignment:
    classes = list(classes) if classes is not None else None
    truth, guess = items(text, gold, classes), items(text, predicted, classes)
    gold_to_pred = [frozenset(p for p, q in enumerate(guess) if g.chars & q.chars) for g in truth]
    pred_to_gold = [frozenset(g for g, t in enumerate(truth) if t.chars & q.chars) for q in guess]
    pred_to_pred = [
        frozenset(o for o, other in enumerate(guess) if o != p and other.chars & q.chars)
        for p, q in enumerate(guess)
    ]
    covered = frozenset().union(*(q.chars for q in guess))
    gold_chars = frozenset().union(*(t.chars for t in truth))

    attribution: dict[int, Attribution] = {}
    for p, q in enumerate(guess):
        touched = pred_to_gold[p]
        lo = hi = 0
        if touched:
            lo = min(min(truth[g].chars) for g in touched)
            hi = max(max(truth[g].chars) for g in touched)
        for c in q.chars - gold_chars:
            kind: Attribution = "spurious" if not touched else "bridge" if lo < c < hi else "spill"
            if c not in attribution or _RANK[kind] > _RANK[attribution[c]]:
                attribution[c] = kind

    return Alignment(
        text=text,
        gold=truth,
        predicted=guess,
        gold_to_pred=gold_to_pred,
        pred_to_gold=pred_to_gold,
        pred_to_pred=pred_to_pred,
        covered=covered,
        gold_chars=gold_chars,
        attribution=attribution,
    )
