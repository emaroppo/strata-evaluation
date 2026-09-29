"""The entities task: did the model find the entities, with the right label and bounds.

One task over the span label type, not the way spans are scored: another
task over the same spans can ask a different question of them.

*Exact* requires the label and both offsets to agree; *partial* accepts an
overlap of the same label, paired one-to-one. Partial shares the exact
counts' denominators: the same predictions and the same truth, scored by a
looser notion of a match. Both are kept because one understates and the
other flatters. See ``docs/adr/0035``.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from strata.contracts import Span

from ..counting import Tally
from ..spans import Entity, entities, overlaps

NAME = "entities"
#: The label type this task reads.
LABEL_TYPE = "span"


@dataclass(frozen=True)
class EntitiesScore:
    exact: Tally
    partial: Tally
    per_class: dict[str, Tally] = field(default_factory=dict)


def paired_overlaps(predicted: Sequence[Entity], truth: Sequence[Entity]) -> int:
    """Predicted entities overlapping a true one of the same label, paired off.

    One-to-one on purpose: otherwise one sentence-wide span scores against
    every entity inside it. See ``docs/adr/0035``.
    """
    unmatched = list(truth)
    matched = 0
    for guess in predicted:
        for i, wanted in enumerate(unmatched):
            if guess[0] == wanted[0] and overlaps(guess, wanted):
                del unmatched[i]
                matched += 1
                break
    return matched


def score(truths: Sequence[Iterable[Span]], predicted: Sequence[Iterable[Span]]) -> EntitiesScore:
    """Score parallel sequences of documents, each a collection of spans."""
    exact = Tally()
    partial_tp = 0
    per_class: dict[str, Tally] = {}

    for wanted_spans, got_spans in zip(truths, predicted, strict=True):
        wanted, got = entities(wanted_spans), entities(got_spans)
        wanted_keys, got_keys = set(wanted), set(got)
        exact += Tally.of(wanted_keys, got_keys)
        partial_tp += paired_overlaps(got, wanted)
        for label in {k[0] for k in wanted_keys | got_keys}:
            per_class[label] = per_class.get(label, Tally()) + Tally.of(
                {k for k in wanted_keys if k[0] == label},
                {k for k in got_keys if k[0] == label},
            )

    partial = Tally(
        partial_tp,
        (exact.tp + exact.fp) - partial_tp,
        (exact.tp + exact.fn) - partial_tp,
    )
    return EntitiesScore(exact=exact, partial=partial, per_class=dict(sorted(per_class.items())))
