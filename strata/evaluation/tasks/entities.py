"""The entities task: did the model find the entities, with the right label and bounds.

*Exact* needs label and both offsets to agree; *partial* accepts a same-label
overlap, paired one-to-one, over the same denominators. See ``docs/adr/0035``.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from strata.contracts import Span, Spans

from ..counting import Tally
from ..spans import Entity, entities, overlaps
from .base import Document, Scored, Task, TaskError

NAME = "entities"
#: The label type this task reads.
LABEL_TYPE = "span"


@dataclass(frozen=True)
class EntitiesScore:
    exact: Tally
    partial: Tally
    per_class: dict[str, Tally] = field(default_factory=dict)
    #: The partial notion of a match, per class, over the same denominators.
    per_class_partial: dict[str, Tally] = field(default_factory=dict)


def paired_overlaps(predicted: Sequence[Entity], truth: Sequence[Entity]) -> int:
    """Predicted entities overlapping a true one of the same label, paired one-to-one. See
    ``docs/adr/0035``.
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
    partial_tp_of: dict[str, int] = {}

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
            partial_tp_of[label] = partial_tp_of.get(label, 0) + paired_overlaps(
                [k for k in got if k[0] == label], [k for k in wanted if k[0] == label]
            )

    partial = _loosened(exact, partial_tp)
    per_class_partial = {
        label: _loosened(t, partial_tp_of.get(label, 0)) for label, t in per_class.items()
    }
    return EntitiesScore(
        exact=exact,
        partial=partial,
        per_class=dict(sorted(per_class.items())),
        per_class_partial=dict(sorted(per_class_partial.items())),
    )


def _loosened(exact: Tally, partial_tp: int) -> Tally:
    """The exact tally's predictions and truth, matched by overlap instead."""
    return Tally(partial_tp, (exact.tp + exact.fp) - partial_tp, (exact.tp + exact.fn) - partial_tp)


class EntitiesParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Only these classes, on both sides; None is every class.
    classes: list[str] | None = None


def _spans(value: object) -> list[Span]:
    if not isinstance(value, Spans):
        raise TaskError(f"Task {NAME!r} scores spans, not {type(value).__name__}.")
    return list(value.values)


def _only(spans: list[Span], classes: list[str] | None) -> list[Span]:
    """The spans with their labels cut to ``classes``, and any left with none dropped."""
    if classes is None:
        return spans
    kept = []
    for span in spans:
        labels = [label for label in span.labels if label in classes]
        if labels:
            kept.append(span.model_copy(update={"labels": labels}))
    return kept


class Entities(Task):
    """The entities task, as a plugin: registered under ``entities``."""

    name = NAME
    # 2: per-class scores carry the partial notion of a match too
    version = "2"
    label_type = LABEL_TYPE
    Params = EntitiesParams

    def score(self, documents: Sequence[Document], params: BaseModel) -> Scored:
        classes = getattr(params, "classes", None)
        scores = score(
            [_only(_spans(d.truth), classes) for d in documents],
            [_only(_spans(d.prediction), classes) for d in documents],
        )
        exact, partial = scores.exact, scores.partial
        return Scored(
            metrics={
                "precision": exact.precision,
                "recall": exact.recall,
                "f1": exact.f1,
                "partial_precision": partial.precision,
                "partial_recall": partial.recall,
                "partial_f1": partial.f1,
            },
            counts={
                "documents": len(documents),
                "tp": exact.tp,
                "fp": exact.fp,
                "fn": exact.fn,
                "partial_tp": partial.tp,
            },
            per_class={
                label: {
                    "precision": t.precision,
                    "recall": t.recall,
                    "f1": t.f1,
                    "partial_precision": scores.per_class_partial[label].precision,
                    "partial_recall": scores.per_class_partial[label].recall,
                    "partial_f1": scores.per_class_partial[label].f1,
                    "support": t.support,
                }
                for label, t in scores.per_class.items()
            },
        )
