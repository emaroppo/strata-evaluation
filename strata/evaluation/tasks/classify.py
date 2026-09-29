"""The classify task: which classes a sample is, scored as sets.

``exact`` counts samples whose asserted set is exactly the model's. It is not
called accuracy: a label set asserting two classes per sample, scored by a
model asserting one, has an exact match of zero. ``micro`` is over every
class assertion. See ``docs/adr/0035``.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from strata.contracts import Choices

from ..counting import Tally
from .base import Document, Scored, Task, TaskError

NAME = "classify"
#: The label type this task reads.
LABEL_TYPE = "classification"


@dataclass(frozen=True)
class ClassifyScore:
    samples: int
    exact: int
    micro: Tally
    per_class: dict[str, Tally] = field(default_factory=dict)

    @property
    def exact_match(self) -> float:
        return self.exact / self.samples if self.samples else 0.0


def score(truths: Sequence[Choices], guesses: Sequence[Choices]) -> ClassifyScore:
    """Score parallel sequences of answers and guesses, one per sample."""
    exact = 0
    per_class: dict[str, Tally] = {}
    for truth, guess in zip(truths, guesses, strict=True):
        wanted, got = set(truth.values), set(guess.values)
        exact += wanted == got
        for name in wanted | got:
            per_class[name] = per_class.get(name, Tally()) + Tally.of({name} & wanted, {name} & got)
    return ClassifyScore(
        samples=len(truths),
        exact=exact,
        micro=sum(per_class.values(), Tally()),
        per_class=dict(sorted(per_class.items())),
    )


def _choices(value: object) -> Choices:
    if not isinstance(value, Choices):
        raise TaskError(f"Task {NAME!r} scores choices, not {type(value).__name__}.")
    return value


class Classify(Task):
    """The classify task, as a plugin: registered under ``classify``."""

    name = NAME
    version = "1"
    label_type = LABEL_TYPE

    def score(self, documents: Sequence[Document], params: BaseModel) -> Scored:
        scores = score(
            [_choices(d.truth) for d in documents], [_choices(d.prediction) for d in documents]
        )
        micro = scores.micro
        return Scored(
            metrics={
                "exact_match": scores.exact_match,
                "precision": micro.precision,
                "recall": micro.recall,
                "f1": micro.f1,
            },
            counts={
                "samples": scores.samples,
                "exact": scores.exact,
                "tp": micro.tp,
                "fp": micro.fp,
                "fn": micro.fn,
            },
            per_class={
                name: {
                    "precision": t.precision,
                    "recall": t.recall,
                    "f1": t.f1,
                    "support": t.support,
                }
                for name, t in scores.per_class.items()
            },
        )
