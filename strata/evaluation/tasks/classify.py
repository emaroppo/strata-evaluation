"""The classify task: which classes a sample is, scored as sets.

``exact`` counts samples whose asserted set is exactly the model's. It is not
called accuracy: a label set asserting two classes per sample, scored by a
model asserting one, has an exact match of zero. ``micro`` is over every
class assertion. See ``docs/adr/0035``.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from strata.contracts import Choices

from ..counting import Tally

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
