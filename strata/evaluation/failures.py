"""The ways a span prediction can be wrong for anonymisation, one metric each.

A failure mode is a small class: a name, a version, the side it counts on,
a one-line description, a worked example, and :meth:`FailureMode.detect`
over an :class:`~strata.evaluation.alignment.Alignment`. The ``mask`` task
aggregates whatever modes it is given, the same way for every mode of a
side, so adding one is one class, and a project can keep its own in a file
of its own (``failures.py:Class``). See ``docs/adr/0042``.

Each mode's example is the case it exists for, and the conformance suite
holds it to raising that mode and no other *isolated* mode outside its
``co_occurs``: a mode that overlaps another fails, naming both.
Aggregates, which summarise several failures in one number, are not
isolated and are not held to it.

Sides, and what :meth:`detect` returns for each:

- ``gold``: the gold entities it flags, as indices. Rated over gold entities.
- ``prediction``: the predictions it flags, as indices. Over predictions.
- ``characters``: a numerator and a denominator of characters.
- ``document``: one number for the document, averaged over documents.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from .alignment import Alignment

Side = Literal["gold", "prediction", "characters", "document"]
SIDES: tuple[Side, ...] = ("gold", "prediction", "characters", "document")

Detected = set[int] | tuple[int, int] | float


class FailureMode(ABC):
    """Subclass this, name it, give it an example, and implement :meth:`detect`."""

    name: ClassVar[str]
    #: Bumped when a change alters what an unchanged document detects.
    version: ClassVar[str]
    side: ClassVar[Side]
    description: ClassVar[str]
    #: The case this mode exists for, as ``(gold, predicted)`` markup over one
    #: text: ``("Dear [John Smith|PER]", "Dear [John|PER] [Smith|PER]")``.
    example: ClassVar[tuple[str, str]]
    #: Isolated modes the example may raise as well, because the case is both.
    co_occurs: ClassVar[tuple[str, ...]] = ()
    #: False for an aggregate: a number summarising several failures, which
    #: another mode's example is expected to move.
    isolated: ClassVar[bool] = True

    @abstractmethod
    def detect(self, a: Alignment) -> Detected: ...

    @classmethod
    def raised(cls, detected: Detected) -> bool:
        """Whether a detection says this failure happened at all."""
        if isinstance(detected, set):
            return bool(detected)
        if isinstance(detected, tuple):
            return detected[0] > 0
        return detected > 0


# ----------------------------------------------------------------------
# what happened to each gold entity
# ----------------------------------------------------------------------


class Missed(FailureMode):
    name = "missed"
    version = "1"
    side = "gold"
    description = "no prediction touches the entity: all of it leaks"
    example = ("Dear [John Smith|PER],", "Dear John Smith,")

    def detect(self, a: Alignment) -> set[int]:
        return {g for g, ps in enumerate(a.gold_to_pred) if not ps}


class Truncated(FailureMode):
    name = "truncated"
    version = "1"
    side = "gold"
    description = "the entity is touched, but some of it is left uncovered: part of it leaks"
    example = ("Dear [John Smith|PER],", "Dear [John|PER] Smith,")

    def detect(self, a: Alignment) -> set[int]:
        return {g for g, ps in enumerate(a.gold_to_pred) if ps and a.gold[g].chars - a.covered}


class Fragmented(FailureMode):
    name = "fragmented"
    version = "1"
    side = "gold"
    description = "the entity is split across predictions that do not overlap: several placeholders"
    example = ("Dear [John Smith|PER],", "Dear [John|PER] [Smith|PER],")

    def detect(self, a: Alignment) -> set[int]:
        return {g for g, ps in enumerate(a.gold_to_pred) if len(a.pieces(ps)) >= 2}


class Mislabelled(FailureMode):
    name = "mislabelled"
    version = "1"
    side = "gold"
    description = (
        "a prediction touching the entity carries none of its labels: the wrong placeholder"
    )
    example = ("Dear [John Smith|PER],", "Dear [John Smith|ORG],")

    def detect(self, a: Alignment) -> set[int]:
        return {
            g
            for g, ps in enumerate(a.gold_to_pred)
            if any(not (a.predicted[p].labels & a.gold[g].labels) for p in ps)
        }


class Shared(FailureMode):
    name = "shared"
    version = "1"
    side = "gold"
    description = "the entity is inside a prediction that also covers another entity"
    example = ("Dear [John Smith|PER] and [Anne|PER],", "Dear [John Smith and Anne|PER],")
    co_occurs = ("merging",)

    def detect(self, a: Alignment) -> set[int]:
        return {
            g for g, ps in enumerate(a.gold_to_pred) if any(len(a.pred_to_gold[p]) >= 2 for p in ps)
        }


class Leaked(FailureMode):
    name = "leaked"
    version = "1"
    side = "gold"
    description = "some character of the entity is left uncovered: missed or truncated"
    example = ("Dear [John Smith|PER],", "Dear [John|PER] Smith,")
    isolated = False

    def detect(self, a: Alignment) -> set[int]:
        return {g for g, t in enumerate(a.gold) if t.chars - a.covered}


# ----------------------------------------------------------------------
# what each prediction did
# ----------------------------------------------------------------------


class Spurious(FailureMode):
    name = "spurious"
    version = "1"
    side = "prediction"
    description = "the prediction touches no entity: it masks text that is not one"
    example = ("Dear [John Smith|PER],", "[Dear|PER] John Smith,")
    co_occurs = ("missed",)

    def detect(self, a: Alignment) -> set[int]:
        return {p for p, gs in enumerate(a.pred_to_gold) if not gs}


class Spilled(FailureMode):
    name = "spilled"
    version = "1"
    side = "prediction"
    description = "the prediction runs past the first or last entity it touches"
    example = ("Dear [John Smith|PER],", "[Dear John Smith|PER],")

    def detect(self, a: Alignment) -> set[int]:
        return {p for p in range(len(a.predicted)) if a.spill(p)}


class Merging(FailureMode):
    name = "merging"
    version = "1"
    side = "prediction"
    description = "the prediction covers two or more entities: one placeholder for several"
    example = ("Dear [John Smith|PER] and [Anne|PER],", "Dear [John Smith and Anne|PER],")
    co_occurs = ("shared",)

    def detect(self, a: Alignment) -> set[int]:
        return {p for p, gs in enumerate(a.pred_to_gold) if len(gs) >= 2}


class Duplicate(FailureMode):
    name = "duplicate"
    version = "1"
    side = "prediction"
    description = "the prediction overlaps another prediction"
    example = ("Dear [John Smith|PER],", "Dear [John [Smith|PER]|PER],")

    def detect(self, a: Alignment) -> set[int]:
        return {p for p, others in enumerate(a.pred_to_pred) if others}


# ----------------------------------------------------------------------
# characters
# ----------------------------------------------------------------------


class CharRecall(FailureMode):
    name = "char_recall"
    version = "1"
    side = "characters"
    description = "entity characters covered by some prediction, of all entity characters"
    example = ("Dear [John Smith|PER],", "Dear [John|PER] Smith,")
    isolated = False

    def detect(self, a: Alignment) -> tuple[int, int]:
        return len(a.gold_chars & a.covered), len(a.gold_chars)


class CharPrecision(FailureMode):
    name = "char_precision"
    version = "1"
    side = "characters"
    description = "covered characters inside some entity, of all covered characters"
    example = ("Dear [John Smith|PER],", "[Dear John Smith|PER],")
    isolated = False

    def detect(self, a: Alignment) -> tuple[int, int]:
        return len(a.covered & a.gold_chars), len(a.covered)


class _Attributed(FailureMode):
    """Covered characters outside every entity, of one attribution, of all covered."""

    attribution: ClassVar[str]
    side = "characters"
    isolated = False

    def detect(self, a: Alignment) -> tuple[int, int]:
        mine = sum(1 for kind in a.attribution.values() if kind == self.attribution)
        return mine, len(a.covered)


class SpuriousChars(_Attributed):
    name = "spurious_chars"
    version = "1"
    attribution = "spurious"
    description = "covered characters in predictions touching no entity, of all covered"
    example = ("Dear [John Smith|PER],", "[Dear|PER] John Smith,")


class SpilledChars(_Attributed):
    name = "spilled_chars"
    version = "1"
    attribution = "spill"
    description = "covered characters beyond the entities a prediction touches, of all covered"
    example = ("Dear [John Smith|PER],", "[Dear John Smith|PER],")


class BridgeChars(_Attributed):
    name = "bridge_chars"
    version = "1"
    attribution = "bridge"
    description = "covered characters between entities one prediction merges, of all covered"
    example = ("Dear [John Smith|PER] and [Anne|PER],", "Dear [John Smith and Anne|PER],")


# ----------------------------------------------------------------------
# the document
# ----------------------------------------------------------------------


class PlaceholderCountError(FailureMode):
    name = "placeholder_count_error"
    version = "1"
    side = "document"
    description = (
        "per class, how far the placeholders the prediction renders are from the "
        "entities there are, summed"
    )
    example = ("Dear [John Smith|PER],", "Dear [John|PER] [Smith|PER],")
    isolated = False

    def detect(self, a: Alignment) -> float:
        labels = set().union(*(t.labels for t in a.gold), *(q.labels for q in a.predicted))
        error = 0
        for label in labels:
            wanted = sum(1 for t in a.gold if label in t.labels)
            rendered = len(a.pieces(p for p, q in enumerate(a.predicted) if label in q.labels))
            error += abs(rendered - wanted)
        return float(error)


#: The first-party modes, as a file names them. Registered under
#: ``strata.failure_modes``; this is what ``mask`` runs when not told.
DEFAULT: tuple[str, ...] = (
    "missed",
    "truncated",
    "fragmented",
    "mislabelled",
    "shared",
    "spurious",
    "spilled",
    "merging",
    "duplicate",
    "leaked",
    "char_recall",
    "char_precision",
    "spurious_chars",
    "spilled_chars",
    "bridge_chars",
    "placeholder_count_error",
)
