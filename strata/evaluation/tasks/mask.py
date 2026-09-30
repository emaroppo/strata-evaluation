"""The mask task: is the anonymised document right?

One task over the span label type, beside ``entities`` and asking something
else of the same spans: not whether each entity was found with the right
bounds, but whether each was hidden, hidden once, under the right label,
and nothing else with it. Each way of getting that wrong is a failure mode
with a metric of its own (:mod:`strata.evaluation.failures`), and which
modes run is a parameter: first-party names, or a project's own
``failures.py:Class``. See ``docs/adr/0043``.

Every mode is aggregated the same way for its side, so this module knows
no mode by name. ``clean`` is derived from whichever modes ran: a gold
entity no gold-side mode flagged, touched only by predictions no
prediction-side mode flagged.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from strata.contracts import AnySchema, Span, Spans

from .. import failures as first_party
from ..alignment import Alignment, align
from ..failures import FailureMode
from ..identity import Identity
from ..registry import PluginError, resolve
from .base import Document, Scored, Task, TaskError

NAME = "mask"
LABEL_TYPE = "span"


class MaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Only these classes, on both sides; None is every class.
    classes: list[str] | None = None
    #: Which failure modes to run, by name or ``file.py:Class``; None is the
    #: first-party set. File references are anchored before they get here.
    failures: list[str] | None = None


def _spans(value: object) -> list[Span]:
    if not isinstance(value, Spans):
        raise TaskError(f"Task {NAME!r} scores spans, not {type(value).__name__}.")
    return list(value.values)


def modes(params: BaseModel) -> list[tuple[FailureMode, Identity]]:
    """The failure modes ``params`` names, resolved, in the order named."""
    refs = getattr(params, "failures", None) or list(first_party.DEFAULT)
    found = []
    for ref in refs:
        try:
            cls, identity = resolve("failure_mode", ref)
        except PluginError as e:
            raise TaskError(f"Task {NAME!r}: {e}") from None
        found.append((cls(), identity))
    names = [mode.name for mode, _ in found]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise TaskError(f"Task {NAME!r} is asked for {', '.join(twice)} more than once.")
    return found


class _Tally:
    """One mode's count and denominator, overall and per class."""

    def __init__(self) -> None:
        self.count: float = 0
        self.of: float = 0
        self.by_class: dict[str, list[float]] = defaultdict(lambda: [0, 0])

    def rate(self) -> float:
        return self.count / self.of if self.of else 0.0

    def add_items(self, flagged: set[int], items: Sequence[Any]) -> None:
        self.count += len(flagged)
        self.of += len(items)
        for i, item in enumerate(items):
            for label in item.labels:
                self.by_class[label][1] += 1
                self.by_class[label][0] += i in flagged

    def as_detail(self, mode: FailureMode, identity: Identity) -> dict[str, Any]:
        return {
            "identity": identity.model_dump(),
            "side": mode.side,
            "description": mode.description,
            "count": self.count,
            "denominator": self.of,
            "rate": self.rate(),
            "per_class": {
                label: {"count": c, "denominator": n, "rate": c / n if n else 0.0}
                for label, (c, n) in sorted(self.by_class.items())
            },
        }


class Mask(Task):
    """The anonymisation task, as a plugin: registered under ``mask``."""

    name = NAME
    version = "1"
    label_type = LABEL_TYPE
    needs_text = True
    Params = MaskParams

    def requires(self, params: BaseModel) -> list[tuple[str, str]]:
        refs = getattr(params, "failures", None) or list(first_party.DEFAULT)
        return [("failure_mode", ref) for ref in refs]

    def check(self, params: BaseModel, schema: AnySchema) -> None:
        super().check(params, schema)
        modes(params)

    def score(self, documents: Sequence[Document], params: BaseModel) -> Scored:
        active = modes(params)
        classes = getattr(params, "classes", None)
        tallies = {mode.name: _Tally() for mode, _ in active}
        clean = _Tally()

        for document in documents:
            if document.text is None:
                raise TaskError(f"Task {NAME!r} needs each document's text, and was not given it.")
            a = align(document.text, _spans(document.truth), _spans(document.prediction), classes)
            flagged_gold: set[int] = set()
            flagged_pred: set[int] = set()
            for mode, _ in active:
                detected = mode.detect(a)
                tally = tallies[mode.name]
                if mode.side in ("gold", "prediction") and isinstance(detected, set):
                    on_gold = mode.side == "gold"
                    tally.add_items(detected, a.gold if on_gold else a.predicted)
                    if on_gold:
                        flagged_gold |= detected
                    else:
                        flagged_pred |= detected
                elif mode.side == "characters" and isinstance(detected, tuple):
                    tally.count += detected[0]
                    tally.of += detected[1]
                elif mode.side == "document" and isinstance(detected, int | float):
                    tally.count += detected
                    tally.of += 1
                else:
                    raise TaskError(
                        f"Failure mode {mode.name!r} counts on the {mode.side} side and "
                        f"returned {type(detected).__name__}, which that side does not take."
                    )
            clean.add_items(_clean(a, flagged_gold, flagged_pred), a.gold)

        metrics = {name: tally.rate() for name, tally in tallies.items()}
        metrics["clean"] = clean.rate()
        counts: dict[str, int] = {"documents": len(documents)}
        for name, tally in [*tallies.items(), ("clean", clean)]:
            if isinstance(tally.count, int) and isinstance(tally.of, int):
                counts[name] = tally.count
                counts[f"{name}.of"] = tally.of
        per_class: dict[str, dict[str, float]] = defaultdict(dict)
        for name, tally in [*tallies.items(), ("clean", clean)]:
            for label, (c, n) in tally.by_class.items():
                per_class[label][name] = c / n if n else 0.0
        return Scored(
            metrics=metrics,
            counts=counts,
            per_class=dict(sorted(per_class.items())),
            detail={
                "failures": {
                    mode.name: tallies[mode.name].as_detail(mode, identity)
                    for mode, identity in active
                }
            },
        )


def _clean(a: Alignment, flagged_gold: set[int], flagged_pred: set[int]) -> set[int]:
    return {
        g
        for g, ps in enumerate(a.gold_to_pred)
        if ps and g not in flagged_gold and not (ps & flagged_pred)
    }
