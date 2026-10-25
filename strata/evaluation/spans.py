"""What spans are, as every task over them reads it: geometry, not scoring."""

from collections.abc import Iterable

from strata.contracts import Span

#: One entity: a label over an end-exclusive character range.
Entity = tuple[str, int, int]


def entities(spans: Iterable[Span]) -> list[Entity]:
    """One entity per label a region carries. See ``docs/adr/0035``."""
    return [(label, s.start, s.end) for s in spans for label in s.labels]


def overlaps(a: Entity, b: Entity) -> bool:
    """Whether two ranges share a character. Touching is not overlapping."""
    return a[1] < b[2] and b[1] < a[2]
