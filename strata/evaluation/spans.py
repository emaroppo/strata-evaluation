"""What spans are, as the tasks over them read it.

Geometry, not scoring: how a region becomes entities, and when two ranges
meet. Every task over the span label type starts here, so they agree on
what an entity is even where they disagree on what counts as getting it
right.
"""

from collections.abc import Iterable

from strata.contracts import Span

#: One entity: a label over an end-exclusive character range.
Entity = tuple[str, int, int]


def entities(spans: Iterable[Span]) -> list[Entity]:
    """One entity per label a region carries.

    A region carrying two labels is two entities: reading it as one would
    score a half-right answer as right. See ``docs/adr/0035``.
    """
    return [(label, s.start, s.end) for s in spans for label in s.labels]


def overlaps(a: Entity, b: Entity) -> bool:
    """Whether two ranges share a character. Touching is not overlapping."""
    return a[1] < b[2] and b[1] < a[2]
