"""What the tasks over spans agree an entity is."""

from strata.contracts import Span
from strata.evaluation.spans import entities, overlaps


def test_a_region_with_two_labels_is_two_entities():
    assert entities([Span(labels=["PER", "ORG"], start=0, end=4)]) == [
        ("PER", 0, 4),
        ("ORG", 0, 4),
    ]


def test_a_region_with_no_labels_is_no_entity():
    assert entities([Span(labels=[], start=0, end=4)]) == []


def test_ranges_that_touch_do_not_overlap():
    """End-exclusive: "John" ending where " Smith" begins shares nothing."""
    assert not overlaps(("PER", 0, 4), ("PER", 4, 10))
    assert overlaps(("PER", 0, 5), ("PER", 4, 10))
