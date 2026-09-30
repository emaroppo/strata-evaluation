"""Inline span markup for examples, and reading a document the way offsets were made."""

import pytest

from strata.evaluation.markup import MarkupError, pair, parse
from strata.evaluation.text import UnreadableDocument, read_document


def test_markup_strips_to_the_text_with_offsets_into_it():
    text, spans = parse("Dear [John Smith|PER], from [Acme|ORG].")
    assert text == "Dear John Smith, from Acme."
    assert [(s.labels, text[s.start : s.end]) for s in spans] == [
        (["PER"], "John Smith"),
        (["ORG"], "Acme"),
    ]


def test_markup_nests_for_overlapping_spans():
    text, spans = parse("[John [Smith|PER]|PER]")
    assert text == "John Smith"
    assert [(s.start, s.end) for s in spans] == [(0, 10), (5, 10)]


def test_a_region_can_carry_two_labels():
    _, spans = parse("[Acme|ORG,LOC]")
    assert spans[0].labels == ["ORG", "LOC"]


@pytest.mark.parametrize("bad", ["[open", "closed]", "no|label", "[x|PER"])
def test_markup_that_does_not_say_what_it_means_is_refused(bad):
    with pytest.raises(MarkupError):
        parse(bad)


def test_a_pair_must_mark_one_text():
    with pytest.raises(MarkupError, match="different texts"):
        pair("[John|PER] Smith", "[Jon|PER] Smith")


def test_a_document_is_read_as_utf8_with_bad_bytes_replaced(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes("Café ".encode() + b"\xff")
    assert read_document(path) == "Café �"


def test_an_unreadable_document_is_refused_not_read_as_empty(tmp_path):
    with pytest.raises(UnreadableDocument, match="Cannot read"):
        read_document(tmp_path / "missing.txt")
