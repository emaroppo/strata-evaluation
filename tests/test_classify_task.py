"""The classify task, as arithmetic."""

import pytest

from strata.contracts import Choices
from strata.evaluation.tasks import classify


def c(*names):
    return Choices(values=list(names))


def test_the_task_names_the_label_type_it_reads():
    assert (classify.NAME, classify.LABEL_TYPE) == ("classify", "classification")


def test_a_perfect_classification_scores_one():
    scores = classify.score([c("cat"), c("dog")], [c("cat"), c("dog")])
    assert scores.exact_match == 1.0
    assert scores.micro.f1 == 1.0


def test_a_half_right_multi_label_answer_is_not_an_exact_match():
    """Why the metric is not called accuracy."""
    scores = classify.score([c("cat", "indoor")], [c("cat")])
    assert scores.exact_match == 0.0
    assert scores.micro.precision == 1.0
    assert scores.micro.recall == 0.5


def test_per_class_counts_what_each_class_was_asserted_and_guessed():
    scores = classify.score([c("cat"), c("cat"), c("cat"), c("dog")], [c("cat")] * 4)
    assert scores.per_class["cat"].precision == pytest.approx(0.75)
    assert scores.per_class["cat"].recall == 1.0
    assert scores.per_class["dog"].recall == 0.0
    assert scores.per_class["dog"].support == 1


def test_an_empty_side_scores_zero_rather_than_raising():
    assert classify.score([], []).exact_match == 0.0


def test_classification_needs_one_guess_per_answer():
    with pytest.raises(ValueError):
        classify.score([c("cat")], [])
