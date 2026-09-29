"""The entities task, as arithmetic.

The cases that matter are the ones where a plausible-looking
implementation is wrong.
"""

from strata.contracts import Span
from strata.evaluation import Tally
from strata.evaluation.tasks import entities


def s(label, start, end):
    return Span(labels=[label], start=start, end=end, text="x" * (end - start))


def test_the_task_names_the_label_type_it_reads():
    assert (entities.NAME, entities.LABEL_TYPE) == ("entities", "span")


def test_a_perfect_prediction_scores_one():
    truth = [[s("PER", 0, 5), s("ORG", 10, 15)]]
    assert entities.score(truth, truth).exact.f1 == 1.0


def test_finding_nothing_scores_zero_rather_than_raising():
    scores = entities.score([[s("PER", 0, 5)]], [[]])
    assert scores.exact.recall == 0.0
    assert scores.exact.f1 == 0.0


def test_an_empty_document_on_both_sides_is_not_a_failure():
    assert entities.score([[]], [[]]).exact == Tally()


def test_a_wrong_label_at_the_right_place_is_not_a_match():
    scores = entities.score([[s("PER", 0, 5)]], [[s("ORG", 0, 5)]])
    assert scores.exact.f1 == 0.0
    # and not by overlap either: the span is right, the answer is not
    assert scores.partial.f1 == 0.0


def test_exact_and_partial_disagree_on_a_boundary():
    """The case the two definitions exist for."""
    scores = entities.score([[s("PER", 0, 5)]], [[s("PER", 0, 6)]])
    assert scores.exact.f1 == 0.0
    assert scores.partial.f1 == 1.0


def test_one_greedy_span_cannot_match_every_entity():
    """Otherwise saying almost nothing scores almost perfectly."""
    truth = [[s("PER", 0, 5), s("PER", 10, 15), s("PER", 20, 25)]]
    scores = entities.score(truth, [[s("PER", 0, 25)]])
    assert scores.partial.tp == 1
    assert scores.partial.recall < 0.4


def test_per_class_scores_are_reported_separately():
    scores = entities.score([[s("PER", 0, 5), s("ORG", 10, 15)]], [[s("PER", 0, 5)]])
    assert scores.per_class["PER"].f1 == 1.0
    assert scores.per_class["ORG"].f1 == 0.0
    assert 0.0 < scores.exact.f1 < 1.0


def test_scores_run_over_many_documents():
    truth = [[s("PER", 0, 5)], [s("ORG", 0, 4)], []]
    predicted = [[s("PER", 0, 5)], [], [s("PER", 0, 3)]]
    # one hit, one miss, one invention
    assert entities.score(truth, predicted).exact == Tally(tp=1, fp=1, fn=1)


def test_a_half_found_two_label_region_is_one_of_two():
    truth = [[Span(labels=["PER", "ORG"], start=0, end=4)]]
    scores = entities.score(truth, [[Span(labels=["PER"], start=0, end=4)]])
    assert scores.exact.recall == 0.5
    assert scores.exact.precision == 1.0
    assert scores.per_class["ORG"].f1 == 0.0
