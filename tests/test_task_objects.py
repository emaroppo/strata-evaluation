"""Tasks as plugins: the same numbers as their functions, with identity and parameters."""

import pytest

from strata.contracts import Choices, ClassificationSchema, Span, Spans, SpanSchema
from strata.evaluation.identity import Identity
from strata.evaluation.tasks import Classify, Document, Entities, TaskError, classify, entities

ID = Identity(name="t", version="1", source="test==0")


def s(label, start, end):
    return Span(labels=[label], start=start, end=end)


def spans(*values):
    return Spans(values=list(values))


def test_classify_records_what_its_function_computes():
    truths = [Choices(values=["cat"]), Choices(values=["cat", "indoor"])]
    guesses = [Choices(values=["cat"]), Choices(values=["cat"])]
    task = Classify()
    recorded = task.recorded(
        [Document(t, g) for t, g in zip(truths, guesses, strict=True)], task.params(), ID
    )
    direct = classify.score(truths, guesses)
    assert recorded.metrics["exact_match"] == direct.exact_match == 0.5
    assert recorded.counts["fn"] == direct.micro.fn == 1
    assert recorded.per_class["indoor"]["support"] == 1
    assert recorded.identity == ID


def test_entities_records_what_its_function_computes():
    truth, guess = spans(s("PER", 0, 5)), spans(s("PER", 0, 6))
    task = Entities()
    recorded = task.recorded([Document(truth, guess)], task.params(), ID)
    direct = entities.score([truth.values], [guess.values])
    assert recorded.metrics["f1"] == direct.exact.f1 == 0.0
    assert recorded.metrics["partial_f1"] == direct.partial.f1 == 1.0


def test_entities_can_be_asked_for_some_classes_only():
    truth = spans(s("PER", 0, 5), s("DATE", 10, 15))
    guess = spans(s("PER", 0, 5))
    task = Entities()
    everything = task.score([Document(truth, guess)], task.params())
    people = task.score([Document(truth, guess)], task.params({"classes": ["PER"]}))
    assert everything.metrics["recall"] == 0.5
    assert people.metrics["recall"] == 1.0
    assert set(people.per_class) == {"PER"}


def test_a_region_keeps_only_the_asked_labels():
    truth = spans(Span(labels=["PER", "ORG"], start=0, end=4))
    guess = spans(s("PER", 0, 4))
    task = Entities()
    scored = task.score([Document(truth, guess)], task.params({"classes": ["PER"]}))
    assert scored.metrics["recall"] == 1.0


def test_a_task_refuses_a_parameter_it_does_not_take():
    with pytest.raises(TaskError, match="will not take"):
        Entities().params({"clases": ["PER"]})
    with pytest.raises(TaskError, match="will not take"):
        Classify().params({"anything": 1})


def test_a_task_refuses_a_class_the_label_set_does_not_have():
    task = Entities()
    with pytest.raises(TaskError, match="asked for LOC"):
        task.check(task.params({"classes": ["LOC"]}), SpanSchema(classes=["PER"]))


def test_a_task_refuses_a_label_set_of_another_type():
    task = Entities()
    with pytest.raises(TaskError, match="reads span label sets"):
        task.check(task.params(), ClassificationSchema(classes=["cat"]))


def test_a_task_refuses_a_value_of_the_wrong_kind():
    task = Classify()
    with pytest.raises(TaskError, match="scores choices"):
        task.score([Document(spans(), spans())], task.params())


@pytest.mark.parametrize("task", [Classify(), Entities()])
def test_empty_input_scores_without_raising(task):
    assert task.score([], task.params()).metrics
