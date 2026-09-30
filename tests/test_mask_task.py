"""The mask task: one metric per failure mode, and each failure raising only its own.

Scored through the task, so what is checked is what a record would say. The
cases are the ones where a plausible implementation is wrong: a split name
that leaks nothing, a merge whose middle is bridge rather than spill, a
duplicate that is not a fragment.
"""

import pytest

from strata.contracts import Spans, SpanSchema
from strata.evaluation.markup import pair
from strata.evaluation.tasks import Document, Mask, TaskError

#: The isolated failure modes, which no other failure's example may raise.
ISOLATED = [
    "missed",
    "truncated",
    "fragmented",
    "mislabelled",
    "shared",
    "spurious",
    "spilled",
    "merging",
    "duplicate",
]


def score(gold: str, predicted: str, **params):
    text, truth, guess = pair(gold, predicted)
    task = Mask()
    return task.score(
        [Document(Spans(values=truth), Spans(values=guess), text)], task.params(params)
    )


def raised(scored) -> set[str]:
    return {name for name in ISOLATED if scored.metrics[name] > 0}


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        ("Dear [John Smith|PER],", "Dear John Smith,", {"missed"}),
        ("Dear [John Smith|PER],", "Dear [John|PER] Smith,", {"truncated"}),
        ("Dear [John Smith|PER],", "Dear [John|PER] [Smith|PER],", {"fragmented"}),
        ("Dear [John Smith|PER],", "Dear [John Smith|ORG],", {"mislabelled"}),
        ("Dear [John Smith|PER],", "[Dear|PER] John Smith,", {"spurious", "missed"}),
        ("Dear [John Smith|PER],", "[Dear John Smith|PER],", {"spilled"}),
        (
            "Dear [John Smith|PER] and [Anne|PER],",
            "Dear [John Smith and Anne|PER],",
            {"merging", "shared"},
        ),
        (
            "Dear [John Smith|PER] and [Anne|PER],",
            "[Dear John Smith and Anne|PER],",
            {"merging", "shared", "spilled"},
        ),
        ("Dear [John Smith|PER],", "Dear [John [Smith|PER]|PER],", {"duplicate"}),
        ("Dear [John Smith|PER],", "Dear [John Smith|PER],", set()),
    ],
)
def test_each_failure_raises_its_own_metric_and_no_other(gold, predicted, expected):
    assert raised(score(gold, predicted)) == expected


def test_a_split_name_leaks_nothing_but_renders_two_placeholders():
    """The case the metrics exist for: "Dear [PER] [PER]"."""
    scored = score("Dear [John Smith|PER],", "Dear [John|PER] [Smith|PER],")
    assert scored.metrics["fragmented"] == 1.0
    assert scored.metrics["leaked"] == 0.0
    assert scored.metrics["char_recall"] == 1.0
    assert scored.metrics["placeholder_count_error"] == 1.0
    assert scored.metrics["clean"] == 0.0


def test_the_words_between_merged_entities_are_bridge_not_spill():
    scored = score("Dear [John Smith|PER] and [Anne|PER],", "Dear [John Smith and Anne|PER],")
    assert scored.metrics["bridge_chars"] > 0
    assert scored.metrics["spilled_chars"] == 0.0
    both = score("Dear [John Smith|PER] and [Anne|PER],", "[Dear John Smith and Anne|PER],")
    assert both.metrics["bridge_chars"] > 0 and both.metrics["spilled_chars"] > 0


@pytest.mark.parametrize(
    ("gold", "predicted"),
    [
        ("Dear [John Smith|PER],", "[Dear|PER] John Smith,"),
        ("Dear [John Smith|PER],", "[Dear John Smith|PER],"),
        ("Dear [John Smith|PER] and [Anne|PER],", "[Dear John Smith and Anne|PER],"),
        ("Dear [John Smith|PER],", "Dear [John [Smith|PER]|PER],"),
    ],
)
def test_the_over_masked_characters_are_all_attributed(gold, predicted):
    m = score(gold, predicted).metrics
    assert m["spurious_chars"] + m["spilled_chars"] + m["bridge_chars"] == pytest.approx(
        1 - m["char_precision"]
    )


def test_a_fragment_and_a_miss_do_not_cancel_in_the_placeholder_count():
    """Two people rendered as two placeholders, but one of them is the wrong person."""
    scored = score(
        "Dear [John Smith|PER], from [Acme|ORG].", "Dear [John|PER] [Smith|PER], from Acme."
    )
    assert scored.metrics["placeholder_count_error"] == 2.0


def test_only_an_exact_single_prediction_is_clean():
    scored = score("Dear [John Smith|PER] and [Anne|PER],", "Dear [John Smith|PER] and [An|PER]ne,")
    assert scored.metrics["clean"] == 0.5
    assert scored.counts["clean"] == 1 and scored.counts["clean.of"] == 2


def test_rates_are_counts_over_their_denominators():
    scored = score("[A|PER] [B|PER] [C|ORG]", "[A|PER] B [C|PER]")
    assert scored.counts["missed"] == 1 and scored.counts["missed.of"] == 3
    assert scored.metrics["missed"] == pytest.approx(1 / 3)
    assert scored.per_class["PER"]["missed"] == 0.5
    assert scored.per_class["ORG"]["mislabelled"] == 1.0


def test_classes_restrict_both_sides():
    scored = score(
        "Dear [John|PER] on [Monday|DATE],", "Dear [John|PER] on Monday,", classes=["PER"]
    )
    assert scored.metrics["missed"] == 0.0
    assert set(scored.per_class) == {"PER"}


def test_nothing_to_score_scores_zeros():
    task = Mask()
    scored = task.score([], task.params())
    assert all(value == 0.0 for value in scored.metrics.values())


def test_the_modes_run_are_a_parameter():
    scored = score("Dear [John Smith|PER],", "Dear [John|PER] Smith,", failures=["truncated"])
    assert set(scored.metrics) == {"truncated", "clean"}
    assert scored.detail["failures"]["truncated"]["identity"]["name"] == "truncated"


def test_a_projects_own_mode_runs_beside_the_first_party_ones(tmp_path):
    (tmp_path / "failures.py").write_text(
        "from strata.evaluation.failures import FailureMode\n\n\n"
        "class Initials(FailureMode):\n"
        '    name = "initials"\n'
        '    version = "1"\n'
        '    side = "prediction"\n'
        '    description = "a prediction of two characters or fewer"\n'
        '    example = ("[J S|PER]", "[J|PER] S")\n'
        "    isolated = False\n\n"
        "    def detect(self, a):\n"
        "        return {p for p, q in enumerate(a.predicted) if len(q.chars) <= 2}\n"
    )
    ref = f"{tmp_path / 'failures.py'}:Initials"
    scored = score("Dear [J Smith|PER],", "Dear [J|PER] [Smith|PER],", failures=["fragmented", ref])
    assert scored.metrics["initials"] == 0.5
    assert scored.detail["failures"]["initials"]["identity"]["source"].startswith("file:")


def test_an_unknown_mode_or_class_is_refused_before_scoring():
    task = Mask()
    schema = SpanSchema(classes=["PER"])
    with pytest.raises(TaskError, match="No failure mode named 'fragmentd'"):
        task.check(task.params({"failures": ["fragmentd"]}), schema)
    with pytest.raises(TaskError, match="asked for LOC"):
        task.check(task.params({"classes": ["LOC"]}), schema)


def test_a_mode_named_twice_is_refused():
    task = Mask()
    with pytest.raises(TaskError, match="more than once"):
        task.check(task.params({"failures": ["missed", "missed"]}), SpanSchema(classes=["PER"]))


def test_the_text_is_required():
    task = Mask()
    with pytest.raises(TaskError, match="needs each document's text"):
        task.score([Document(Spans(), Spans(), None)], task.params())


def test_a_mode_returning_the_wrong_shape_for_its_side_is_refused_by_name(tmp_path):
    (tmp_path / "failures.py").write_text(
        "from strata.evaluation.failures import FailureMode\n\n\n"
        "class Confused(FailureMode):\n"
        '    name = "confused"\n'
        '    version = "1"\n'
        '    side = "gold"\n'
        '    description = "counts on the gold side and returns a number"\n'
        '    example = ("[A|PER]", "A")\n\n'
        "    def detect(self, a):\n"
        "        return 1.0\n"
    )
    ref = f"{tmp_path / 'failures.py'}:Confused"
    with pytest.raises(TaskError, match="'confused' counts on the gold side and returned float"):
        score("[A|PER]", "A", failures=[ref])
