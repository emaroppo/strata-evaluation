"""The text-generation task: each metric on outputs small enough to count by hand."""

import random

import pytest

from strata.evaluation.identity import Identity
from strata.evaluation.registry import resolve
from strata.evaluation.tasks import Generated, Generation, TaskError, TextGeneration
from strata.evaluation.tasks.text_generation import (
    SuffixAutomaton,
    chrf,
    distinct,
    longest_runs,
    quantile,
)

ID = Identity(name="t", version="1", source="test==0")
LONG = " ".join(f"w{i}" for i in range(1, 31))


def generation(*outputs, targets=None, training=()):
    made = [o if isinstance(o, Generated) else Generated(o) for o in outputs]
    return Generation(made, list(targets or [""] * len(made)), list(training))


def test_the_task_names_the_media_it_reads():
    assert (TextGeneration.name, TextGeneration.media) == ("text-generation", "text")


def test_it_is_registered_and_its_numbers_carry_the_package():
    found, identity = resolve("task", "text-generation")
    assert found is TextGeneration
    assert identity.source.startswith("strata-evaluation==")
    task = found()
    recorded = task.recorded(generation("a b"), task.params(), identity)
    assert recorded.identity == identity


def test_distinct_ngrams_are_counted_across_every_output():
    texts = ["a b c", "a b c d"]
    assert distinct(texts, 1) == pytest.approx(4 / 7)
    assert distinct(texts, 2) == pytest.approx(3 / 5)
    assert distinct(texts, 3) == pytest.approx(2 / 3)


def test_distinct_ngrams_count_words_as_written_and_nothing_scores_zero():
    assert distinct(["A a"], 1) == 1.0
    assert distinct(["one"], 2) == 0.0
    assert distinct([], 1) == 0.0


def test_a_run_is_found_whatever_the_whitespace_and_case():
    runs = longest_runs(["The  QUICK\nbrown cat"], ["the quick brown fox"])
    assert runs == [3]


def test_a_run_never_spans_two_training_targets():
    """Read end to end, the targets hold all five words in a row; no one target does."""
    runs = longest_runs(
        ["quick brown fox jumps over"], ["the quick brown fox", "jumps over the lazy dog"]
    )
    assert runs == [3]


def test_the_automaton_agrees_with_comparing_every_pair():
    rng = random.Random(0)
    vocabulary = ["a", "b", "c"]

    def text(n):
        return [rng.choice(vocabulary) for _ in range(n)]

    def brute(query, targets):
        best = 0
        for target in targets:
            for i in range(len(query)):
                for j in range(len(target)):
                    k = 0
                    while i + k < len(query) and j + k < len(target):
                        if query[i + k] != target[j + k]:
                            break
                        k += 1
                    best = max(best, k)
        return best

    for _ in range(50):
        targets = [text(rng.randint(0, 12)) for _ in range(rng.randint(0, 4))]
        query = text(rng.randint(0, 12))
        assert SuffixAutomaton(targets).longest_match(query) == brute(query, targets)


def test_the_quantiles_interpolate_between_ranks():
    assert quantile([3, 0, 25, 2, 0], 0.5) == 2
    assert quantile([3, 0, 25, 2, 0], 0.9) == pytest.approx(16.2)
    assert quantile([], 0.5) == 0.0


def test_memorisation_is_reported_as_a_spread_of_longest_runs():
    task = TextGeneration()
    outputs = [" ".join(LONG.split()[:25]), "w1 w2 w3", "x y", "w5 x w6 w7", ""]
    scored = task.score(generation(*outputs, training=[LONG]), task.params())
    assert scored.detail["longest_run"] == [25, 3, 0, 2, 0]
    assert scored.metrics["longest_run_median"] == 2
    assert scored.metrics["longest_run_p90"] == pytest.approx(16.2)
    assert scored.metrics["longest_run_max"] == 25
    assert scored.metrics["longest_run_over_20"] == pytest.approx(0.2)
    assert scored.counts["outputs"] == 5


def test_ended_leaves_out_an_output_that_did_not_say_and_says_so():
    task = TextGeneration()
    ended = [True, False, None, True, True]
    scored = task.score(generation(*[Generated("x", ended=e) for e in ended]), task.params())
    assert scored.metrics["ended"] == 0.75
    assert (scored.counts["ended"], scored.counts["ended.of"]) == (3, 4)
    assert scored.detail["ended_unknown"] == 1


def test_ended_says_nothing_extra_when_every_output_said():
    task = TextGeneration()
    scored = task.score(generation(Generated("x", ended=True)), task.params())
    assert scored.metrics["ended"] == 1.0
    assert "ended_unknown" not in scored.detail


def test_chrf_is_off_by_default():
    task = TextGeneration()
    scored = task.score(generation("the cat", targets=["the cat"]), task.params())
    assert "chrf" not in scored.metrics


def test_chrf_scores_each_output_against_its_own_target_when_asked():
    task = TextGeneration()
    scored = task.score(
        generation("the cat sat on the mat", targets=["the cat is on the mat"]),
        task.params({"chrf": True}),
    )
    # sacrebleu 2.4.3's corpus chrF of the pair is 64.5779420625287.
    assert scored.metrics["chrf"] == pytest.approx(0.645779420625287)


def test_chrf_averages_only_the_orders_both_sides_reach():
    """By hand: 1-grams p 1, r 2/3; 2-grams p 1, r 1/2; no 3-gram on one side.

    F2 = 5 * 1 * 7/12 / (4 + 7/12) = 7/11.
    """
    assert chrf(["ab"], ["abc"]) == pytest.approx(7 / 11)


def test_chrf_pools_counts_over_pairs_and_ignores_whitespace():
    assert chrf(["a b"], ["ab"]) == 1.0
    # sacrebleu 2.4.3: 52.91454885135162.
    assert chrf(
        ["the cat sat on the mat", "hello there"], ["the cat is on the mat", "hello world"]
    ) == pytest.approx(0.5291454885135162)
    assert chrf([], []) == 0.0


def test_an_output_needs_its_target():
    with pytest.raises(TaskError, match="one output per validation sample"):
        Generation([Generated("x")], [], [])


def test_a_generation_task_refuses_a_parameter_it_does_not_take():
    with pytest.raises(TaskError, match="will not take"):
        TextGeneration().params({"chrF": True})


def test_a_generation_task_refuses_outputs_of_another_media():
    task = TextGeneration()
    with pytest.raises(TaskError, match="reads generated text"):
        task.check(task.params(), "image")


def test_no_outputs_scores_zeros():
    task = TextGeneration()
    scored = task.score(generation(training=["a b"]), task.params({"chrf": True}))
    assert scored.counts["outputs"] == 0
    assert set(scored.metrics.values()) == {0.0}
