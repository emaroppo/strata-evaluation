"""Tallies: counts that pool, and rates read from them."""

from strata.evaluation import Tally


def test_an_empty_tally_reads_as_zero_rather_than_raising():
    assert (Tally().precision, Tally().recall, Tally().f1) == (0.0, 0.0, 0.0)


def test_tallies_pool_by_count_not_by_rate():
    """Averaging two rates weighs a sample of one like a sample of a thousand."""
    pooled = Tally(1, 0, 0) + Tally(0, 0, 3)
    assert pooled.recall == 0.25
    assert pooled.support == 4


def test_a_tally_of_two_sets_counts_agreement_and_both_disagreements():
    assert Tally.of({"a", "b"}, {"b", "c"}) == Tally(tp=1, fp=1, fn=1)
