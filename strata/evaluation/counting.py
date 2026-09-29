"""Counts, and the rates read from them.

Knows nothing about label types or tasks: a task decides what counts as a
hit, and hands the count here. Kept as counts until read, because a rate
computed early cannot be pooled with another and a tally can.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tally:
    """True positives, false positives and false negatives for one question."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @classmethod
    def of(cls, truth: set, guessed: set) -> "Tally":
        """What agreeing and disagreeing on two sets comes to."""
        return cls(len(truth & guessed), len(guessed - truth), len(truth - guessed))

    def __add__(self, other: "Tally") -> "Tally":
        return Tally(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def support(self) -> int:
        """How many times the answer asserts it."""
        return self.tp + self.fn
