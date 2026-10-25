"""The text-generation task: what a text generator's outputs are, not whether they match.

Memorisation (longest run shared with a training target), diversity
(distinct n-grams), chrF against the target when asked, and the share that
``ended`` on their own. See ``docs/adr/0054`` and ``docs/adr/0042``.
"""

from collections import Counter
from collections.abc import Hashable, Sequence

from pydantic import BaseModel, ConfigDict

from .base import Generation, GenerationTask, Scored

NAME = "text-generation"
#: The media this task reads.
MEDIA = "text"
#: Outputs sharing a run longer than this with a training target are counted apart.
LONG_RUN = 20
#: chrF as usually reported: character n-grams up to six, recall weighted twice precision.
CHRF_ORDER = 6
CHRF_BETA = 2


class TextGenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Score each output against its validation target with chrF; off for open-ended writing.
    chrf: bool = False


def words(text: str) -> list[str]:
    """``text`` as memorisation reads it: split on whitespace, case folded."""
    return text.casefold().split()


def distinct(texts: Sequence[str], n: int) -> float:
    """Distinct ``n``-grams over all ``n``-grams, across every text, in whitespace words."""
    grams: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        tokens = text.split()
        grams.update(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    total = grams.total()
    return len(grams) / total if total else 0.0


class SuffixAutomaton:
    """Every run of consecutive words in a set of texts, for longest-match queries in linear time.

    Each text ends in a separator no query can contain, so a match never runs into the next text.
    """

    def __init__(self, texts: Sequence[Sequence[str]]) -> None:
        self.next: list[dict[Hashable, int]] = [{}]
        self.link = [-1]
        self.length = [0]
        last = 0
        for n, text in enumerate(texts):
            # The separator is an int, where every query word is a str.
            for token in [*text, n]:
                last = self._extend(last, token)

    def _extend(self, last: int, token: Hashable) -> int:
        cur = len(self.next)
        self.next.append({})
        self.length.append(self.length[last] + 1)
        self.link.append(0)
        p = last
        while p != -1 and token not in self.next[p]:
            self.next[p][token] = cur
            p = self.link[p]
        if p != -1:
            q = self.next[p][token]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.next)
                self.next.append(dict(self.next[q]))
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])
                while p != -1 and self.next[p].get(token) == q:
                    self.next[p][token] = clone
                    p = self.link[p]
                self.link[q] = self.link[cur] = clone
        return cur

    def longest_match(self, query: Sequence[str]) -> int:
        """The most consecutive words of ``query`` found together in one text."""
        state, length, best = 0, 0, 0
        for token in query:
            while state and token not in self.next[state]:
                state = self.link[state]
                length = self.length[state]
            if token in self.next[state]:
                state = self.next[state][token]
                length += 1
            best = max(best, length)
        return best


def longest_runs(outputs: Sequence[str], training: Sequence[str]) -> list[int]:
    """Each output's longest run of words shared with any one training target."""
    automaton = SuffixAutomaton([words(t) for t in training])
    return [automaton.longest_match(words(o)) for o in outputs]


def quantile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile, interpolated between the two nearest ranks; zero for nothing."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    at = q * (len(ordered) - 1)
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _char_ngrams(text: str, n: int) -> Counter[str]:
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf(outputs: Sequence[str], targets: Sequence[str]) -> float:
    """Corpus chrF of the outputs against their targets, from 0 to 1: sacrebleu's defaults, over a
    hundred.
    """
    totals = [[0, 0, 0] for _ in range(CHRF_ORDER)]
    for output, target in zip(outputs, targets, strict=True):
        hypothesis, reference = "".join(output.split()), "".join(target.split())
        for n, total in enumerate(totals, start=1):
            made, wanted = _char_ngrams(hypothesis, n), _char_ngrams(reference, n)
            total[0] += made.total()
            total[1] += wanted.total()
            total[2] += (made & wanted).total()
    precision = recall = 0.0
    orders = 0
    for made, wanted, matched in totals:
        if made and wanted:
            precision += matched / made
            recall += matched / wanted
            orders += 1
    if not orders:
        return 0.0
    precision, recall = precision / orders, recall / orders
    if not precision + recall:
        return 0.0
    factor = CHRF_BETA**2
    return (1 + factor) * precision * recall / (factor * precision + recall)


class TextGeneration(GenerationTask):
    """The text-generation task, as a plugin: registered under ``text-generation``."""

    name = NAME
    version = "1"
    media = MEDIA
    Params = TextGenerationParams

    def score(self, generation: Generation, params: BaseModel) -> Scored:
        texts = [o.output for o in generation.outputs]
        said = [o.ended for o in generation.outputs if o.ended is not None]
        runs = longest_runs(texts, generation.training)

        metrics = {
            "ended": sum(said) / len(said) if said else 0.0,
            "distinct_1": distinct(texts, 1),
            "distinct_2": distinct(texts, 2),
            "distinct_3": distinct(texts, 3),
            "longest_run_median": quantile(runs, 0.5),
            "longest_run_p90": quantile(runs, 0.9),
            "longest_run_max": float(max(runs, default=0)),
            f"longest_run_over_{LONG_RUN}": (
                sum(r > LONG_RUN for r in runs) / len(runs) if runs else 0.0
            ),
        }
        if getattr(params, "chrf", False):
            metrics["chrf"] = chrf(texts, generation.targets)

        detail: dict = {"longest_run": runs}
        unknown = len(texts) - len(said)
        if unknown:
            # Left out of ``ended``, not counted either way. docs/adr/0036
            detail["ended_unknown"] = unknown
        return Scored(
            metrics=metrics,
            counts={"outputs": len(texts), "ended": sum(said), "ended.of": len(said)},
            detail=detail,
        )
