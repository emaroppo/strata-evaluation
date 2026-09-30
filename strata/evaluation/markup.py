"""Spans written inline, for examples a person can read: ``Dear [John Smith|PER],``.

``[text|LABEL]`` marks a span; spans nest, so two overlapping predictions
are ``[John [Smith|PER]|PER]``. What the markup strips to is the document's
text, and the offsets are into it. For failure modes' worked examples and
for tests; nothing reads annotations this way.
"""

from strata.contracts import Span


class MarkupError(ValueError):
    """Markup that does not say what spans it means."""


def parse(markup: str) -> tuple[str, list[Span]]:
    """The text the markup strips to, and its spans in reading order."""
    text: list[str] = []
    starts: list[int] = []
    spans: list[Span] = []
    i = 0
    while i < len(markup):
        char = markup[i]
        if char == "[":
            starts.append(len(text))
        elif char == "|":
            if not starts:
                raise MarkupError(f"A label with no span open, at {i}: {markup!r}")
            close = markup.find("]", i)
            if close == -1:
                raise MarkupError(f"A label never closed, at {i}: {markup!r}")
            labels = markup[i + 1 : close].split(",")
            start = starts.pop()
            spans.append(
                Span(labels=labels, start=start, end=len(text), text="".join(text[start:]))
            )
            i = close
        elif char == "]":
            raise MarkupError(f"A span closed with no label, at {i}: {markup!r}")
        else:
            text.append(char)
        i += 1
    if starts:
        raise MarkupError(f"{len(starts)} span(s) never closed: {markup!r}")
    return "".join(text), sorted(spans, key=lambda s: (s.start, s.end))


def pair(gold: str, predicted: str) -> tuple[str, list[Span], list[Span]]:
    """One text marked twice, as an answer and a guess. Refused if the texts differ."""
    text, truth = parse(gold)
    other, guess = parse(predicted)
    if text != other:
        raise MarkupError(f"The two markups are of different texts: {text!r} and {other!r}")
    return text, truth, guess
