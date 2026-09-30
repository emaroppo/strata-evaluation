# strata-evaluation

How a prediction is scored against an answer: the counts, the matching,
and the numbers read from them. Pure functions over `strata-contracts`
values, so anything that has predictions and answers can score them the
same way — the `evaluate` stage in `strata-modelling`, a model's own
training loop, a notebook reading records.

Not on PyPI: it installs from its repository at a release tag. uv takes a
git source only for a package named directly, so the strata packages
beneath it are named beside it.

```bash
g=git+https://github.com/emaroppo
uv add "strata-evaluation @ $g/strata-evaluation@v0.1.0" \
       "strata-contracts @ $g/strata-contracts@v0.1.0" \
       "strata-common @ $g/strata-common@v0.1.0"
```

Depends on `strata-contracts`, and on `strata-common` for the entry-point
resolver. May not import an ML framework, a store, a model or Label
Studio: a score is a function of a prediction and an answer, and nothing
else.

## What it holds

A score is kept as a **`Tally`** — true positives, false positives, false
negatives — until it is read. Precision, recall and F1 are properties of
it, and tallies add, so two scores pool by count rather than by averaging
rates.

| function | scores | as |
|---|---|---|
| `choice_scores(truths, guesses)` | classification, one `Choices` per sample | exact match over the set of classes, micro and per-class tallies over every assertion |
| `span_scores(truths, predicted)` | spans, one collection per document | exact and partial entity tallies, micro and per class |

Exact match is not called accuracy: a label set asserting two classes per
sample, scored by a model asserting one, has an exact match of zero. For
spans, a region carrying two labels is two entities, and partial matching
is one-to-one, so a sentence-wide span cannot score against every entity
inside it. Exact and partial are both kept, because one understates and
the other flatters. The reasons are recorded in the workspace's
`docs/adr/0035`.

## Tests

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
