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

**Tasks.** A task is what is being achieved, and how getting it right is
counted: it reads one label type and asks one question of it, and several
tasks can read the same label type. Subclass `Task` as a model subclasses
`Model`: a name, a version, the label type, the parameters it takes, and
`score`. What it records carries the **identity** of the code that
produced it, so two scores are compared only across one.

| task | reads | asks |
|---|---|---|
| `classify` | classification | exact match over each sample's set of classes; micro and per-class precision, recall, F1 |
| `entities` | span | was each entity found with its label and bounds: exact and partial (one-to-one) precision, recall, F1, micro and per class |
| `mask` | span | is the anonymised document right: one metric per failure mode, below |

Exact match is not called accuracy: a label set asserting two classes per
sample, scored by a model asserting one, has an exact match of zero. A
region carrying two labels is two entities to `entities`, and partial
matching is one-to-one, so a sentence-wide span cannot score against every
entity inside it (`docs/adr/0035` in the workspace).

**Failure modes.** `mask` measures each way a span prediction can be wrong
for anonymisation separately, so a change that trades one failure for
another shows as both. Characters mean non-whitespace characters, so the
space in a split name is neither a leak nor a spill.

| mode | on | the case |
|---|---|---|
| `missed` | gold entity | no prediction touches it |
| `truncated` | gold entity | touched, but part of it is left uncovered |
| `fragmented` | gold entity | split across predictions that do not overlap: "Dear [PER] [PER]" |
| `mislabelled` | gold entity | a prediction touching it has none of its labels |
| `shared` | gold entity | inside a prediction that covers another entity too |
| `spurious` | prediction | touches no entity |
| `spilled` | prediction | runs past the first or last entity it touches |
| `merging` | prediction | covers two or more entities; the words between are *bridge* |
| `duplicate` | prediction | overlaps another prediction |

Beside them, aggregates: `leaked`, `char_recall`, `char_precision`, the
over-masked characters split as `spurious_chars`, `spilled_chars` and
`bridge_chars` (summing to 1 − `char_precision`), and
`placeholder_count_error` per document, counted per class so a fragment and
a miss cannot cancel. `clean` is an entity no active mode flagged.

**Plugins.** Tasks and failure modes are found by name through the
`strata.evaluation_tasks` and `strata.failure_modes` entry point groups, or
from a project's own `file.py:Class`. A file's identity is its path and the
hash of its bytes, so editing it is a new identity:

```python
from strata.evaluation.failures import FailureMode

class SignatureBlock(FailureMode):
    name = "signature_block"
    version = "1"
    side = "prediction"
    description = "a prediction covering a whole signature block"
    example = ("Regards,\n[Anne|PER]", "[Regards,\nAnne|PER]")

    def detect(self, a):
        ...
```

`strata.evaluation.conformance` holds the promise as tests a plugin runs:
a failure mode's example raises it and no isolated first-party mode outside
its `co_occurs`, so one that measures what another does fails, naming it.

Underneath: `alignment`, one document's answer and guess laid over each
other character by character, which every mode reads; `spans`, the geometry
tasks share; `counting.Tally`, counts that pool; `text`, a document read as
its offsets were made, refused rather than scored as empty when it cannot
be read; and `markup`, spans written inline (`Dear [John Smith|PER],`) for
examples.

## Tests

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
