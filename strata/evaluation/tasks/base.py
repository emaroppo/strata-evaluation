"""What a task is: a question asked of one label type, or of a generator's outputs.

Subclass :class:`Task`, as a model subclasses ``LabelModel``: name it,
version it, say which label type it reads, and implement :meth:`Task.score`.
A task that scores what a generator made subclasses :class:`GenerationTask`
instead, as a model that makes media subclasses ``MediaModel``, and says
which media it reads. Both are found by name through the
``strata.evaluation_tasks`` entry point group, or from a project's own
``file.py:Class``. See ``docs/adr/0042`` and ``docs/adr/0054``.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from strata.contracts import AnySchema, Value

from ..identity import Identity


class TaskError(ValueError):
    """A task asked for something it cannot score, found before anything is scored."""


@dataclass(frozen=True)
class Document:
    """One sample as a task sees it: the answer, the guess, and the text if asked for."""

    truth: Value
    prediction: Value
    #: None unless the task says it ``needs_text``, so a classification run
    #: reads no file it does not use.
    text: str | None = None


@dataclass
class Scored:
    """What :meth:`Task.score` returns: the task's own numbers, before identity is attached.

    ``metrics`` is what a line and a grid show. ``counts`` keeps the tallies
    the rates came from, so two scores can be pooled by count. ``detail`` is
    the task's own, as long as it is JSON.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


class TaskScore(BaseModel):
    """A task's score as recorded: its numbers, what produced them, and with what."""

    model_config = ConfigDict(extra="forbid")

    identity: Identity
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class Generated:
    """One output a generator made from a validation sample's input."""

    #: What it made, as its media's task reads it: for text, the text.
    output: str
    #: True when it stopped on its own, False when it reached the token
    #: limit, None when the generator did not say.
    ended: bool | None = None


@dataclass(frozen=True)
class Generation:
    """What a generation task is handed: the outputs, and the targets beside them.

    The evaluate stage generates one output for each validation sample's
    input, so ``outputs`` and ``targets`` are positional. ``training`` is
    every training sample's target, which memorisation is measured against.
    See ``docs/adr/0054``.
    """

    outputs: Sequence[Generated]
    targets: Sequence[str]
    training: Sequence[str]

    def __post_init__(self) -> None:
        if len(self.outputs) != len(self.targets):
            raise TaskError(
                f"{len(self.outputs)} outputs and {len(self.targets)} targets: there is "
                f"one output per validation sample, in the same order."
            )


class NoParams(BaseModel):
    """A task that takes nothing refuses anything it is given."""

    model_config = ConfigDict(extra="forbid")


class BaseTask(ABC):
    """What every task shares: a name, a version, and what it can be asked.

    Not subclassed directly: a task is a :class:`Task` or a
    :class:`GenerationTask`, and which one says what it reads. The registry
    finds either under the one entry point group.
    """

    #: What a file names it by, and what its score is recorded under.
    name: ClassVar[str]
    #: Bumped when a change alters what an unchanged input scores. A score's
    #: identity carries it, so two scores are compared only across one.
    version: ClassVar[str]
    #: What it can be asked, as a model refusing unknown keys.
    Params: ClassVar[type[BaseModel]] = NoParams

    def params(self, given: dict | None = None) -> BaseModel:
        """``given`` as this task's parameters, or a :class:`TaskError` naming the problem."""
        try:
            return self.Params.model_validate(given or {})
        except ValueError as e:
            raise TaskError(f"Task {self.name!r} will not take these parameters: {e}") from None

    def requires(self, params: BaseModel) -> list[tuple[str, str]]:
        """Other plugins these parameters name, as ``(kind, ref)``.

        Resolved to identities alongside the task's own, so a change in one
        of them is a change in the score's identity. None by default.
        """
        return []

    def _recorded(self, scored: Scored, params: BaseModel, identity: Identity) -> TaskScore:
        return TaskScore(
            identity=identity,
            params=params.model_dump(mode="json"),
            metrics=scored.metrics,
            counts=scored.counts,
            per_class=scored.per_class,
            detail=scored.detail,
        )


class Task(BaseTask):
    """Subclass this, name it, and implement :meth:`score`."""

    #: The label type it reads: classification, span or bbox.
    label_type: ClassVar[str]
    #: Whether :meth:`score` is handed each document's text.
    needs_text: ClassVar[bool] = False

    def check(self, params: BaseModel, schema: AnySchema) -> None:
        """Refuse, before anything is scored, what this label set makes impossible.

        The default refuses a ``classes`` parameter naming a class the label
        set does not have, and a label set of another label type.
        """
        if schema.label_type != self.label_type:
            raise TaskError(
                f"Task {self.name!r} reads {self.label_type} label sets, and this one "
                f"is {schema.label_type}."
            )
        wanted = getattr(params, "classes", None)
        if wanted:
            unknown = sorted(set(wanted) - set(schema.classes))
            if unknown:
                raise TaskError(
                    f"Task {self.name!r} is asked for {', '.join(unknown)}, which this "
                    f"label set does not have (it has: {', '.join(schema.classes)})."
                )

    @abstractmethod
    def score(self, documents: Sequence[Document], params: BaseModel) -> Scored:
        """Score every document together. Empty input scores zeros, never raises."""
        ...

    def recorded(
        self, documents: Sequence[Document], params: BaseModel, identity: Identity
    ) -> TaskScore:
        """:meth:`score`, with what produced it attached."""
        return self._recorded(self.score(documents, params), params, identity)


class GenerationTask(BaseTask):
    """Subclass this, name it, say which media it reads, and implement :meth:`score`.

    A generator's outputs are scored on what they are, not against an
    answer, so the task is chosen by the media of what the model makes:
    the media of the type it ``trains_on``. See ``docs/adr/0054``.
    """

    #: The media of the outputs it reads, as a sample type names it: text.
    media: ClassVar[str]

    def check(self, params: BaseModel, media: str) -> None:
        """Refuse, before anything is generated, outputs of another media."""
        if media != self.media:
            raise TaskError(
                f"Task {self.name!r} reads generated {self.media}, and this model makes {media}."
            )

    @abstractmethod
    def score(self, generation: Generation, params: BaseModel) -> Scored:
        """Score every output together. No outputs scores zeros, never raises."""
        ...

    def recorded(self, generation: Generation, params: BaseModel, identity: Identity) -> TaskScore:
        """:meth:`score`, with what produced it attached."""
        return self._recorded(self.score(generation, params), params, identity)
