"""What a task is: a question asked of one label type, and how its answer is recorded.

Subclass :class:`Task`, as a model subclasses ``Model``: name it, version
it, say which label type it reads, and implement :meth:`Task.score`. A
task is found by name through the ``strata.evaluation_tasks`` entry point
group, or from a project's own ``file.py:Class``. See ``docs/adr/0042``.
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


class NoParams(BaseModel):
    """A task that takes nothing refuses anything it is given."""

    model_config = ConfigDict(extra="forbid")


class Task(ABC):
    """Subclass this, name it, and implement :meth:`score`."""

    #: What a file names it by, and what its score is recorded under.
    name: ClassVar[str]
    #: Bumped when a change alters what an unchanged input scores. A score's
    #: identity carries it, so two scores are compared only across one.
    version: ClassVar[str]
    #: The label type it reads: classification, span or bbox.
    label_type: ClassVar[str]
    #: Whether :meth:`score` is handed each document's text.
    needs_text: ClassVar[bool] = False
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
        scored = self.score(documents, params)
        return TaskScore(
            identity=identity,
            params=params.model_dump(mode="json"),
            metrics=scored.metrics,
            counts=scored.counts,
            per_class=scored.per_class,
            detail=scored.detail,
        )
