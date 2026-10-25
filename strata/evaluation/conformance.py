"""What a task or a failure mode promises, as tests its author runs.

Subclass a contract in a plugin's tests and supply its fixture, as a model
does with ``ModelContract``. See ``docs/adr/0033`` and ``docs/adr/0054``.
"""

import pytest

from .alignment import align
from .failures import DEFAULT, SIDES, FailureMode
from .markup import pair
from .registry import resolve
from .tasks.base import BaseTask, Generation, GenerationTask, Task, TaskError


def _detect(mode: FailureMode, example: tuple[str, str]):
    text, gold, predicted = pair(*example)
    return mode.detect(align(text, gold, predicted))


class FailureModeContract:
    """Subclass this in a plugin's tests and supply the ``mode`` fixture (the class)."""

    @pytest.fixture
    def mode(self) -> type[FailureMode]:
        raise NotImplementedError("Supply the failure mode class as the `mode` fixture.")

    def test_it_is_a_failure_mode(self, mode):
        assert isinstance(mode, type) and issubclass(mode, FailureMode)

    def test_it_says_what_it_is(self, mode):
        assert mode.name and mode.version and mode.description
        assert mode.side in SIDES, f"side {mode.side!r} is not one of {SIDES}"

    def test_its_example_raises_it(self, mode):
        detected = _detect(mode(), mode.example)
        assert mode.raised(detected), f"{mode.name}'s own example does not raise it"

    def test_its_example_raises_no_other_mode(self, mode):
        """One failure, one metric: no isolated mode outside ``co_occurs`` is raised."""
        if not mode.isolated:
            pytest.skip(f"{mode.name} is an aggregate, which other failures move")
        others = []
        for name in DEFAULT:
            other, _ = resolve("failure_mode", name)
            if other.name == mode.name or not other.isolated or other.name in mode.co_occurs:
                continue
            if other.raised(_detect(other(), mode.example)):
                others.append(other.name)
        assert not others, (
            f"{mode.name}'s example also raises {', '.join(others)}: either they "
            f"measure one failure, or the example shows two, and co_occurs should say so"
        )

    def test_detecting_twice_gives_the_same_answer(self, mode):
        text, gold, predicted = pair(*mode.example)
        a = align(text, gold, predicted)
        assert mode().detect(a) == mode().detect(a)


class TaskContract:
    """Subclass this in a plugin's tests and supply the ``task`` fixture (the class)."""

    @pytest.fixture
    def task(self) -> type[BaseTask]:
        raise NotImplementedError("Supply the task class as the `task` fixture.")

    def test_it_is_a_task(self, task):
        assert isinstance(task, type) and issubclass(task, Task | GenerationTask)

    def test_it_says_what_it_is(self, task):
        assert task.name and task.version
        if issubclass(task, GenerationTask):
            assert isinstance(task.media, str) and task.media, "a generation task names its media"
        else:
            assert task.label_type in ("classification", "span", "bbox")

    def test_it_refuses_a_parameter_it_does_not_take(self, task):
        with pytest.raises(TaskError):
            task().params({"no_such_parameter_anywhere": 1})

    def test_nothing_to_score_scores_without_raising(self, task):
        instance = task()
        if isinstance(instance, GenerationTask):
            scored = instance.score(Generation([], [], []), instance.params())
        else:
            scored = instance.score([], instance.params())
        assert isinstance(scored.metrics, dict)

    def test_a_generation_task_refuses_another_media(self, task):
        if not issubclass(task, GenerationTask):
            pytest.skip(f"{task.name} reads a label type, which its check refuses instead")
        instance = task()
        instance.check(instance.params(), task.media)
        with pytest.raises(TaskError):
            instance.check(instance.params(), f"not-{task.media}")
