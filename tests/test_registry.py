"""Finding a task by name or by file, and knowing which code was found."""

import pytest

from strata.evaluation import registry
from strata.evaluation.registry import PluginError, anchored, available, resolve
from strata.evaluation.tasks import Classify, Entities

LOCAL = """
from strata.evaluation.tasks import Scored, Task


class Loud(Task):
    name = "loud"
    version = "{version}"
    label_type = "span"

    def score(self, documents, params):
        return Scored(metrics={{"documents": float(len(documents))}})
"""


def test_the_first_party_tasks_are_registered():
    assert {"classify", "entities"} <= set(available("task"))


def test_a_short_name_resolves_through_the_entry_point():
    found, identity = resolve("task", "entities")
    assert found is Entities
    assert identity.name == "entities" and identity.version == "1"
    assert identity.source.startswith("strata-evaluation==")


def test_an_unknown_name_is_refused_listing_what_is_installed():
    with pytest.raises(PluginError, match=r"No task named 'entitys'.*classify"):
        resolve("task", "entitys")


def test_a_file_reference_resolves_against_the_root(tmp_path):
    (tmp_path / "tasks.py").write_text(LOCAL.format(version="1"))
    found, identity = resolve("task", "tasks.py:Loud", root=tmp_path)
    assert found.name == "loud"
    assert identity.source.startswith(f"file:{tmp_path.resolve() / 'tasks.py'}#sha256:")


def test_editing_a_file_changes_its_identity_and_what_runs(tmp_path):
    """The bytes are the version that means something, declared or not."""
    path = tmp_path / "tasks.py"
    path.write_text(LOCAL.format(version="1"))
    first, before = resolve("task", "tasks.py:Loud", root=tmp_path)
    path.write_text(LOCAL.format(version="1").replace("len(documents)", "2 * len(documents)"))
    second, after = resolve("task", "tasks.py:Loud", root=tmp_path)
    assert before.source != after.source
    assert second is not first
    assert second().score([object()], None).metrics["documents"] == 2.0


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(PluginError, match="missing file"):
        resolve("task", "nope.py:Loud", root=tmp_path)


def test_a_reference_to_something_that_is_not_a_task_is_refused(tmp_path):
    (tmp_path / "tasks.py").write_text("class Loud:\n    name = 'loud'\n")
    with pytest.raises(PluginError, match="not a Task"):
        resolve("task", "tasks.py:Loud", root=tmp_path)


def test_an_installed_module_reference_resolves():
    found, identity = resolve("task", "strata.evaluation.tasks.classify:Classify")
    assert found is Classify
    assert identity.source.startswith("file:") and "classify.py" in identity.source


def test_an_entry_point_naming_another_task_is_refused(monkeypatch):
    """A score is recorded under the task's name; a registration disagreeing is a lie."""
    from importlib.metadata import EntryPoint

    wrong = EntryPoint("counts", "strata.evaluation.tasks.classify:Classify", "x")
    monkeypatch.setattr(registry, "entries", lambda kind: [wrong])
    with pytest.raises(PluginError, match="names itself 'classify'"):
        resolve("task", "counts")


def test_anchoring_makes_a_file_reference_absolute_and_leaves_names_alone(tmp_path):
    assert anchored("entities", tmp_path) == "entities"
    assert anchored("pkg.mod:Task", tmp_path) == "pkg.mod:Task"
    assert anchored("tasks.py:Loud", tmp_path) == f"{(tmp_path / 'tasks.py').resolve()}:Loud"


def test_an_unknown_kind_is_refused():
    with pytest.raises(PluginError, match="No kind"):
        resolve("x", "y")
