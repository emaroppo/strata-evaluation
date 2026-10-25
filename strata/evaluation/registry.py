"""Finding a task or a failure mode by name, with the identity of what was found.

A short name resolves through an entry point group; ``file.py:Class`` or
``module:Class`` is a direct reference. See ``docs/adr/0034`` and ``docs/adr/0042``.
"""

import hashlib
import importlib
import importlib.util
import sys
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

from strata.common import plugins

from .failures import FailureMode
from .identity import Identity, of_entry_point, of_file
from .tasks.base import BaseTask


class PluginError(ValueError):
    """A name that does not resolve to the kind of thing asked for."""


#: Each kind: its entry point group, and the base class a plugin of it subclasses.
KINDS: dict[str, tuple[str, type]] = {
    "task": ("strata.evaluation_tasks", BaseTask),
    "failure_mode": ("strata.failure_modes", FailureMode),
}

#: What a refusal calls a base whose class name is not what a plugin subclasses.
_CALLED = {"BaseTask": "Task"}


def entries(kind: str) -> list[EntryPoint]:
    """What is installed for ``kind``. The seam a test patches."""
    group, _ = _kind(kind)
    return list(entry_points(group=group))


def available(kind: str) -> dict[str, str]:
    """Registered names for ``kind``, and what each resolves to."""
    return plugins.available(entries(kind))


def resolve(kind: str, ref: str, root: Path | None = None) -> tuple[Any, Identity]:
    """The class ``ref`` names, as ``kind``, with its identity."""
    _, wanted = _kind(kind)
    if ":" not in ref:
        entry = plugins.find(
            entries(kind),
            ref,
            what=kind.replace("_", " "),
            error=PluginError,
            hint=" A name with a ':' is a direct reference instead, e.g. failures.py:Class.",
        )
        found = plugins.load(entry, wanted, error=PluginError)
        if found.name != ref:
            raise PluginError(
                f"{ref!r} is registered for {found.__name__}, which names itself "
                f"{found.name!r}; a score is recorded under the name, so they must agree."
            )
        return found, of_entry_point(found.name, found.version, entry)

    target, class_name = ref.rsplit(":", 1)
    if target.endswith(".py"):
        path = _anchor(target, root)
        module = _module_from_file(path)
    else:
        module = _module_from_import(target)
        path = Path(module.__file__ or "")
    found = getattr(module, class_name, None)
    if found is None:
        raise PluginError(f"No {class_name!r} in {target}.")
    if not (isinstance(found, type) and issubclass(found, wanted)):
        called = _CALLED.get(wanted.__name__, wanted.__name__)
        raise PluginError(f"{ref!r} is {found!r}, which is not a {called}.")
    return found, of_file(found.name, found.version, path)


def _kind(kind: str) -> tuple[str, type]:
    try:
        return KINDS[kind]
    except KeyError:
        raise PluginError(f"No kind {kind!r}. Known: {', '.join(sorted(KINDS))}.") from None


def _anchor(target: str, root: Path | None) -> Path:
    path = Path(target)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if not path.exists():
        raise PluginError(f"A reference points at a missing file: {path}")
    return path.resolve()


def _module_from_file(path: Path):
    """Loaded once per path and content: an edited file is loaded again under its new identity."""
    digest = hashlib.sha256(str(path).encode() + path.read_bytes()).hexdigest()[:12]
    module_name = f"strata_evaluation_local_{path.stem}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PluginError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_name]
        raise PluginError(f"{path} will not import: {exc}") from exc
    return module


def _module_from_import(target: str):
    try:
        return importlib.import_module(target)
    except ImportError as exc:
        raise PluginError(f"{target!r} will not import: {exc}") from exc
