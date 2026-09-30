"""Finding a task or a failure mode by the name a file gives it, and saying what was found.

Two paths, and the name says which, as for models (``docs/adr/0034``). A
short name resolves through the kind's entry point group; anything with a
``:`` is a direct reference, ``file.py:Class`` anchored at a root, or an
installed ``module:Class``. Whatever resolves comes back with its
:class:`~strata.evaluation.identity.Identity`, which is what a score records
and a comparison checks. See ``docs/adr/0043``.
"""

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

from strata.common import plugins

from .identity import Identity, of_entry_point, of_file


class PluginError(ValueError):
    """A name that does not resolve to the kind of thing asked for."""


def _task_base() -> type:
    from .tasks.base import Task

    return Task


#: Each kind: the entry point group it is registered under, and the class
#: what resolves must subclass. The base is looked up late, so a kind's
#: module is imported only when that kind is asked for.
KINDS: dict[str, tuple[str, Callable[[], type]]] = {
    "task": ("strata.evaluation_tasks", _task_base),
}


def entries(kind: str) -> list[EntryPoint]:
    """What is installed for ``kind``. The seam a test patches."""
    group, _ = _kind(kind)
    return list(entry_points(group=group))


def available(kind: str) -> dict[str, str]:
    """Registered names for ``kind``, and what each resolves to."""
    return plugins.available(entries(kind))


def anchored(ref: str, root: Path | None) -> str:
    """``ref`` with a ``file.py`` target made absolute against ``root``.

    So it resolves from anywhere once written into a request; the
    experiment makes it relative to the project again before keying. The
    same rule as ``strata.modelling``'s ``absolute``.
    """
    if ":" not in ref:
        return ref
    target, name = ref.rsplit(":", 1)
    if not target.endswith(".py"):
        return ref
    path = Path(target)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    return f"{path.resolve()}:{name}"


def resolve(kind: str, ref: str, root: Path | None = None) -> tuple[Any, Identity]:
    """The class ``ref`` names, as ``kind``, with its identity."""
    _, base = _kind(kind)
    wanted = base()
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
        raise PluginError(f"{ref!r} is {found!r}, which is not a {wanted.__name__}.")
    return found, of_file(found.name, found.version, path)


def _kind(kind: str) -> tuple[str, Callable[[], type]]:
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
    """Loaded once per path and content, and registered before it runs.

    Keyed on the whole path, as a model's file is, so two projects' files of
    one name are two modules (``docs/adr/0025``). And on the bytes, so a file
    edited while a process runs is loaded again rather than scored by the
    old code under the new file's identity.
    """
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
