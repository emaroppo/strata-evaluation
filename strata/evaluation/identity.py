"""Which code produced a number.

A score is comparable with another only when the same code produced both.
With tasks and failure modes resolvable from a project's own file, "the
same code" cannot be the package version alone, so every score carries the
identity of what computed it: its name, the version it declares, and where
it came from. See ``docs/adr/0042``.
"""

import hashlib
from importlib.metadata import EntryPoint
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Identity(BaseModel):
    """A plugin's name, its declared version, and its source.

    ``source`` is ``"<distribution>==<version>"`` for something an installed
    package registered, or ``"file:<path>#sha256:<digest>"`` for a project's
    own file, whose bytes are the only version that means anything: editing
    the file changes the identity, declared version or not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    source: str


def of_entry_point(name: str, version: str, entry: EntryPoint) -> Identity:
    dist = getattr(entry, "dist", None)
    source = f"{dist.name}=={dist.version}" if dist is not None else f"entry-point:{entry.value}"
    return Identity(name=name, version=version, source=source)


def of_file(name: str, version: str, path: Path) -> Identity:
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Identity(name=name, version=version, source=f"file:{path}#sha256:{digest}")
