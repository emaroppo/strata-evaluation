"""How a prediction is scored against an answer.

``tasks`` say what counts as getting it right, ``spans`` what a region is,
``counting`` turns hits and misses into rates. See ``docs/adr/0035``.
"""

from . import tasks
from .counting import Tally

#: Modules a sibling may import by path, each task module among them (docs/adr/0015).
PUBLIC_MODULES = frozenset(
    {
        "alignment",
        "conformance",
        "failures",
        "identity",
        "markup",
        "registry",
        "spans",
        "tasks",
        "tasks.base",
        "tasks.classify",
        "tasks.entities",
        "tasks.mask",
        "tasks.text_generation",
        "text",
    }
)

__all__ = ["PUBLIC_MODULES", "Tally", "tasks"]
