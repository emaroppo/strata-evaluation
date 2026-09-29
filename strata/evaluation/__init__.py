"""How a prediction is scored against an answer.

Three layers, each knowing less than the one above: ``tasks`` say what is
being achieved and what counts as getting it right; ``spans`` says what the
span label type's regions are, for every task that reads them; ``counting``
turns hits and misses into rates and knows nothing of either. See
``docs/adr/0035``.
"""

from . import tasks
from .counting import Tally

#: Modules a sibling may import by path (docs/adr/0015). Each task is a
#: module of its own, so ``tasks`` and what is under it are promised whole.
PUBLIC_MODULES = frozenset(
    {"identity", "spans", "tasks", "tasks.base", "tasks.classify", "tasks.entities"}
)

__all__ = ["PUBLIC_MODULES", "Tally", "tasks"]
