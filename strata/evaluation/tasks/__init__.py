"""What is being achieved, and how getting it right is counted.

A task reads one label type and asks one question of it. Several tasks can
read the same label type: entity extraction and anonymisation both read
spans, and disagree on what a mistake is. Each module names itself and the
label type it reads, and holds a :class:`Task` subclass registered under
that name.
"""

from . import classify, entities, mask
from .base import Document, NoParams, Scored, Task, TaskError, TaskScore
from .classify import Classify
from .entities import Entities
from .mask import Mask

__all__ = [
    "Classify",
    "Document",
    "Entities",
    "Mask",
    "NoParams",
    "Scored",
    "Task",
    "TaskError",
    "TaskScore",
    "classify",
    "entities",
    "mask",
]
