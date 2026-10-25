"""What is being achieved, and how getting it right is counted.

A :class:`Task` reads one label type; a :class:`GenerationTask` reads the
media a generator makes. See ``docs/adr/0035`` and ``docs/adr/0054``.
"""

from . import classify, entities, mask, text_generation
from .base import (
    BaseTask,
    Document,
    Generated,
    Generation,
    GenerationTask,
    NoParams,
    Scored,
    Task,
    TaskError,
    TaskScore,
)
from .classify import Classify
from .entities import Entities
from .mask import Mask
from .text_generation import TextGeneration

__all__ = [
    "BaseTask",
    "Classify",
    "Document",
    "Entities",
    "Generated",
    "Generation",
    "GenerationTask",
    "Mask",
    "NoParams",
    "Scored",
    "Task",
    "TaskError",
    "TaskScore",
    "TextGeneration",
    "classify",
    "entities",
    "mask",
    "text_generation",
]
