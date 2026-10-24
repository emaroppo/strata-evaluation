"""What is being achieved, and how getting it right is counted.

A task reads one label type and asks one question of it. Several tasks can
read the same label type: entity extraction and anonymisation both read
spans, and disagree on what a mistake is. Each module names itself and the
label type it reads, and holds a :class:`Task` subclass registered under
that name. A generator's outputs have no label type: a task scoring them
names the media it reads instead, and is a :class:`GenerationTask`
(``docs/adr/0054``).
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
