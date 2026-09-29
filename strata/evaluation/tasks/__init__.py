"""What is being achieved, and how getting it right is counted.

A task reads one label type and asks one question of it. Several tasks can
read the same label type: entity extraction and anonymisation both read
spans, and disagree on what a mistake is. Each module names itself and the
label type it reads.
"""

from . import classify, entities

__all__ = ["classify", "entities"]
