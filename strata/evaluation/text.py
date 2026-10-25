"""A document's text, read as the tagger read it: UTF-8, undecodable bytes replaced.

An unreadable document is refused, not scored as empty. See ``docs/adr/0036``.
"""

from pathlib import Path


class UnreadableDocument(ValueError):
    """A document a text task needs and cannot read."""


def read_document(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise UnreadableDocument(f"Cannot read {path}: {e}") from None
