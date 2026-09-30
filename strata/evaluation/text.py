"""A document's text, read the way the offsets into it were made.

Span offsets index the text a tagger read, so scoring against them reads it
the same way: UTF-8, with undecodable bytes replaced, through
``Path.read_text``. A document that cannot be read is refused, not scored
as empty: an empty text would turn every entity in it into a miss and say
nothing about why (``docs/adr/0036``).
"""

from pathlib import Path


class UnreadableDocument(ValueError):
    """A document a text task needs and cannot read."""


def read_document(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise UnreadableDocument(f"Cannot read {path}: {e}") from None
