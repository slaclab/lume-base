from __future__ import annotations


class ReadOnlyError(TypeError):
    """Raised when a write is attempted on a read-only variable."""
