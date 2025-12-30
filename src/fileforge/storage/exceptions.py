"""Custom exceptions for FileForge storage operations."""


class FileNotFoundError(Exception):
    """Raised when a file operation targets a nonexistent file."""
    pass


class FileExistsError(Exception):
    """Raised when a file operation would overwrite an existing file."""
    pass


class PermissionError(Exception):
    """Raised when a file operation lacks necessary permissions."""
    pass


class BatchOperationError(Exception):
    """Raised when a batch operation fails and needs to be rolled back."""
    pass


class AlreadyUndoneError(Exception):
    """Raised when trying to undo an operation that was already undone."""
    pass


class UndoError(Exception):
    """Raised when an undo operation cannot be completed."""
    pass
