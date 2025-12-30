"""Storage module for FileForge file operations and history tracking."""

from .actions import FileActions
from .history import OperationHistory
from .exceptions import (
    FileNotFoundError,
    FileExistsError,
    PermissionError,
    BatchOperationError,
    AlreadyUndoneError,
    UndoError,
)

__all__ = [
    'FileActions',
    'OperationHistory',
    'FileNotFoundError',
    'FileExistsError',
    'PermissionError',
    'BatchOperationError',
    'AlreadyUndoneError',
    'UndoError',
]
