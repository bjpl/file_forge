"""Operation history and undo functionality for FileForge."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import shutil
import json

from .exceptions import AlreadyUndoneError, UndoError


class OperationHistory:
    """Manages operation history and undo/redo functionality."""

    def __init__(self, database, actions):
        """
        Initialize OperationHistory.

        Args:
            database: Database instance for querying operations
            actions: FileActions instance for performing undo operations
        """
        self.database = database
        self.actions = actions

    def undo_operation(self, operation: Dict[str, Any]):
        """
        Undo a specific operation.

        Args:
            operation: Operation record from database

        Returns:
            OperationResult indicating success or failure

        Raises:
            AlreadyUndoneError: If operation was already undone
            UndoError: If operation cannot be undone
        """
        if operation.get('undone', False):
            raise AlreadyUndoneError(f"Operation {operation['id']} already undone")

        op_type = operation['operation_type']

        try:
            if op_type == 'rename':
                result = self._undo_rename(operation)
            elif op_type == 'move':
                result = self._undo_move(operation)
            elif op_type == 'copy':
                result = self._undo_copy(operation)
            elif op_type == 'delete':
                result = self._undo_delete(operation)
            elif op_type == 'tag':
                result = self._undo_tag(operation)
            else:
                raise UndoError(f"Unknown operation type: {op_type}")

            if result.success:
                # Mark operation as undone
                self.database.mark_operation_undone(operation['id'])

            return result

        except Exception as e:
            raise UndoError(f"Failed to undo operation: {e}")

    def _undo_rename(self, operation: Dict[str, Any]):
        """Undo a rename operation by renaming back."""
        from .actions import OperationResult

        dest_path = Path(operation['dest_path'])
        source_path = Path(operation['source_path'])

        if dest_path.exists():
            dest_path.rename(source_path)
            return OperationResult(
                success=True,
                operation='undo_rename',
                source=str(dest_path),
                destination=str(source_path)
            )
        else:
            return OperationResult(
                success=False,
                operation='undo_rename',
                error_message=f"File not found: {dest_path}"
            )

    def _undo_move(self, operation: Dict[str, Any]):
        """Undo a move operation by moving back."""
        from .actions import OperationResult

        dest_path = Path(operation['dest_path'])
        source_path = Path(operation['source_path'])

        if dest_path.exists():
            # Create source directory if needed
            source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest_path), str(source_path))
            return OperationResult(
                success=True,
                operation='undo_move',
                source=str(dest_path),
                destination=str(source_path)
            )
        else:
            return OperationResult(
                success=False,
                operation='undo_move',
                error_message=f"File not found: {dest_path}"
            )

    def _undo_copy(self, operation: Dict[str, Any]):
        """Undo a copy operation by removing the copied file."""
        from .actions import OperationResult

        dest_path = Path(operation['dest_path'])

        if dest_path.exists():
            dest_path.unlink()
            return OperationResult(
                success=True,
                operation='undo_copy',
                source=str(dest_path)
            )
        else:
            return OperationResult(
                success=False,
                operation='undo_copy',
                error_message=f"Copy not found: {dest_path}"
            )

    def _undo_delete(self, operation: Dict[str, Any]):
        """Undo a delete operation by restoring from trash."""
        from .actions import OperationResult

        if operation.get('permanent', False):
            raise UndoError("Cannot undo permanent delete")

        backup_path = operation.get('backup_path')
        if not backup_path:
            raise UndoError("No backup path found for delete operation")

        backup_path = Path(backup_path)
        source_path = Path(operation['source_path'])

        if backup_path.exists():
            # Create parent directory if needed
            source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_path), str(source_path))
            return OperationResult(
                success=True,
                operation='undo_delete',
                source=str(backup_path),
                destination=str(source_path)
            )
        else:
            return OperationResult(
                success=False,
                operation='undo_delete',
                error_message=f"Backup not found: {backup_path}"
            )

    def _undo_tag(self, operation: Dict[str, Any]):
        """Undo a tag operation by restoring old tags."""
        from .actions import OperationResult

        source_path = operation['source_path']
        old_value = operation['old_value']

        # Restore old tags
        self.database.update_file_tags(source_path, old_value)

        return OperationResult(
            success=True,
            operation='undo_tag',
            source=source_path
        )

    def undo_last(self):
        """
        Undo the most recent operation.

        Returns:
            OperationResult or None if no operations to undo
        """
        last_op = self.database.get_last_operation()

        if not last_op:
            return None

        if last_op.get('undone', False):
            return None

        return self.undo_operation(last_op)

    def undo_batch(self, batch_id: str) -> List:
        """
        Undo all operations in a batch (in reverse order).

        Args:
            batch_id: Batch identifier

        Returns:
            List of OperationResults
        """
        operations = self.database.get_operations_by_batch(batch_id)

        # Reverse order for LIFO undo
        operations = reversed(operations)

        results = []
        for operation in operations:
            if not operation.get('undone', False):
                result = self.undo_operation(operation)
                results.append(result)

        return results
