"""File operations with undo support and operation tracking."""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import shutil
import json
import uuid
import time

from .exceptions import (
    FileNotFoundError,
    FileExistsError,
    PermissionError,
    BatchOperationError,
)


@dataclass
class OperationResult:
    """Result of a file operation."""
    success: bool
    operation: str
    source: Optional[str] = None
    destination: Optional[str] = None
    error: Optional[Exception] = None
    error_message: str = ""
    metadata: Optional[Dict[str, Any]] = None


class FileActions:
    """Handles file operations with tracking and undo support."""

    def __init__(self, database, trash_dir: Optional[Path] = None):
        """
        Initialize FileActions.

        Args:
            database: Database instance for tracking operations
            trash_dir: Directory for soft-deleted files
        """
        self.database = database
        self.trash_dir = trash_dir
        self.last_result: Optional[OperationResult] = None

    def rename(
        self,
        source: Path,
        new_name: str,
        overwrite: bool = False,
        batch_id: Optional[str] = None
    ) -> OperationResult:
        """
        Rename a file.

        Args:
            source: Source file path
            new_name: New filename or full path
            overwrite: Whether to overwrite if destination exists

        Returns:
            OperationResult indicating success or failure

        Raises:
            FileNotFoundError: If source doesn't exist
            FileExistsError: If destination exists and overwrite=False
        """
        source = Path(source)

        if not source.exists():
            self.last_result = OperationResult(
                success=False,
                operation='rename',
                source=str(source),
                error=FileNotFoundError(f"Source file not found: {source}"),
                error_message=f"Source file not found: {source}"
            )
            raise FileNotFoundError(f"Source file not found: {source}")

        # Determine destination path
        if Path(new_name).is_absolute():
            dest = Path(new_name)
        else:
            dest = source.parent / new_name

        # Check for conflicts
        if dest.exists() and not overwrite:
            self.last_result = OperationResult(
                success=False,
                operation='rename',
                source=str(source),
                destination=str(dest),
                error=FileExistsError(f"Destination already exists: {dest}"),
                error_message=f"Destination already exists: {dest}"
            )
            raise FileExistsError(f"Destination already exists: {dest}")

        # Record operation before executing
        old_name = source.name
        new_name_only = dest.name

        # Perform rename
        try:
            source.rename(dest)

            # Record operation
            self.database.add_operation({
                'operation_type': 'rename',
                'source_path': str(source),
                'dest_path': str(dest),
                'old_value': old_name,
                'new_value': new_name_only,
                'timestamp': time.time(),
                'batch_id': batch_id,
                'undone': False
            })

            result = OperationResult(
                success=True,
                operation='rename',
                source=str(source),
                destination=str(dest)
            )
            self.last_result = result
            return result

        except Exception as e:
            self.last_result = OperationResult(
                success=False,
                operation='rename',
                source=str(source),
                destination=str(dest),
                error=e,
                error_message=str(e)
            )
            raise

    def move(
        self,
        source: Path,
        destination: Path,
        overwrite: bool = False,
        batch_id: Optional[str] = None
    ) -> OperationResult:
        """
        Move a file to a new location.

        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite if destination exists

        Returns:
            OperationResult indicating success or failure

        Raises:
            FileNotFoundError: If source doesn't exist
            FileExistsError: If destination exists and overwrite=False
        """
        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Check destination exists - catch permission errors
        try:
            dest_exists = destination.exists()
        except OSError as e:
            # Permission denied accessing destination
            raise PermissionError(f"Permission denied: {destination}") from e

        if dest_exists and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")

        # Create destination directories if needed
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise PermissionError(f"Cannot create directory: {destination.parent}") from e

        # Perform move
        try:
            shutil.move(str(source), str(destination))

            # Record operation
            self.database.add_operation({
                'operation_type': 'move',
                'source_path': str(source),
                'dest_path': str(destination),
                'timestamp': time.time(),
                'batch_id': batch_id,
                'undone': False
            })

            result = OperationResult(
                success=True,
                operation='move',
                source=str(source),
                destination=str(destination)
            )
            self.last_result = result
            return result

        except Exception as e:
            self.last_result = OperationResult(
                success=False,
                operation='move',
                source=str(source),
                destination=str(destination),
                error=e,
                error_message=str(e)
            )
            raise

    def copy(
        self,
        source: Path,
        destination: Path,
        preserve_metadata: bool = False,
        overwrite: bool = False
    ) -> OperationResult:
        """
        Copy a file to a new location.

        Args:
            source: Source file path
            destination: Destination file path
            preserve_metadata: Whether to preserve file metadata
            overwrite: Whether to overwrite if destination exists

        Returns:
            OperationResult indicating success or failure
        """
        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")

        # Create destination directories if needed
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Perform copy
        try:
            if preserve_metadata:
                shutil.copy2(str(source), str(destination))
            else:
                shutil.copy(str(source), str(destination))

            # Record operation
            self.database.add_operation({
                'operation_type': 'copy',
                'source_path': str(source),
                'dest_path': str(destination),
                'timestamp': time.time(),
                'batch_id': None,
                'undone': False
            })

            result = OperationResult(
                success=True,
                operation='copy',
                source=str(source),
                destination=str(destination)
            )
            self.last_result = result
            return result

        except Exception as e:
            self.last_result = OperationResult(
                success=False,
                operation='copy',
                source=str(source),
                destination=str(destination),
                error=e,
                error_message=str(e)
            )
            raise

    def delete(
        self,
        source: Path,
        permanent: bool = False,
        batch_id: Optional[str] = None
    ) -> OperationResult:
        """
        Delete a file (soft delete to trash by default).

        Args:
            source: File to delete
            permanent: If True, permanently delete without trash

        Returns:
            OperationResult indicating success or failure

        Raises:
            FileNotFoundError: If source doesn't exist
        """
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        backup_path = None

        try:
            if not permanent and self.trash_dir:
                # Soft delete - move to trash
                self.trash_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time() * 1000)
                backup_name = f"{source.stem}_{timestamp}{source.suffix}"
                backup_path = self.trash_dir / backup_name
                shutil.move(str(source), str(backup_path))
            else:
                # Permanent delete
                source.unlink()

            # Record operation
            op_data = {
                'operation_type': 'delete',
                'source_path': str(source),
                'timestamp': time.time(),
                'batch_id': batch_id,
                'undone': False,
                'permanent': permanent
            }

            if backup_path:
                op_data['backup_path'] = str(backup_path)

            self.database.add_operation(op_data)

            result = OperationResult(
                success=True,
                operation='delete',
                source=str(source),
                metadata={'backup_path': str(backup_path)} if backup_path else None
            )
            self.last_result = result
            return result

        except Exception as e:
            self.last_result = OperationResult(
                success=False,
                operation='delete',
                source=str(source),
                error=e,
                error_message=str(e)
            )
            raise

    def add_tags(self, source: Path, tags: List[str], batch_id: Optional[str] = None) -> OperationResult:
        """
        Add tags to a file.

        Args:
            source: File to tag
            tags: Tags to add

        Returns:
            OperationResult indicating success or failure
        """
        source = Path(source)

        # Get current tags
        file_record = self.database.get_file(str(source))
        if file_record:
            current_tags = json.loads(file_record.get('tags', '[]'))
        else:
            current_tags = []

        old_tags = current_tags.copy()

        # Add new tags
        for tag in tags:
            if tag not in current_tags:
                current_tags.append(tag)

        new_tags_json = json.dumps(current_tags)

        # Update database
        self.database.update_file_tags(str(source), new_tags_json)

        # Record operation for undo
        self.database.add_operation({
            'operation_type': 'tag',
            'source_path': str(source),
            'old_value': json.dumps(old_tags),
            'new_value': new_tags_json,
            'timestamp': time.time(),
            'batch_id': batch_id,
            'undone': False
        })

        return OperationResult(
            success=True,
            operation='add_tags',
            source=str(source)
        )

    def remove_tags(self, source: Path, tags: List[str], batch_id: Optional[str] = None) -> OperationResult:
        """
        Remove tags from a file.

        Args:
            source: File to remove tags from
            tags: Tags to remove

        Returns:
            OperationResult indicating success or failure
        """
        source = Path(source)

        # Get current tags
        file_record = self.database.get_file(str(source))
        if file_record:
            current_tags = json.loads(file_record.get('tags', '[]'))
        else:
            current_tags = []

        old_tags = current_tags.copy()

        # Remove tags
        for tag in tags:
            if tag in current_tags:
                current_tags.remove(tag)

        new_tags_json = json.dumps(current_tags)

        # Update database
        self.database.update_file_tags(str(source), new_tags_json)

        # Record operation for undo
        self.database.add_operation({
            'operation_type': 'tag',
            'source_path': str(source),
            'old_value': json.dumps(old_tags),
            'new_value': new_tags_json,
            'timestamp': time.time(),
            'batch_id': batch_id,
            'undone': False
        })

        return OperationResult(
            success=True,
            operation='remove_tags',
            source=str(source)
        )

    def batch_rename(
        self,
        operations: List[Tuple[Path, str]],
        atomic: bool = False
    ) -> List[OperationResult]:
        """
        Perform batch rename operations.

        Args:
            operations: List of (source, new_name) tuples
            atomic: If True, rollback all on any failure

        Returns:
            List of OperationResults

        Raises:
            BatchOperationError: If atomic=True and any operation fails
        """
        batch_id = str(uuid.uuid4())
        results = []
        completed = []

        try:
            for source, new_name in operations:
                # Use batch_id for all operations
                result = self.rename(source, new_name, batch_id=batch_id)
                results.append(result)
                completed.append((source, new_name, result))

        except Exception as e:
            if atomic:
                # Rollback all completed operations
                for source, new_name, result in reversed(completed):
                    try:
                        dest = Path(result.destination)
                        if dest.exists():
                            dest.rename(source)
                    except:
                        pass  # Best effort rollback

                raise BatchOperationError(f"Batch operation failed: {e}")
            else:
                results.append(OperationResult(
                    success=False,
                    operation='rename',
                    error=e,
                    error_message=str(e)
                ))

        return results

    def batch_move(
        self,
        operations: List[Tuple[Path, Path]],
        atomic: bool = False
    ) -> List[OperationResult]:
        """
        Perform batch move operations.

        Args:
            operations: List of (source, destination) tuples
            atomic: If True, rollback all on any failure

        Returns:
            List of OperationResults
        """
        batch_id = str(uuid.uuid4())
        results = []
        completed = []

        try:
            for source, dest in operations:
                result = self.move(source, dest, batch_id=batch_id)
                results.append(result)
                completed.append((source, dest, result))

        except Exception as e:
            if atomic:
                for source, dest, result in reversed(completed):
                    try:
                        dest_path = Path(result.destination)
                        if dest_path.exists():
                            shutil.move(str(dest_path), str(source))
                    except:
                        pass

                raise BatchOperationError(f"Batch operation failed: {e}")
            else:
                results.append(OperationResult(
                    success=False,
                    operation='move',
                    error=e,
                    error_message=str(e)
                ))

        return results

    def batch_delete(
        self,
        files: List[Path],
        permanent: bool = False
    ) -> List[OperationResult]:
        """
        Perform batch delete operations.

        Args:
            files: List of files to delete
            permanent: If True, permanently delete

        Returns:
            List of OperationResults
        """
        batch_id = str(uuid.uuid4())
        results = []

        for file in files:
            try:
                result = self.delete(file, permanent=permanent, batch_id=batch_id)
                results.append(result)

            except Exception as e:
                results.append(OperationResult(
                    success=False,
                    operation='delete',
                    source=str(file),
                    error=e,
                    error_message=str(e)
                ))

        return results

    def batch_add_tags(
        self,
        files: List[Path],
        tags: List[str]
    ) -> List[OperationResult]:
        """
        Add tags to multiple files.

        Args:
            files: List of files to tag
            tags: Tags to add

        Returns:
            List of OperationResults
        """
        batch_id = str(uuid.uuid4())
        results = []

        for file in files:
            try:
                result = self.add_tags(file, tags, batch_id=batch_id)
                results.append(result)

            except Exception as e:
                results.append(OperationResult(
                    success=False,
                    operation='add_tags',
                    source=str(file),
                    error=e,
                    error_message=str(e)
                ))

        return results

    def preview_rename(self, source: Path, new_name: str) -> Dict[str, Any]:
        """
        Preview a rename operation without executing.

        Args:
            source: Source file
            new_name: Proposed new name

        Returns:
            Dictionary with preview information
        """
        source = Path(source)

        if Path(new_name).is_absolute():
            dest = Path(new_name)
        else:
            dest = source.parent / new_name

        will_succeed = source.exists() and not dest.exists()

        preview = {
            'operation': 'rename',
            'source': str(source),
            'destination': str(dest),
            'will_succeed': will_succeed
        }

        if dest.exists():
            preview['conflict'] = f"Destination already exists: {dest}"

        if not source.exists():
            preview['error'] = f"Source not found: {source}"

        return preview

    def preview_move(self, source: Path, destination: Path) -> Dict[str, Any]:
        """
        Preview a move operation without executing.

        Args:
            source: Source file
            destination: Destination path

        Returns:
            Dictionary with preview information
        """
        source = Path(source)
        destination = Path(destination)

        will_succeed = source.exists() and not destination.exists()

        preview = {
            'operation': 'move',
            'source': str(source),
            'destination': str(destination),
            'will_succeed': will_succeed
        }

        if destination.exists():
            preview['conflict'] = f"Destination already exists: {destination}"

        if not source.exists():
            preview['error'] = f"Source not found: {source}"

        return preview

    def preview_batch_rename(
        self,
        operations: List[Tuple[Path, str]]
    ) -> List[Dict[str, Any]]:
        """
        Preview batch rename operations.

        Args:
            operations: List of (source, new_name) tuples

        Returns:
            List of preview dictionaries
        """
        previews = []
        for source, new_name in operations:
            previews.append(self.preview_rename(source, new_name))
        return previews
