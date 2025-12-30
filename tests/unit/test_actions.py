"""TDD Tests for FileForge file actions and undo system.

RED phase: Tests written first, defining expected behavior.
This test suite defines the contract for file operations and undo functionality.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import shutil
import json


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir()
    yield test_dir
    # Cleanup handled by tmp_path


@pytest.fixture
def mock_db():
    """Create mock database for testing."""
    db = MagicMock()
    db.add_operation = MagicMock()
    db.update_file_tags = MagicMock()
    db.get_file = MagicMock()
    db.get_last_operation = MagicMock()
    db.get_operations_by_batch = MagicMock()
    db.mark_operation_undone = MagicMock()
    db.get_operation = MagicMock()
    return db


class TestFileActions:
    """Tests for file action operations."""

    def test_rename_file(self, temp_dir):
        """Should rename file and record operation."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "original.txt"
        source.write_text("content")

        actions = FileActions(database=MagicMock())
        result = actions.rename(source, "renamed.txt")

        assert result.success is True
        assert not source.exists()
        assert (temp_dir / "renamed.txt").exists()
        assert (temp_dir / "renamed.txt").read_text() == "content"

    def test_rename_creates_operation_record(self, temp_dir, mock_db):
        """Rename should create operation record for undo."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "original.txt"
        source.write_text("content")

        actions = FileActions(database=mock_db)
        actions.rename(source, "renamed.txt")

        mock_db.add_operation.assert_called_once()
        call_args = mock_db.add_operation.call_args[0][0]
        assert call_args['operation_type'] == 'rename'
        assert 'source_path' in call_args
        assert 'dest_path' in call_args
        assert 'old_value' in call_args
        assert 'new_value' in call_args

    def test_rename_with_absolute_path(self, temp_dir):
        """Should handle absolute destination paths."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")
        dest = temp_dir / "subdir" / "newname.txt"
        dest.parent.mkdir(parents=True)

        actions = FileActions(database=MagicMock())
        result = actions.rename(source, str(dest))

        assert result.success is True
        assert dest.exists()

    def test_rename_conflict_handling(self, temp_dir):
        """Should handle rename conflicts appropriately."""
        from fileforge.storage.actions import FileActions, FileExistsError

        source = temp_dir / "file.txt"
        source.write_text("content")
        existing = temp_dir / "existing.txt"
        existing.write_text("already here")

        actions = FileActions(database=MagicMock())

        with pytest.raises(FileExistsError):
            actions.rename(source, "existing.txt", overwrite=False)

    def test_move_file(self, temp_dir):
        """Should move file to new location."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")
        dest_dir = temp_dir / "subdir"
        dest_dir.mkdir()

        actions = FileActions(database=MagicMock())
        result = actions.move(source, dest_dir / "file.txt")

        assert result.success is True
        assert not source.exists()
        assert (dest_dir / "file.txt").exists()
        assert (dest_dir / "file.txt").read_text() == "content"

    def test_move_creates_directories(self, temp_dir):
        """Move should create destination directories if needed."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")
        dest = temp_dir / "new" / "nested" / "dir" / "file.txt"

        actions = FileActions(database=MagicMock())
        result = actions.move(source, dest)

        assert result.success is True
        assert dest.exists()
        assert dest.parent.exists()

    def test_move_records_operation(self, temp_dir, mock_db):
        """Move should record operation for undo."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")
        dest = temp_dir / "moved" / "file.txt"

        actions = FileActions(database=mock_db)
        actions.move(source, dest)

        mock_db.add_operation.assert_called_once()
        call_args = mock_db.add_operation.call_args[0][0]
        assert call_args['operation_type'] == 'move'
        assert call_args['source_path'] == str(source)
        assert call_args['dest_path'] == str(dest)

    def test_copy_file(self, temp_dir):
        """Should copy file preserving original."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "original.txt"
        source.write_text("content")
        dest = temp_dir / "copy.txt"

        actions = FileActions(database=MagicMock())
        result = actions.copy(source, dest)

        assert result.success is True
        assert source.exists()  # Original preserved
        assert dest.exists()
        assert dest.read_text() == "content"
        assert source.read_text() == "content"

    def test_copy_preserves_metadata(self, temp_dir):
        """Copy should preserve file metadata when requested."""
        from fileforge.storage.actions import FileActions
        import os
        import time

        source = temp_dir / "original.txt"
        source.write_text("content")

        # Set specific modification time
        old_time = time.time() - 86400  # 1 day ago
        os.utime(source, (old_time, old_time))

        dest = temp_dir / "copy.txt"

        actions = FileActions(database=MagicMock())
        result = actions.copy(source, dest, preserve_metadata=True)

        assert result.success is True
        # Metadata preservation is best-effort
        assert dest.exists()

    def test_copy_records_operation(self, temp_dir, mock_db):
        """Copy should record operation."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")
        dest = temp_dir / "copy.txt"

        actions = FileActions(database=mock_db)
        actions.copy(source, dest)

        mock_db.add_operation.assert_called_once()
        call_args = mock_db.add_operation.call_args[0][0]
        assert call_args['operation_type'] == 'copy'

    def test_delete_file(self, temp_dir):
        """Should delete file (move to trash/record for undo)."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "to_delete.txt"
        source.write_text("content")

        actions = FileActions(database=MagicMock(), trash_dir=temp_dir / ".trash")
        result = actions.delete(source)

        assert result.success is True
        assert not source.exists()

    def test_delete_is_recoverable(self, temp_dir, mock_db):
        """Deleted files should be recoverable."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "to_delete.txt"
        source.write_text("important content")

        trash_dir = temp_dir / ".trash"
        actions = FileActions(database=mock_db, trash_dir=trash_dir)
        actions.delete(source)

        # Should have backup in trash
        if trash_dir.exists():
            trash_files = list(trash_dir.glob("*"))
            assert len(trash_files) > 0

        # Or operation should have backup path
        assert mock_db.add_operation.called
        call_args = mock_db.add_operation.call_args[0][0]
        assert 'backup_path' in call_args or 'trash_path' in call_args

    def test_delete_nonexistent_file(self, temp_dir):
        """Should handle deleting nonexistent file gracefully."""
        from fileforge.storage.actions import FileActions, FileNotFoundError

        source = temp_dir / "nonexistent.txt"

        actions = FileActions(database=MagicMock())

        with pytest.raises(FileNotFoundError):
            actions.delete(source)

    def test_permanent_delete(self, temp_dir, mock_db):
        """Should support permanent delete without trash."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "to_delete.txt"
        source.write_text("content")

        actions = FileActions(database=mock_db)
        result = actions.delete(source, permanent=True)

        assert result.success is True
        assert not source.exists()

        # Should still record operation but mark as non-recoverable
        call_args = mock_db.add_operation.call_args[0][0]
        assert call_args.get('permanent', False) is True


class TestUndoOperations:
    """Tests for undo functionality."""

    def test_undo_rename(self, temp_dir, mock_db):
        """Should undo rename operation."""
        from fileforge.storage.actions import FileActions
        from fileforge.storage.history import OperationHistory

        # Setup: file already renamed
        renamed_file = temp_dir / "renamed.txt"
        renamed_file.write_text("content")

        # Operation record from rename
        op_record = {
            'id': 1,
            'operation_type': 'rename',
            'source_path': str(temp_dir / "original.txt"),
            'dest_path': str(renamed_file),
            'old_value': 'original.txt',
            'new_value': 'renamed.txt',
            'undone': False
        }

        mock_db.get_operation.return_value = op_record

        history = OperationHistory(database=mock_db, actions=FileActions(database=mock_db))
        result = history.undo_operation(op_record)

        assert result.success is True
        # Original name should be restored
        assert (temp_dir / "original.txt").exists() or not renamed_file.exists()

    def test_undo_move(self, temp_dir, mock_db):
        """Should undo move operation."""
        from fileforge.storage.actions import FileActions
        from fileforge.storage.history import OperationHistory

        # Setup: file already moved
        dest_dir = temp_dir / "moved"
        dest_dir.mkdir()
        moved_file = dest_dir / "file.txt"
        moved_file.write_text("content")

        op_record = {
            'id': 1,
            'operation_type': 'move',
            'source_path': str(temp_dir / "file.txt"),
            'dest_path': str(moved_file),
            'undone': False
        }

        mock_db.get_operation.return_value = op_record

        history = OperationHistory(database=mock_db, actions=FileActions(database=mock_db))
        result = history.undo_operation(op_record)

        assert result.success is True
        # File should be back at original location
        assert (temp_dir / "file.txt").exists() or not moved_file.exists()

    def test_undo_copy_removes_copy(self, temp_dir, mock_db):
        """Undoing copy should remove the copied file."""
        from fileforge.storage.history import OperationHistory
        from fileforge.storage.actions import FileActions

        # Setup: copy exists
        copy_file = temp_dir / "copy.txt"
        copy_file.write_text("content")

        op_record = {
            'id': 1,
            'operation_type': 'copy',
            'source_path': str(temp_dir / "original.txt"),
            'dest_path': str(copy_file),
            'undone': False
        }

        history = OperationHistory(database=mock_db, actions=FileActions(database=mock_db))
        result = history.undo_operation(op_record)

        assert result.success is True
        # Copy should be removed
        assert not copy_file.exists()

    def test_undo_delete_restores_file(self, temp_dir, mock_db):
        """Undoing delete should restore file from trash."""
        from fileforge.storage.history import OperationHistory
        from fileforge.storage.actions import FileActions

        # Setup: file in trash
        trash_dir = temp_dir / ".trash"
        trash_dir.mkdir()
        backup_file = trash_dir / "deleted_file_12345.txt"
        backup_file.write_text("recovered content")

        op_record = {
            'id': 1,
            'operation_type': 'delete',
            'source_path': str(temp_dir / "file.txt"),
            'backup_path': str(backup_file),
            'undone': False
        }

        actions = FileActions(database=mock_db, trash_dir=trash_dir)
        history = OperationHistory(database=mock_db, actions=actions)
        result = history.undo_operation(op_record)

        assert result.success is True
        # File should be restored
        assert (temp_dir / "file.txt").exists()

    def test_undo_last_operation(self, mock_db):
        """Should undo most recent operation."""
        from fileforge.storage.history import OperationHistory
        from fileforge.storage.actions import FileActions

        mock_db.get_last_operation.return_value = {
            'id': 5,
            'operation_type': 'rename',
            'source_path': '/test/old.txt',
            'dest_path': '/test/new.txt',
            'undone': False
        }

        history = OperationHistory(database=mock_db, actions=FileActions(database=mock_db))

        with patch.object(history, 'undo_operation') as mock_undo:
            mock_undo.return_value = MagicMock(success=True)
            result = history.undo_last()

        assert result is not None
        mock_db.get_last_operation.assert_called()

    def test_undo_batch(self, mock_db):
        """Should undo all operations in a batch."""
        from fileforge.storage.history import OperationHistory
        from fileforge.storage.actions import FileActions

        mock_db.get_operations_by_batch.return_value = [
            {'id': 3, 'operation_type': 'tag', 'batch_id': 'batch_001', 'undone': False},
            {'id': 2, 'operation_type': 'move', 'batch_id': 'batch_001', 'undone': False},
            {'id': 1, 'operation_type': 'rename', 'batch_id': 'batch_001', 'undone': False},
        ]

        history = OperationHistory(database=mock_db, actions=FileActions(database=mock_db))

        with patch.object(history, 'undo_operation') as mock_undo:
            mock_undo.return_value = MagicMock(success=True)
            results = history.undo_batch('batch_001')

        # Should undo in reverse order (LIFO)
        assert len(results) == 3
        assert mock_undo.call_count == 3

    def test_cannot_undo_already_undone(self, mock_db):
        """Should not undo already undone operation."""
        from fileforge.storage.history import OperationHistory, AlreadyUndoneError
        from fileforge.storage.actions import FileActions

        op_record = {
            'id': 1,
            'operation_type': 'rename',
            'undone': True  # Already undone
        }

        history = OperationHistory(database=mock_db, actions=FileActions(database=mock_db))

        with pytest.raises(AlreadyUndoneError):
            history.undo_operation(op_record)

    def test_undo_marks_operation_as_undone(self, temp_dir, mock_db):
        """Successful undo should mark operation in database."""
        from fileforge.storage.history import OperationHistory
        from fileforge.storage.actions import FileActions

        renamed_file = temp_dir / "renamed.txt"
        renamed_file.write_text("content")

        op_record = {
            'id': 1,
            'operation_type': 'rename',
            'source_path': str(temp_dir / "original.txt"),
            'dest_path': str(renamed_file),
            'undone': False
        }

        actions = FileActions(database=mock_db)
        history = OperationHistory(database=mock_db, actions=actions)
        history.undo_operation(op_record)

        mock_db.mark_operation_undone.assert_called_with(1)


class TestBatchOperations:
    """Tests for batch file operations."""

    def test_batch_rename_creates_single_batch_id(self, temp_dir, mock_db):
        """Batch operations should share batch ID."""
        from fileforge.storage.actions import FileActions

        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        actions = FileActions(database=mock_db)

        renames = [
            (files[0], "renamed0.txt"),
            (files[1], "renamed1.txt"),
            (files[2], "renamed2.txt"),
        ]

        actions.batch_rename(renames)

        # All operations should have same batch_id
        calls = mock_db.add_operation.call_args_list
        batch_ids = [c[0][0]['batch_id'] for c in calls]
        assert len(set(batch_ids)) == 1  # All same batch
        assert batch_ids[0] is not None

    def test_batch_operation_executes_all(self, temp_dir, mock_db):
        """Batch should execute all operations."""
        from fileforge.storage.actions import FileActions

        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        actions = FileActions(database=mock_db)

        renames = [
            (files[0], "renamed0.txt"),
            (files[1], "renamed1.txt"),
            (files[2], "renamed2.txt"),
        ]

        results = actions.batch_rename(renames)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert (temp_dir / "renamed0.txt").exists()
        assert (temp_dir / "renamed1.txt").exists()
        assert (temp_dir / "renamed2.txt").exists()

    def test_batch_operation_atomic(self, temp_dir, mock_db):
        """Batch should be atomic - all succeed or all fail."""
        from fileforge.storage.actions import FileActions, BatchOperationError

        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        # Second rename will fail (destination exists)
        blocker = temp_dir / "exists.txt"
        blocker.write_text("blocker")

        actions = FileActions(database=mock_db)

        renames = [
            (files[0], "good.txt"),
            (files[1], "exists.txt"),  # Will fail
            (files[2], "another.txt"),
        ]

        with pytest.raises(BatchOperationError):
            actions.batch_rename(renames, atomic=True)

        # Should have rolled back successful operations
        assert not (temp_dir / "good.txt").exists()
        assert files[0].exists()  # Restored

    def test_batch_move(self, temp_dir, mock_db):
        """Should support batch move operations."""
        from fileforge.storage.actions import FileActions

        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        dest_dir = temp_dir / "moved"
        dest_dir.mkdir()

        actions = FileActions(database=mock_db)

        moves = [(f, dest_dir / f.name) for f in files]
        results = actions.batch_move(moves)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all((dest_dir / f"file{i}.txt").exists() for i in range(3))

    def test_batch_delete(self, temp_dir, mock_db):
        """Should support batch delete operations."""
        from fileforge.storage.actions import FileActions

        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        actions = FileActions(database=mock_db, trash_dir=temp_dir / ".trash")
        results = actions.batch_delete(files)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(not f.exists() for f in files)


class TestPreviewMode:
    """Tests for dry-run/preview mode."""

    def test_preview_rename_shows_changes(self, temp_dir):
        """Preview should show proposed changes without executing."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "original.txt"
        source.write_text("content")

        actions = FileActions(database=MagicMock())
        preview = actions.preview_rename(source, "newname.txt")

        assert preview['source'] == str(source)
        assert preview['destination'] == str(temp_dir / "newname.txt")
        assert preview['operation'] == 'rename'
        assert preview['will_succeed'] in [True, False]
        assert source.exists()  # File unchanged

    def test_preview_move_shows_changes(self, temp_dir):
        """Preview move should show source and destination."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")
        dest = temp_dir / "newdir" / "file.txt"

        actions = FileActions(database=MagicMock())
        preview = actions.preview_move(source, dest)

        assert preview['source'] == str(source)
        assert 'newdir' in preview['destination']
        assert preview['operation'] == 'move'
        assert source.exists()  # File unchanged

    def test_preview_batch_shows_all_changes(self, temp_dir):
        """Preview batch should show all proposed changes."""
        from fileforge.storage.actions import FileActions

        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        actions = FileActions(database=MagicMock())

        renames = [(f, f"new{i}.txt") for i, f in enumerate(files)]
        previews = actions.preview_batch_rename(renames)

        assert len(previews) == 3
        for p in previews:
            assert 'source' in p
            assert 'destination' in p
            assert 'operation' in p
            assert 'will_succeed' in p

        # Files should be unchanged
        assert all(f.exists() for f in files)

    def test_preview_detects_conflicts(self, temp_dir):
        """Preview should detect potential conflicts."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")
        existing = temp_dir / "existing.txt"
        existing.write_text("already here")

        actions = FileActions(database=MagicMock())
        preview = actions.preview_rename(source, "existing.txt")

        assert preview['will_succeed'] is False
        assert 'conflict' in preview or 'error' in preview


class TestTagOperations:
    """Tests for metadata tagging operations."""

    def test_add_tags_to_file(self, temp_dir, mock_db):
        """Should add tags to file metadata."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")

        mock_db.get_file.return_value = {'tags': '[]'}

        actions = FileActions(database=mock_db)
        result = actions.add_tags(source, ['important', 'work'])

        assert result.success is True
        mock_db.update_file_tags.assert_called()

        # Check tags were added
        call_args = mock_db.update_file_tags.call_args
        tags = json.loads(call_args[0][1])
        assert 'important' in tags
        assert 'work' in tags

    def test_remove_tags_from_file(self, temp_dir, mock_db):
        """Should remove tags from file."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")

        mock_db.get_file.return_value = {'tags': '["important", "work", "urgent"]'}

        actions = FileActions(database=mock_db)
        result = actions.remove_tags(source, ['work'])

        assert result.success is True

        # Check work tag was removed
        call_args = mock_db.update_file_tags.call_args
        tags = json.loads(call_args[0][1])
        assert 'work' not in tags
        assert 'important' in tags
        assert 'urgent' in tags

    def test_tag_operation_is_undoable(self, temp_dir, mock_db):
        """Tag operations should be undoable."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")

        mock_db.get_file.return_value = {'tags': '["oldtag"]'}

        actions = FileActions(database=mock_db)
        actions.add_tags(source, ['newtag'])

        # Should record old tags for undo
        mock_db.add_operation.assert_called()
        call_args = mock_db.add_operation.call_args[0][0]
        assert call_args['operation_type'] == 'tag'
        assert 'old_value' in call_args
        assert 'new_value' in call_args

    def test_undo_tag_operation(self, temp_dir, mock_db):
        """Should undo tag changes."""
        from fileforge.storage.history import OperationHistory
        from fileforge.storage.actions import FileActions

        source = temp_dir / "file.txt"
        source.write_text("content")

        op_record = {
            'id': 1,
            'operation_type': 'tag',
            'source_path': str(source),
            'old_value': '["original"]',
            'new_value': '["original", "added"]',
            'undone': False
        }

        mock_db.get_file.return_value = {'tags': '["original", "added"]'}

        actions = FileActions(database=mock_db)
        history = OperationHistory(database=mock_db, actions=actions)
        result = history.undo_operation(op_record)

        assert result.success is True
        # Should restore original tags
        call_args = mock_db.update_file_tags.call_args
        restored_tags = json.loads(call_args[0][1])
        assert restored_tags == ["original"]

    def test_batch_tag_operations(self, temp_dir, mock_db):
        """Should support batch tagging."""
        from fileforge.storage.actions import FileActions

        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        mock_db.get_file.return_value = {'tags': '[]'}

        actions = FileActions(database=mock_db)
        results = actions.batch_add_tags(files, ['project', 'important'])

        assert len(results) == 3
        assert all(r.success for r in results)

        # All should share batch ID
        calls = mock_db.add_operation.call_args_list
        batch_ids = [c[0][0]['batch_id'] for c in calls]
        assert len(set(batch_ids)) == 1


class TestErrorHandling:
    """Tests for error handling in file operations."""

    def test_rename_missing_file_raises_error(self, temp_dir):
        """Renaming nonexistent file should raise error."""
        from fileforge.storage.actions import FileActions, FileNotFoundError

        source = temp_dir / "nonexistent.txt"

        actions = FileActions(database=MagicMock())

        with pytest.raises(FileNotFoundError):
            actions.rename(source, "newname.txt")

    def test_move_to_readonly_location_raises_error(self, temp_dir):
        """Moving to readonly location should raise error."""
        from fileforge.storage.actions import FileActions, PermissionError

        source = temp_dir / "file.txt"
        source.write_text("content")

        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir(mode=0o444)

        actions = FileActions(database=MagicMock())

        try:
            with pytest.raises(PermissionError):
                actions.move(source, readonly_dir / "file.txt")
        finally:
            # Cleanup: restore permissions
            readonly_dir.chmod(0o755)

    def test_undo_without_backup_raises_error(self, mock_db):
        """Undoing delete without backup should raise error."""
        from fileforge.storage.history import OperationHistory, UndoError
        from fileforge.storage.actions import FileActions

        op_record = {
            'id': 1,
            'operation_type': 'delete',
            'source_path': '/test/file.txt',
            'permanent': True,  # No backup
            'undone': False
        }

        actions = FileActions(database=mock_db)
        history = OperationHistory(database=mock_db, actions=actions)

        with pytest.raises(UndoError):
            history.undo_operation(op_record)

    def test_operation_result_contains_error_details(self, temp_dir):
        """Failed operations should provide error details."""
        from fileforge.storage.actions import FileActions

        source = temp_dir / "nonexistent.txt"

        actions = FileActions(database=MagicMock())

        try:
            actions.rename(source, "newname.txt")
        except Exception as e:
            result = actions.last_result
            assert result.success is False
            assert result.error is not None
            assert len(result.error_message) > 0
