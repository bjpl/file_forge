"""TDD Tests for GUI Services Layer.

RED phase: Tests written first, defining expected behavior.
Services wrap the existing backend for GUI consumption.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path


class TestFileService:
    """Tests for FileService - wraps file operations."""

    def test_file_service_initializes_with_config(self):
        """FileService should initialize with configuration."""
        from fileforge.gui.services import FileService
        mock_config = MagicMock()
        service = FileService(config=mock_config)
        assert service.config == mock_config

    def test_file_service_provides_database_access(self):
        """FileService should provide database access."""
        from fileforge.gui.services import FileService
        service = FileService()
        assert hasattr(service, 'database')

    def test_file_service_can_scan_directory(self):
        """FileService should scan directories."""
        from fileforge.gui.services import FileService
        service = FileService()
        result = service.scan(Path('/test/dir'), recursive=True)
        assert 'files_found' in result

    def test_file_service_scan_is_async(self):
        """FileService scan should run asynchronously."""
        from fileforge.gui.services import FileService
        service = FileService()
        # Should return a future/task, not block
        task = service.scan_async(Path('/test/dir'))
        assert hasattr(task, 'result') or hasattr(task, '__await__')

    def test_file_service_emits_progress_during_scan(self):
        """FileService should emit progress during scan."""
        from fileforge.gui.services import FileService
        service = FileService()
        callback = MagicMock()
        service.progress_updated.connect(callback)
        service.scan(Path('/test/dir'))
        assert callback.called

    def test_file_service_can_organize_files(self):
        """FileService should organize files."""
        from fileforge.gui.services import FileService
        service = FileService()
        files = [Path('/test/file1.jpg'), Path('/test/file2.pdf')]
        result = service.organize(files)
        assert 'operations' in result

    def test_file_service_supports_dry_run(self):
        """FileService should support dry-run mode."""
        from fileforge.gui.services import FileService
        service = FileService()
        files = [Path('/test/file.jpg')]
        result = service.organize(files, dry_run=True)
        assert result['dry_run'] is True
        assert 'proposed_actions' in result

    def test_file_service_can_cancel_operation(self):
        """FileService should support operation cancellation."""
        from fileforge.gui.services import FileService
        service = FileService()
        task = service.scan_async(Path('/test/dir'))
        service.cancel()
        assert service.is_cancelled is True


class TestProcessingService:
    """Tests for ProcessingService - manages background processing."""

    def test_processing_service_has_thread_pool(self):
        """ProcessingService should use thread pool."""
        from fileforge.gui.services import ProcessingService
        service = ProcessingService()
        assert hasattr(service, 'thread_pool')

    def test_processing_service_can_queue_work(self):
        """ProcessingService should queue work items."""
        from fileforge.gui.services import ProcessingService
        service = ProcessingService()
        service.queue_work(lambda: None)
        assert service.pending_count >= 1

    def test_processing_service_tracks_active_workers(self):
        """ProcessingService should track active workers."""
        from fileforge.gui.services import ProcessingService
        service = ProcessingService()
        assert hasattr(service, 'active_workers')

    def test_processing_service_can_process_file(self):
        """ProcessingService should process individual files."""
        from fileforge.gui.services import ProcessingService
        service = ProcessingService()
        result = service.process_file(Path('/test/image.jpg'))
        assert result is not None

    def test_processing_service_can_batch_process(self):
        """ProcessingService should batch process files."""
        from fileforge.gui.services import ProcessingService
        service = ProcessingService()
        files = [Path(f'/test/file{i}.jpg') for i in range(10)]
        results = service.process_batch(files)
        assert len(results) == 10

    def test_processing_service_emits_completion_signal(self):
        """ProcessingService should emit signal on completion."""
        from fileforge.gui.services import ProcessingService
        service = ProcessingService()
        callback = MagicMock()
        service.processing_complete.connect(callback)
        service.process_file(Path('/test/file.jpg'))
        # After processing completes
        callback.assert_called()

    def test_processing_service_handles_errors(self):
        """ProcessingService should handle processing errors."""
        from fileforge.gui.services import ProcessingService
        service = ProcessingService()
        callback = MagicMock()
        service.error_occurred.connect(callback)
        # Process non-existent file
        service.process_file(Path('/nonexistent/file.xyz'))
        callback.assert_called()


class TestDatabaseService:
    """Tests for DatabaseService - wraps database operations."""

    def test_database_service_connects_on_init(self):
        """DatabaseService should connect to database on init."""
        from fileforge.gui.services import DatabaseService
        service = DatabaseService()
        assert service.is_connected is True

    def test_database_service_provides_stats(self):
        """DatabaseService should provide statistics."""
        from fileforge.gui.services import DatabaseService
        service = DatabaseService()
        stats = service.get_stats()
        assert 'total_files' in stats

    def test_database_service_can_query_files(self):
        """DatabaseService should query files."""
        from fileforge.gui.services import DatabaseService
        service = DatabaseService()
        results = service.query_files(tag='invoice')
        assert isinstance(results, list)

    def test_database_service_can_search_text(self):
        """DatabaseService should support text search."""
        from fileforge.gui.services import DatabaseService
        service = DatabaseService()
        results = service.search('quarterly report')
        assert isinstance(results, list)

    def test_database_service_caches_results(self):
        """DatabaseService should cache frequent queries."""
        from fileforge.gui.services import DatabaseService
        service = DatabaseService()
        # First call
        results1 = service.get_stats()
        # Second call should be cached
        results2 = service.get_stats()
        assert results1 == results2

    def test_database_service_can_invalidate_cache(self):
        """DatabaseService should invalidate cache."""
        from fileforge.gui.services import DatabaseService
        service = DatabaseService()
        service.get_stats()
        service.invalidate_cache()
        assert service.cache_size == 0

    def test_database_service_provides_face_clusters(self):
        """DatabaseService should provide face clusters."""
        from fileforge.gui.services import DatabaseService
        service = DatabaseService()
        clusters = service.get_face_clusters()
        assert isinstance(clusters, list)

    def test_database_service_provides_recent_operations(self):
        """DatabaseService should provide recent operations."""
        from fileforge.gui.services import DatabaseService
        service = DatabaseService()
        operations = service.get_recent_operations(limit=10)
        assert isinstance(operations, list)


class TestWatcherService:
    """Tests for WatcherService - wraps file watching."""

    def test_watcher_service_can_watch_directory(self):
        """WatcherService should watch directories."""
        from fileforge.gui.services import WatcherService
        service = WatcherService()
        service.watch(Path('/test/dir'))
        assert service.is_watching is True

    def test_watcher_service_can_stop_watching(self):
        """WatcherService should stop watching."""
        from fileforge.gui.services import WatcherService
        service = WatcherService()
        service.watch(Path('/test/dir'))
        service.stop()
        assert service.is_watching is False

    def test_watcher_service_emits_file_created_signal(self):
        """WatcherService should emit signal on file creation."""
        from fileforge.gui.services import WatcherService
        service = WatcherService()
        assert hasattr(service, 'file_created')

    def test_watcher_service_emits_file_modified_signal(self):
        """WatcherService should emit signal on file modification."""
        from fileforge.gui.services import WatcherService
        service = WatcherService()
        assert hasattr(service, 'file_modified')

    def test_watcher_service_supports_debounce(self):
        """WatcherService should support debounce."""
        from fileforge.gui.services import WatcherService
        service = WatcherService(debounce_seconds=2)
        assert service.debounce_seconds == 2

    def test_watcher_service_can_watch_multiple_dirs(self):
        """WatcherService should watch multiple directories."""
        from fileforge.gui.services import WatcherService
        service = WatcherService()
        service.watch(Path('/test/dir1'))
        service.watch(Path('/test/dir2'))
        assert len(service.watched_paths) == 2


class TestConfigService:
    """Tests for ConfigService - wraps configuration."""

    def test_config_service_loads_config(self):
        """ConfigService should load configuration."""
        from fileforge.gui.services import ConfigService
        service = ConfigService()
        assert service.config is not None

    def test_config_service_provides_settings(self):
        """ConfigService should provide all settings."""
        from fileforge.gui.services import ConfigService
        service = ConfigService()
        settings = service.get_all_settings()
        assert 'database' in settings
        assert 'scanning' in settings

    def test_config_service_can_update_setting(self):
        """ConfigService should update settings."""
        from fileforge.gui.services import ConfigService
        service = ConfigService()
        service.update('scanning.recursive', False)
        assert service.get('scanning.recursive') is False

    def test_config_service_can_save(self):
        """ConfigService should save configuration."""
        from fileforge.gui.services import ConfigService
        service = ConfigService()
        result = service.save()
        assert result is True

    def test_config_service_validates_before_save(self):
        """ConfigService should validate before saving."""
        from fileforge.gui.services import ConfigService
        service = ConfigService()
        service.update('processing.workers', -1)
        result, errors = service.validate()
        assert result is False

    def test_config_service_emits_change_signal(self):
        """ConfigService should emit signal on change."""
        from fileforge.gui.services import ConfigService
        service = ConfigService()
        callback = MagicMock()
        service.config_changed.connect(callback)
        service.update('scanning.recursive', False)
        callback.assert_called()
