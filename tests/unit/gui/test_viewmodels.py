"""TDD Tests for GUI ViewModels.

RED phase: Tests written first, defining expected behavior.
ViewModels follow MVVM pattern - they expose data and commands for views.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from datetime import datetime


class TestDashboardViewModel:
    """Tests for Dashboard view model."""

    def test_dashboard_vm_has_statistics(self):
        """DashboardViewModel should expose statistics."""
        from fileforge.gui.viewmodels import DashboardViewModel
        vm = DashboardViewModel()
        assert hasattr(vm, 'total_files')
        assert hasattr(vm, 'total_faces')
        assert hasattr(vm, 'total_objects')
        assert hasattr(vm, 'database_size')

    def test_dashboard_vm_loads_stats_from_database(self):
        """DashboardViewModel should load stats from database."""
        from fileforge.gui.viewmodels import DashboardViewModel
        mock_db = MagicMock()
        mock_db.get_stats.return_value = {
            'total_files': 1500,
            'total_faces': 230,
            'detected_objects': 4500,
            'database_size_mb': 45.2
        }
        vm = DashboardViewModel(database=mock_db)
        vm.refresh()
        assert vm.total_files == 1500
        assert vm.total_faces == 230

    def test_dashboard_vm_has_recent_activity(self):
        """DashboardViewModel should expose recent activity."""
        from fileforge.gui.viewmodels import DashboardViewModel
        vm = DashboardViewModel()
        assert hasattr(vm, 'recent_activity')
        assert isinstance(vm.recent_activity, list)

    def test_dashboard_vm_has_quick_actions(self):
        """DashboardViewModel should expose quick actions."""
        from fileforge.gui.viewmodels import DashboardViewModel
        vm = DashboardViewModel()
        assert hasattr(vm, 'quick_actions')
        assert 'scan' in [a['id'] for a in vm.quick_actions]
        assert 'organize' in [a['id'] for a in vm.quick_actions]

    def test_dashboard_vm_emits_signal_on_refresh(self):
        """DashboardViewModel should emit signal when data refreshed."""
        from fileforge.gui.viewmodels import DashboardViewModel
        vm = DashboardViewModel()
        callback = MagicMock()
        vm.data_changed.connect(callback)
        vm.refresh()
        callback.assert_called_once()

    def test_dashboard_vm_has_processing_status(self):
        """DashboardViewModel should show if processing is active."""
        from fileforge.gui.viewmodels import DashboardViewModel
        vm = DashboardViewModel()
        assert hasattr(vm, 'is_processing')
        assert isinstance(vm.is_processing, bool)


class TestFileBrowserViewModel:
    """Tests for File Browser view model."""

    def test_browser_vm_has_current_path(self):
        """FileBrowserViewModel should track current path."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        assert hasattr(vm, 'current_path')
        assert isinstance(vm.current_path, Path)

    def test_browser_vm_can_navigate(self):
        """FileBrowserViewModel should allow navigation."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        vm.navigate_to(Path('/home/user/Documents'))
        assert vm.current_path == Path('/home/user/Documents')

    def test_browser_vm_has_file_list(self):
        """FileBrowserViewModel should expose file list."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        assert hasattr(vm, 'files')
        assert isinstance(vm.files, list)

    def test_browser_vm_can_filter_by_type(self):
        """FileBrowserViewModel should filter files by type."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        vm.set_filter('images')
        assert vm.current_filter == 'images'

    def test_browser_vm_supports_selection(self):
        """FileBrowserViewModel should support file selection."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        vm.select_file(Path('/test/file.jpg'))
        assert Path('/test/file.jpg') in vm.selected_files

    def test_browser_vm_supports_multi_selection(self):
        """FileBrowserViewModel should support multiple selection."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        files = [Path('/test/file1.jpg'), Path('/test/file2.jpg')]
        vm.select_files(files)
        assert len(vm.selected_files) == 2

    def test_browser_vm_has_view_mode(self):
        """FileBrowserViewModel should support view modes."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        assert hasattr(vm, 'view_mode')
        assert vm.view_mode in ['list', 'grid', 'details']

    def test_browser_vm_can_change_view_mode(self):
        """FileBrowserViewModel should allow changing view mode."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        vm.set_view_mode('grid')
        assert vm.view_mode == 'grid'

    def test_browser_vm_supports_drag_drop(self):
        """FileBrowserViewModel should handle drag-drop."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        dropped_files = ['/external/file1.jpg', '/external/file2.pdf']
        result = vm.handle_drop(dropped_files)
        assert result is True

    def test_browser_vm_emits_signal_on_path_change(self):
        """FileBrowserViewModel should emit signal on navigation."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        callback = MagicMock()
        vm.path_changed.connect(callback)
        vm.navigate_to(Path('/new/path'))
        callback.assert_called_once()

    def test_browser_vm_has_breadcrumbs(self):
        """FileBrowserViewModel should provide breadcrumb navigation."""
        from fileforge.gui.viewmodels import FileBrowserViewModel
        vm = FileBrowserViewModel()
        vm.navigate_to(Path('/home/user/Documents/Work'))
        assert hasattr(vm, 'breadcrumbs')
        assert len(vm.breadcrumbs) >= 3


class TestProcessingViewModel:
    """Tests for Processing Queue view model."""

    def test_processing_vm_has_queue(self):
        """ProcessingViewModel should expose processing queue."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        assert hasattr(vm, 'queue')
        assert isinstance(vm.queue, list)

    def test_processing_vm_can_add_to_queue(self):
        """ProcessingViewModel should allow adding files to queue."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        vm.add_to_queue([Path('/test/file.jpg')])
        assert len(vm.queue) == 1

    def test_processing_vm_has_current_file(self):
        """ProcessingViewModel should track currently processing file."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        assert hasattr(vm, 'current_file')

    def test_processing_vm_has_progress(self):
        """ProcessingViewModel should track progress."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        assert hasattr(vm, 'progress')
        assert 0 <= vm.progress <= 100

    def test_processing_vm_can_start_processing(self):
        """ProcessingViewModel should start processing."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        vm.add_to_queue([Path('/test/file.jpg')])
        vm.start()
        assert vm.is_running is True

    def test_processing_vm_can_pause(self):
        """ProcessingViewModel should support pause."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        vm.add_to_queue([Path('/test/file.jpg')])
        vm.start()
        vm.pause()
        assert vm.is_paused is True

    def test_processing_vm_can_cancel(self):
        """ProcessingViewModel should support cancellation."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        vm.add_to_queue([Path('/test/file.jpg')])
        vm.start()
        vm.cancel()
        assert vm.is_running is False

    def test_processing_vm_has_log(self):
        """ProcessingViewModel should maintain processing log."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        assert hasattr(vm, 'log_entries')
        assert isinstance(vm.log_entries, list)

    def test_processing_vm_emits_progress_signal(self):
        """ProcessingViewModel should emit progress signals."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        callback = MagicMock()
        vm.progress_changed.connect(callback)
        vm._update_progress(50)
        callback.assert_called_once_with(50)

    def test_processing_vm_has_eta(self):
        """ProcessingViewModel should estimate time remaining."""
        from fileforge.gui.viewmodels import ProcessingViewModel
        vm = ProcessingViewModel()
        assert hasattr(vm, 'eta_seconds')


class TestResultsViewModel:
    """Tests for Results Gallery view model."""

    def test_results_vm_has_processed_files(self):
        """ResultsViewModel should expose processed files."""
        from fileforge.gui.viewmodels import ResultsViewModel
        vm = ResultsViewModel()
        assert hasattr(vm, 'files')
        assert isinstance(vm.files, list)

    def test_results_vm_can_filter_by_category(self):
        """ResultsViewModel should filter by category."""
        from fileforge.gui.viewmodels import ResultsViewModel
        vm = ResultsViewModel()
        vm.set_category_filter('Documents')
        assert vm.category_filter == 'Documents'

    def test_results_vm_can_filter_by_tag(self):
        """ResultsViewModel should filter by tag."""
        from fileforge.gui.viewmodels import ResultsViewModel
        vm = ResultsViewModel()
        vm.set_tag_filter('invoice')
        assert 'invoice' in vm.tag_filters

    def test_results_vm_can_search(self):
        """ResultsViewModel should support text search."""
        from fileforge.gui.viewmodels import ResultsViewModel
        vm = ResultsViewModel()
        vm.search('quarterly report')
        assert vm.search_query == 'quarterly report'

    def test_results_vm_has_undo_capability(self):
        """ResultsViewModel should support undo."""
        from fileforge.gui.viewmodels import ResultsViewModel
        vm = ResultsViewModel()
        assert hasattr(vm, 'can_undo')
        assert hasattr(vm, 'undo')

    def test_results_vm_can_undo_last_operation(self):
        """ResultsViewModel should undo last operation."""
        from fileforge.gui.viewmodels import ResultsViewModel
        mock_history = MagicMock()
        mock_history.undo_last.return_value = True
        vm = ResultsViewModel(history=mock_history)
        result = vm.undo()
        assert result is True
        mock_history.undo_last.assert_called_once()

    def test_results_vm_has_sort_options(self):
        """ResultsViewModel should support sorting."""
        from fileforge.gui.viewmodels import ResultsViewModel
        vm = ResultsViewModel()
        assert hasattr(vm, 'sort_by')
        assert hasattr(vm, 'sort_order')

    def test_results_vm_can_export(self):
        """ResultsViewModel should support export."""
        from fileforge.gui.viewmodels import ResultsViewModel
        vm = ResultsViewModel()
        assert hasattr(vm, 'export_json')
        assert hasattr(vm, 'export_csv')


class TestSettingsViewModel:
    """Tests for Settings view model."""

    def test_settings_vm_has_config(self):
        """SettingsViewModel should expose configuration."""
        from fileforge.gui.viewmodels import SettingsViewModel
        vm = SettingsViewModel()
        assert hasattr(vm, 'config')

    def test_settings_vm_has_sections(self):
        """SettingsViewModel should organize settings in sections."""
        from fileforge.gui.viewmodels import SettingsViewModel
        vm = SettingsViewModel()
        assert hasattr(vm, 'sections')
        section_names = [s['name'] for s in vm.sections]
        assert 'Database' in section_names
        assert 'Scanning' in section_names
        assert 'Processing' in section_names

    def test_settings_vm_can_update_setting(self):
        """SettingsViewModel should allow updating settings."""
        from fileforge.gui.viewmodels import SettingsViewModel
        vm = SettingsViewModel()
        vm.update_setting('scanning.recursive', False)
        assert vm.get_setting('scanning.recursive') is False

    def test_settings_vm_can_save(self):
        """SettingsViewModel should save configuration."""
        from fileforge.gui.viewmodels import SettingsViewModel
        vm = SettingsViewModel()
        vm.update_setting('scanning.recursive', False)
        result = vm.save()
        assert result is True

    def test_settings_vm_can_reset_to_defaults(self):
        """SettingsViewModel should reset to defaults."""
        from fileforge.gui.viewmodels import SettingsViewModel
        vm = SettingsViewModel()
        vm.update_setting('scanning.recursive', False)
        vm.reset_to_defaults()
        assert vm.get_setting('scanning.recursive') is True

    def test_settings_vm_validates_settings(self):
        """SettingsViewModel should validate settings."""
        from fileforge.gui.viewmodels import SettingsViewModel
        vm = SettingsViewModel()
        # Invalid: negative workers
        is_valid, errors = vm.validate_setting('processing.workers', -1)
        assert is_valid is False
        assert len(errors) > 0

    def test_settings_vm_emits_signal_on_change(self):
        """SettingsViewModel should emit signal on change."""
        from fileforge.gui.viewmodels import SettingsViewModel
        vm = SettingsViewModel()
        callback = MagicMock()
        vm.setting_changed.connect(callback)
        vm.update_setting('scanning.recursive', False)
        callback.assert_called_once()

    def test_settings_vm_has_theme_setting(self):
        """SettingsViewModel should include theme setting."""
        from fileforge.gui.viewmodels import SettingsViewModel
        vm = SettingsViewModel()
        assert vm.get_setting('appearance.theme') in ['light', 'dark', 'system']
