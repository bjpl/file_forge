"""TDD Tests for GUI Application State Management.

RED phase: Tests written first, defining expected behavior.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestAppState:
    """Tests for global application state."""

    def test_app_state_is_singleton(self):
        """AppState should be a singleton."""
        from fileforge.gui.state import AppState
        state1 = AppState.instance()
        state2 = AppState.instance()
        assert state1 is state2

    def test_app_state_has_current_view(self):
        """AppState should track current view."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        assert hasattr(state, 'current_view')
        assert state.current_view in ['dashboard', 'browser', 'processing', 'results', 'settings']

    def test_app_state_can_change_view(self):
        """AppState should allow changing views."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        state.set_view('browser')
        assert state.current_view == 'browser'

    def test_app_state_emits_signal_on_view_change(self):
        """AppState should emit signal when view changes."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        callback = MagicMock()
        state.view_changed.connect(callback)
        state.set_view('settings')
        callback.assert_called_once_with('settings')

    def test_app_state_has_processing_status(self):
        """AppState should track processing status."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        assert hasattr(state, 'is_processing')
        assert isinstance(state.is_processing, bool)

    def test_app_state_has_selected_files(self):
        """AppState should track selected files."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        assert hasattr(state, 'selected_files')
        assert isinstance(state.selected_files, list)

    def test_app_state_can_select_files(self):
        """AppState should allow selecting files."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        files = [Path('/test/file1.jpg'), Path('/test/file2.pdf')]
        state.select_files(files)
        assert state.selected_files == files

    def test_app_state_emits_signal_on_file_selection(self):
        """AppState should emit signal when files selected."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        callback = MagicMock()
        state.files_selected.connect(callback)
        files = [Path('/test/file.jpg')]
        state.select_files(files)
        callback.assert_called_once()

    def test_app_state_has_config(self):
        """AppState should hold configuration."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        assert hasattr(state, 'config')

    def test_app_state_has_database_connection(self):
        """AppState should provide database access."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        assert hasattr(state, 'database')

    def test_app_state_reset_clears_selection(self):
        """Reset should clear file selection."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        state.select_files([Path('/test/file.jpg')])
        state.reset()
        assert state.selected_files == []


class TestAppStateTheme:
    """Tests for theme management in AppState."""

    def test_app_state_has_theme(self):
        """AppState should track current theme."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        assert hasattr(state, 'theme')
        assert state.theme in ['light', 'dark', 'system']

    def test_app_state_can_set_theme(self):
        """AppState should allow changing theme."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        state.set_theme('dark')
        assert state.theme == 'dark'

    def test_app_state_emits_signal_on_theme_change(self):
        """AppState should emit signal when theme changes."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        callback = MagicMock()
        state.theme_changed.connect(callback)
        state.set_theme('light')
        callback.assert_called_once_with('light')

    def test_app_state_persists_theme(self):
        """Theme preference should persist."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        state.set_theme('dark')
        # Simulate restart
        AppState._instance = None
        state2 = AppState.instance()
        assert state2.theme == 'dark'


class TestAppStateProcessing:
    """Tests for processing state management."""

    def test_app_state_tracks_processing_progress(self):
        """AppState should track processing progress."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        assert hasattr(state, 'processing_progress')
        assert 0 <= state.processing_progress <= 100

    def test_app_state_can_update_progress(self):
        """AppState should allow updating progress."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        state.set_progress(50)
        assert state.processing_progress == 50

    def test_app_state_emits_signal_on_progress(self):
        """AppState should emit signal on progress update."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        callback = MagicMock()
        state.progress_changed.connect(callback)
        state.set_progress(75)
        callback.assert_called_once_with(75)

    def test_app_state_has_processing_log(self):
        """AppState should maintain processing log."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        assert hasattr(state, 'processing_log')
        assert isinstance(state.processing_log, list)

    def test_app_state_can_add_log_entry(self):
        """AppState should allow adding log entries."""
        from fileforge.gui.state import AppState
        state = AppState.instance()
        state.add_log("Processing file1.jpg")
        assert "Processing file1.jpg" in state.processing_log


# Fixture to reset singleton between tests
@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset AppState singleton before each test."""
    from fileforge.gui import state
    if hasattr(state, 'AppState'):
        state.AppState._instance = None
    yield
    if hasattr(state, 'AppState'):
        state.AppState._instance = None
