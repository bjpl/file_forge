"""Application State Management.

Singleton pattern for global application state with Qt signals.
"""
from pathlib import Path
from typing import List, Optional, Any
from PySide6.QtCore import QObject, Signal, QSettings

from ..config import Settings, load_config
from ..storage.database import Database


class AppState(QObject):
    """Global application state singleton.

    Manages:
    - Current view/navigation
    - File selection
    - Processing status
    - Theme preferences
    - Configuration
    """

    # Signals
    view_changed = Signal(str)
    files_selected = Signal(list)
    theme_changed = Signal(str)
    progress_changed = Signal(int)
    processing_started = Signal()
    processing_stopped = Signal()
    log_entry_added = Signal(str)

    _instance: Optional['AppState'] = None

    def __init__(self):
        super().__init__()
        self._current_view = 'dashboard'
        self._selected_files: List[Path] = []
        self._theme = 'system'
        self._is_processing = False
        self._processing_progress = 0
        self._processing_log: List[str] = []
        self._config: Optional[Settings] = None
        self._database: Optional[Database] = None
        self._settings = QSettings('FileForge', 'FileForge')

        # Load persisted settings
        self._load_persisted_settings()

    @classmethod
    def instance(cls) -> 'AppState':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = AppState()
        return cls._instance

    def _load_persisted_settings(self):
        """Load settings from QSettings."""
        self._theme = self._settings.value('theme', 'system')

    def _save_persisted_settings(self):
        """Save settings to QSettings."""
        self._settings.setValue('theme', self._theme)
        self._settings.sync()

    # View management
    @property
    def current_view(self) -> str:
        """Get current view name."""
        return self._current_view

    def set_view(self, view: str):
        """Set current view."""
        valid_views = ['dashboard', 'browser', 'processing', 'results', 'settings']
        if view in valid_views:
            self._current_view = view
            self.view_changed.emit(view)

    # File selection
    @property
    def selected_files(self) -> List[Path]:
        """Get selected files."""
        return self._selected_files.copy()

    def select_files(self, files: List[Path]):
        """Set selected files."""
        self._selected_files = files.copy()
        self.files_selected.emit(self._selected_files)

    def clear_selection(self):
        """Clear file selection."""
        self._selected_files = []
        self.files_selected.emit([])

    # Theme management
    @property
    def theme(self) -> str:
        """Get current theme."""
        return self._theme

    def set_theme(self, theme: str):
        """Set theme."""
        if theme in ['light', 'dark', 'system']:
            self._theme = theme
            self._save_persisted_settings()
            self.theme_changed.emit(theme)

    # Processing state
    @property
    def is_processing(self) -> bool:
        """Check if processing is active."""
        return self._is_processing

    def start_processing(self):
        """Mark processing as started."""
        self._is_processing = True
        self._processing_progress = 0
        self.processing_started.emit()

    def stop_processing(self):
        """Mark processing as stopped."""
        self._is_processing = False
        self.processing_stopped.emit()

    @property
    def processing_progress(self) -> int:
        """Get processing progress (0-100)."""
        return self._processing_progress

    def set_progress(self, value: int):
        """Set processing progress."""
        self._processing_progress = max(0, min(100, value))
        self.progress_changed.emit(self._processing_progress)

    # Processing log
    @property
    def processing_log(self) -> List[str]:
        """Get processing log entries."""
        return self._processing_log.copy()

    def add_log(self, message: str):
        """Add log entry."""
        self._processing_log.append(message)
        self.log_entry_added.emit(message)

    def clear_log(self):
        """Clear processing log."""
        self._processing_log = []

    # Configuration
    @property
    def config(self) -> Settings:
        """Get configuration."""
        if self._config is None:
            self._config = load_config()
        return self._config

    def reload_config(self):
        """Reload configuration from disk."""
        self._config = load_config()

    # Database
    @property
    def database(self) -> Database:
        """Get database connection."""
        if self._database is None:
            self._database = Database(self.config.database.path)
        return self._database

    # Reset
    def reset(self):
        """Reset state to defaults."""
        self._selected_files = []
        self._is_processing = False
        self._processing_progress = 0
        self._processing_log = []
        self.files_selected.emit([])
