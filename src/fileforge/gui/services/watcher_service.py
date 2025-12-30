"""Watcher Service - wraps file watching."""
from pathlib import Path
from typing import Optional, List, Set
from threading import Thread, Event
import time
from PySide6.QtCore import QObject, Signal


class WatcherService(QObject):
    """Service for watching file system changes.

    Monitors directories for file changes and
    emits Qt signals when changes occur.
    """

    file_created = Signal(Path)
    file_modified = Signal(Path)
    file_deleted = Signal(Path)
    error_occurred = Signal(str)

    def __init__(self, debounce_seconds: float = 1.0):
        super().__init__()
        self._debounce_seconds = debounce_seconds
        self._watched_paths: Set[Path] = set()
        self._is_watching = False
        self._stop_event = Event()
        self._watch_thread: Optional[Thread] = None
        self._file_states: dict = {}  # Track file modification times

    @property
    def debounce_seconds(self) -> float:
        """Get debounce interval."""
        return self._debounce_seconds

    @property
    def is_watching(self) -> bool:
        """Check if watching is active."""
        return self._is_watching

    @property
    def watched_paths(self) -> List[Path]:
        """Get list of watched paths."""
        return list(self._watched_paths)

    def watch(self, directory: Path):
        """Start watching a directory.

        Args:
            directory: Directory to watch
        """
        if not directory.exists():
            self.error_occurred.emit(f"Directory not found: {directory}")
            return

        self._watched_paths.add(directory)
        self._is_watching = True

        # Start watching thread if not already running
        if self._watch_thread is None or not self._watch_thread.is_alive():
            self._stop_event.clear()
            self._watch_thread = Thread(target=self._watch_loop, daemon=True)
            self._watch_thread.start()

    def stop(self):
        """Stop watching all directories."""
        self._is_watching = False
        self._stop_event.set()

        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2.0)

        self._watched_paths.clear()
        self._file_states.clear()

    def unwatch(self, directory: Path):
        """Stop watching a specific directory.

        Args:
            directory: Directory to stop watching
        """
        if directory in self._watched_paths:
            self._watched_paths.remove(directory)

        if not self._watched_paths:
            self.stop()

    def _watch_loop(self):
        """Main watching loop."""
        while not self._stop_event.is_set():
            try:
                self._check_changes()
            except Exception as e:
                self.error_occurred.emit(str(e))

            # Wait with debounce
            self._stop_event.wait(self._debounce_seconds)

    def _check_changes(self):
        """Check for file changes in watched directories."""
        for directory in list(self._watched_paths):
            if not directory.exists():
                continue

            # Get current files
            try:
                current_files = {}
                for item in directory.rglob('*'):
                    if item.is_file():
                        try:
                            current_files[str(item)] = item.stat().st_mtime
                        except (OSError, PermissionError):
                            pass

                # Check for new files
                for path_str, mtime in current_files.items():
                    if path_str not in self._file_states:
                        # New file
                        self.file_created.emit(Path(path_str))
                    elif self._file_states[path_str] != mtime:
                        # Modified file
                        self.file_modified.emit(Path(path_str))

                # Check for deleted files
                for path_str in list(self._file_states.keys()):
                    if path_str not in current_files:
                        self.file_deleted.emit(Path(path_str))

                # Update state
                self._file_states = current_files

            except (OSError, PermissionError) as e:
                self.error_occurred.emit(str(e))
