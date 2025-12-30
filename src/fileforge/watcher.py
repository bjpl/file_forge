"""File watcher module using watchdog."""
from pathlib import Path
from typing import Optional, Callable
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent


class FileForgeEventHandler(FileSystemEventHandler):
    """Handler for file system events."""

    def __init__(self, callback: Callable, debounce_seconds: int = 2):
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._pending = {}
        self._last_processed = {}

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_processing(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_processing(Path(event.src_path))

    def _schedule_processing(self, path: Path):
        now = time.time()
        if path in self._last_processed:
            if now - self._last_processed[path] < self.debounce_seconds:
                return
        self._last_processed[path] = now
        self.callback(path)


class FileWatcher:
    """Watch directories for new files."""

    def __init__(self, callback: Callable, debounce: int = 2):
        self.callback = callback
        self.debounce = debounce
        self.observer = None

    def watch(self, path: Path):
        """Start watching a directory."""
        handler = FileForgeEventHandler(self.callback, self.debounce)
        self.observer = Observer()
        self.observer.schedule(handler, str(path), recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

    def stop(self):
        """Stop watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
