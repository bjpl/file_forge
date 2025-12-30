"""Processing Queue ViewModel."""
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import time
from PySide6.QtCore import QObject, Signal, QThreadPool, QRunnable, Slot

from ..state import AppState
from ...pipeline.orchestrator import PipelineOrchestrator


class ProcessingWorker(QRunnable):
    """Worker for background file processing."""

    class Signals(QObject):
        progress = Signal(int, str)  # progress, message
        file_complete = Signal(Path, dict)  # path, result
        error = Signal(Path, str)  # path, error message
        finished = Signal(dict)  # final results

    def __init__(self, files: List[Path], orchestrator: PipelineOrchestrator):
        super().__init__()
        self.files = files
        self.orchestrator = orchestrator
        self.signals = self.Signals()
        self._cancelled = False
        self._paused = False

    def cancel(self):
        """Cancel processing."""
        self._cancelled = True

    def pause(self):
        """Pause processing."""
        self._paused = True

    def resume(self):
        """Resume processing."""
        self._paused = False

    @Slot()
    def run(self):
        """Run processing."""
        results = {
            'processed': 0,
            'succeeded': 0,
            'failed': 0,
            'errors': []
        }

        total = len(self.files)
        for i, file_path in enumerate(self.files):
            if self._cancelled:
                break

            while self._paused:
                time.sleep(0.1)
                if self._cancelled:
                    break

            try:
                # Process file
                progress = int((i / total) * 100)
                self.signals.progress.emit(progress, f"Processing {file_path.name}")

                result = self.orchestrator.process_file(file_path)
                results['processed'] += 1
                results['succeeded'] += 1
                self.signals.file_complete.emit(file_path, result or {})

            except Exception as e:
                results['failed'] += 1
                results['errors'].append({'file': str(file_path), 'error': str(e)})
                self.signals.error.emit(file_path, str(e))

        self.signals.progress.emit(100, "Complete")
        self.signals.finished.emit(results)


class ProcessingViewModel(QObject):
    """ViewModel for Processing Queue view.

    Manages:
    - Processing queue
    - Background workers
    - Progress tracking
    - Start/pause/cancel controls
    """

    progress_changed = Signal(int)
    queue_changed = Signal()
    log_added = Signal(str)
    processing_complete = Signal(dict)

    def __init__(self, orchestrator: Optional[PipelineOrchestrator] = None):
        super().__init__()
        self._queue: List[Path] = []
        self._current_file: Optional[Path] = None
        self._progress = 0
        self._is_running = False
        self._is_paused = False
        self._log_entries: List[str] = []
        self._eta_seconds = 0
        self._start_time: Optional[float] = None
        self._orchestrator = orchestrator
        self._thread_pool = QThreadPool.globalInstance()
        self._current_worker: Optional[ProcessingWorker] = None

    @property
    def queue(self) -> List[Path]:
        """Get processing queue."""
        return self._queue.copy()

    @property
    def current_file(self) -> Optional[Path]:
        """Get currently processing file."""
        return self._current_file

    @property
    def progress(self) -> int:
        """Get progress percentage."""
        return self._progress

    @property
    def is_running(self) -> bool:
        """Check if processing is running."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Check if processing is paused."""
        return self._is_paused

    @property
    def log_entries(self) -> List[str]:
        """Get log entries."""
        return self._log_entries.copy()

    @property
    def eta_seconds(self) -> int:
        """Get estimated time remaining."""
        return self._eta_seconds

    def add_to_queue(self, files: List[Path]):
        """Add files to processing queue."""
        self._queue.extend(files)
        self.queue_changed.emit()

    def remove_from_queue(self, file_path: Path):
        """Remove file from queue."""
        if file_path in self._queue:
            self._queue.remove(file_path)
            self.queue_changed.emit()

    def clear_queue(self):
        """Clear processing queue."""
        self._queue = []
        self.queue_changed.emit()

    def start(self):
        """Start processing."""
        if not self._queue or self._is_running:
            return

        self._is_running = True
        self._is_paused = False
        self._start_time = time.time()
        AppState.instance().start_processing()

        # Create worker
        config = AppState.instance().config
        if self._orchestrator is None:
            db = AppState.instance().database
            self._orchestrator = PipelineOrchestrator(config, db)

        self._current_worker = ProcessingWorker(self._queue.copy(), self._orchestrator)
        self._current_worker.signals.progress.connect(self._on_progress)
        self._current_worker.signals.file_complete.connect(self._on_file_complete)
        self._current_worker.signals.error.connect(self._on_error)
        self._current_worker.signals.finished.connect(self._on_finished)

        self._thread_pool.start(self._current_worker)
        self._add_log("Processing started")

    def pause(self):
        """Pause processing."""
        if self._current_worker:
            self._current_worker.pause()
            self._is_paused = True
            self._add_log("Processing paused")

    def resume(self):
        """Resume processing."""
        if self._current_worker:
            self._current_worker.resume()
            self._is_paused = False
            self._add_log("Processing resumed")

    def cancel(self):
        """Cancel processing."""
        if self._current_worker:
            self._current_worker.cancel()
        self._is_running = False
        self._is_paused = False
        AppState.instance().stop_processing()
        self._add_log("Processing cancelled")

    def _on_progress(self, progress: int, message: str):
        """Handle progress update."""
        self._progress = progress
        self._update_eta()
        self.progress_changed.emit(progress)
        AppState.instance().set_progress(progress)
        self._add_log(message)

    def _on_file_complete(self, path: Path, result: dict):
        """Handle file completion."""
        self._current_file = None
        if path in self._queue:
            self._queue.remove(path)
        self.queue_changed.emit()

    def _on_error(self, path: Path, error: str):
        """Handle processing error."""
        self._add_log(f"Error processing {path.name}: {error}")

    def _on_finished(self, results: dict):
        """Handle processing completion."""
        self._is_running = False
        self._progress = 100
        AppState.instance().stop_processing()
        self._add_log(f"Complete: {results['succeeded']} succeeded, {results['failed']} failed")
        self.processing_complete.emit(results)

    def _update_progress(self, value: int):
        """Update progress value."""
        self._progress = value
        self.progress_changed.emit(value)

    def _update_eta(self):
        """Update ETA estimate."""
        if self._start_time and self._progress > 0:
            elapsed = time.time() - self._start_time
            total_estimated = elapsed / (self._progress / 100)
            self._eta_seconds = int(total_estimated - elapsed)

    def _add_log(self, message: str):
        """Add log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._log_entries.append(entry)
        AppState.instance().add_log(entry)
        self.log_added.emit(entry)
