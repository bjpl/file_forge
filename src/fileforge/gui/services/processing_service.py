"""Processing Service - manages background processing."""
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from PySide6.QtCore import QObject, Signal, QThreadPool, QRunnable, Slot

from ..state import AppState
from ...config import Settings, load_config
from ...storage.database import Database


class ProcessingWorker(QRunnable):
    """Worker for processing a single file."""

    class Signals(QObject):
        finished = Signal(Path, dict)
        error = Signal(Path, str)

    def __init__(self, file_path: Path, processor: Callable):
        super().__init__()
        self.file_path = file_path
        self.processor = processor
        self.signals = self.Signals()

    @Slot()
    def run(self):
        """Run processing."""
        try:
            result = self.processor(self.file_path)
            self.signals.finished.emit(self.file_path, result or {})
        except Exception as e:
            self.signals.error.emit(self.file_path, str(e))


class ProcessingService(QObject):
    """Service for background file processing.

    Manages a thread pool for concurrent processing
    with Qt signals for progress and completion.
    """

    processing_started = Signal()
    processing_complete = Signal(dict)
    file_processed = Signal(Path, dict)
    error_occurred = Signal(Path, str)
    progress_updated = Signal(int)

    def __init__(self, config: Optional[Settings] = None):
        super().__init__()
        self._config = config or load_config()
        self._thread_pool = QThreadPool.globalInstance()
        self._pending_work: List[Callable] = []
        self._active_workers = 0
        self._processed_count = 0
        self._error_count = 0
        self._results: List[Dict[str, Any]] = []

    @property
    def thread_pool(self) -> QThreadPool:
        """Get thread pool."""
        return self._thread_pool

    @property
    def active_workers(self) -> int:
        """Get active worker count."""
        return self._active_workers

    @property
    def pending_count(self) -> int:
        """Get pending work count."""
        return len(self._pending_work)

    def queue_work(self, work: Callable):
        """Queue work item for processing.

        Args:
            work: Callable to execute
        """
        self._pending_work.append(work)

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file.

        Args:
            file_path: Path to file

        Returns:
            Processing result
        """
        result = {'path': str(file_path), 'success': False}

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Simulate processing based on file type
            ext = file_path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                result['type'] = 'image'
                result['size'] = file_path.stat().st_size
                result['success'] = True
            elif ext in ['.pdf', '.docx']:
                result['type'] = 'document'
                result['size'] = file_path.stat().st_size
                result['success'] = True
            else:
                result['type'] = 'other'
                result['size'] = file_path.stat().st_size
                result['success'] = True

            self.file_processed.emit(file_path, result)
            self.processing_complete.emit(result)

        except Exception as e:
            result['error'] = str(e)
            self.error_occurred.emit(file_path, str(e))

        return result

    def process_batch(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files.

        Args:
            files: List of file paths

        Returns:
            List of processing results
        """
        self._results = []
        self._processed_count = 0
        self._error_count = 0
        total = len(files)

        self.processing_started.emit()

        for file_path in files:
            result = self.process_file(file_path)
            self._results.append(result)
            self._processed_count += 1

            if not result.get('success'):
                self._error_count += 1

            progress = int((self._processed_count / total) * 100)
            self.progress_updated.emit(progress)

        final_result = {
            'total': total,
            'processed': self._processed_count,
            'errors': self._error_count,
            'results': self._results
        }

        self.processing_complete.emit(final_result)
        return self._results

    def process_async(self, file_path: Path, processor: Optional[Callable] = None):
        """Process file asynchronously.

        Args:
            file_path: Path to file
            processor: Optional custom processor function
        """
        def default_processor(path: Path) -> Dict[str, Any]:
            return self.process_file(path)

        worker = ProcessingWorker(
            file_path,
            processor or default_processor
        )
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)

        self._active_workers += 1
        self._thread_pool.start(worker)

    def _on_worker_finished(self, path: Path, result: dict):
        """Handle worker completion."""
        self._active_workers -= 1
        self.file_processed.emit(path, result)

    def _on_worker_error(self, path: Path, error: str):
        """Handle worker error."""
        self._active_workers -= 1
        self.error_occurred.emit(path, error)
