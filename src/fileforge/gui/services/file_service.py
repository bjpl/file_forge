"""File Service - wraps file operations for GUI."""
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import Future, ThreadPoolExecutor
from PySide6.QtCore import QObject, Signal

from ..state import AppState
from ...config import Settings, load_config
from ...storage.database import Database


class FileService(QObject):
    """Service for file operations.

    Wraps backend file scanning and organization
    with Qt signals for progress updates.
    """

    progress_updated = Signal(int, str)  # progress, message
    scan_complete = Signal(dict)
    organize_complete = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, config: Optional[Settings] = None,
                 database: Optional[Database] = None):
        super().__init__()
        self._config = config or load_config()
        self._database = database
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._is_cancelled = False
        self._current_future: Optional[Future] = None

    @property
    def config(self) -> Settings:
        """Get configuration."""
        return self._config

    @property
    def database(self) -> Database:
        """Get database connection."""
        if self._database is None:
            self._database = AppState.instance().database
        return self._database

    @property
    def is_cancelled(self) -> bool:
        """Check if current operation is cancelled."""
        return self._is_cancelled

    def scan(self, directory: Path, recursive: bool = True) -> Dict[str, Any]:
        """Scan directory for files.

        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories

        Returns:
            Scan results with files_found count
        """
        self._is_cancelled = False
        files_found = []

        try:
            # Emit starting progress
            self.progress_updated.emit(0, f"Scanning {directory}")

            if directory.exists() and directory.is_dir():
                pattern = '**/*' if recursive else '*'
                all_items = list(directory.glob(pattern))
                total = len(all_items)

                for i, item in enumerate(all_items):
                    if self._is_cancelled:
                        break

                    if item.is_file():
                        ext = item.suffix.lower()
                        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp',
                                   '.pdf', '.docx', '.xlsx', '.txt', '.md']:
                            files_found.append({
                                'path': str(item),
                                'name': item.name,
                                'size': item.stat().st_size,
                                'extension': ext
                            })

                    if total > 0:
                        progress = int((i / total) * 100)
                        self.progress_updated.emit(progress, f"Scanning: {item.name}")

            self.progress_updated.emit(100, "Scan complete")

        except Exception as e:
            self.error_occurred.emit(str(e))

        result = {
            'files_found': len(files_found),
            'files': files_found,
            'directory': str(directory),
            'recursive': recursive,
            'cancelled': self._is_cancelled
        }

        self.scan_complete.emit(result)
        return result

    def scan_async(self, directory: Path, recursive: bool = True) -> Future:
        """Scan directory asynchronously.

        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories

        Returns:
            Future with scan results
        """
        self._is_cancelled = False
        self._current_future = self._executor.submit(
            self.scan, directory, recursive
        )
        return self._current_future

    def organize(self, files: List[Path], dry_run: bool = False) -> Dict[str, Any]:
        """Organize files based on their content.

        Args:
            files: List of file paths
            dry_run: If True, only show proposed actions

        Returns:
            Organization results
        """
        self._is_cancelled = False
        operations = []
        proposed_actions = []

        try:
            self.progress_updated.emit(0, "Analyzing files")

            for i, file_path in enumerate(files):
                if self._is_cancelled:
                    break

                if file_path.exists():
                    # Determine category based on extension
                    ext = file_path.suffix.lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                        category = 'images'
                    elif ext in ['.pdf', '.docx', '.xlsx']:
                        category = 'documents'
                    elif ext in ['.txt', '.md']:
                        category = 'text'
                    else:
                        category = 'other'

                    action = {
                        'source': str(file_path),
                        'category': category,
                        'action': 'move' if not dry_run else 'proposed_move'
                    }

                    if dry_run:
                        proposed_actions.append(action)
                    else:
                        operations.append(action)

                progress = int(((i + 1) / len(files)) * 100)
                self.progress_updated.emit(progress, f"Processing: {file_path.name}")

            self.progress_updated.emit(100, "Organization complete")

        except Exception as e:
            self.error_occurred.emit(str(e))

        result = {
            'dry_run': dry_run,
            'operations': operations,
            'proposed_actions': proposed_actions,
            'total_processed': len(files),
            'cancelled': self._is_cancelled
        }

        self.organize_complete.emit(result)
        return result

    def cancel(self):
        """Cancel current operation."""
        self._is_cancelled = True
        if self._current_future:
            self._current_future.cancel()

    def shutdown(self):
        """Shutdown executor."""
        self._executor.shutdown(wait=False)
