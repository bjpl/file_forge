"""File Browser ViewModel."""
from pathlib import Path
from typing import Optional, List, Dict, Any
from PySide6.QtCore import QObject, Signal

from ..state import AppState


class FileBrowserViewModel(QObject):
    """ViewModel for File Browser view.

    Manages:
    - Current directory navigation
    - File listing with filtering
    - Selection management
    - View mode (list/grid/details)
    - Drag-drop handling
    """

    path_changed = Signal(Path)
    files_changed = Signal()
    selection_changed = Signal(list)
    view_mode_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._current_path = Path.home()
        self._files: List[Dict[str, Any]] = []
        self._selected_files: List[Path] = []
        self._current_filter = 'all'
        self._view_mode = 'list'
        self._sort_by = 'name'
        self._sort_ascending = True

    @property
    def current_path(self) -> Path:
        """Get current directory path."""
        return self._current_path

    @property
    def files(self) -> List[Dict[str, Any]]:
        """Get file list for current directory."""
        return self._files.copy()

    @property
    def selected_files(self) -> List[Path]:
        """Get selected files."""
        return self._selected_files.copy()

    @property
    def current_filter(self) -> str:
        """Get current file type filter."""
        return self._current_filter

    @property
    def view_mode(self) -> str:
        """Get current view mode."""
        return self._view_mode

    @property
    def breadcrumbs(self) -> List[Dict[str, Any]]:
        """Get breadcrumb navigation items."""
        parts = self._current_path.parts
        crumbs = []
        for i, part in enumerate(parts):
            path = Path(*parts[:i+1])
            crumbs.append({
                'label': part if part != '/' else 'Root',
                'path': path
            })
        return crumbs

    def navigate_to(self, path: Path):
        """Navigate to a directory."""
        if path.exists() and path.is_dir():
            self._current_path = path
            self._refresh_files()
            self.path_changed.emit(path)

    def navigate_up(self):
        """Navigate to parent directory."""
        parent = self._current_path.parent
        if parent != self._current_path:
            self.navigate_to(parent)

    def _refresh_files(self):
        """Refresh file list from filesystem."""
        self._files = []
        try:
            for item in self._current_path.iterdir():
                if self._matches_filter(item):
                    self._files.append({
                        'path': item,
                        'name': item.name,
                        'is_dir': item.is_dir(),
                        'size': item.stat().st_size if item.is_file() else 0,
                        'modified': item.stat().st_mtime,
                    })
            self._sort_files()
        except PermissionError:
            pass
        self.files_changed.emit()

    def _matches_filter(self, path: Path) -> bool:
        """Check if path matches current filter."""
        if self._current_filter == 'all':
            return True

        if path.is_dir():
            return True

        ext = path.suffix.lower()
        if self._current_filter == 'images':
            return ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        elif self._current_filter == 'documents':
            return ext in ['.pdf', '.docx', '.doc', '.xlsx', '.pptx', '.odt']
        elif self._current_filter == 'text':
            return ext in ['.txt', '.md', '.markdown', '.rst']
        return True

    def _sort_files(self):
        """Sort file list."""
        # Directories first, then by sort key
        def sort_key(f):
            is_dir = 0 if f['is_dir'] else 1
            if self._sort_by == 'name':
                return (is_dir, f['name'].lower())
            elif self._sort_by == 'size':
                return (is_dir, f['size'])
            elif self._sort_by == 'modified':
                return (is_dir, f['modified'])
            return (is_dir, f['name'].lower())

        self._files.sort(key=sort_key, reverse=not self._sort_ascending)

    def set_filter(self, filter_type: str):
        """Set file type filter."""
        self._current_filter = filter_type
        self._refresh_files()

    def set_view_mode(self, mode: str):
        """Set view mode."""
        if mode in ['list', 'grid', 'details']:
            self._view_mode = mode
            self.view_mode_changed.emit(mode)

    def select_file(self, path: Path):
        """Select a single file."""
        self._selected_files = [path]
        AppState.instance().select_files(self._selected_files)
        self.selection_changed.emit(self._selected_files)

    def select_files(self, paths: List[Path]):
        """Select multiple files."""
        self._selected_files = paths.copy()
        AppState.instance().select_files(self._selected_files)
        self.selection_changed.emit(self._selected_files)

    def toggle_selection(self, path: Path):
        """Toggle file selection."""
        if path in self._selected_files:
            self._selected_files.remove(path)
        else:
            self._selected_files.append(path)
        AppState.instance().select_files(self._selected_files)
        self.selection_changed.emit(self._selected_files)

    def clear_selection(self):
        """Clear selection."""
        self._selected_files = []
        AppState.instance().clear_selection()
        self.selection_changed.emit([])

    def handle_drop(self, paths: List[str]) -> bool:
        """Handle dropped files."""
        # Convert to Path objects and add to processing queue
        dropped = [Path(p) for p in paths if Path(p).exists()]
        if dropped:
            AppState.instance().select_files(dropped)
            return True
        return False

    def refresh(self):
        """Refresh current directory."""
        self._refresh_files()
