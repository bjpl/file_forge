"""Results Gallery ViewModel."""
from pathlib import Path
from typing import Optional, List, Dict, Any
from PySide6.QtCore import QObject, Signal

from ..state import AppState
from ...storage.database import Database
from ...storage.history import OperationHistory


class ResultsViewModel(QObject):
    """ViewModel for Results Gallery view.

    Manages:
    - Processed files display
    - Filtering by category/tag
    - Search functionality
    - Undo operations
    - Export functionality
    """

    data_changed = Signal()
    filter_changed = Signal()

    def __init__(self, database: Optional[Database] = None,
                 history: Optional[OperationHistory] = None):
        super().__init__()
        self._database = database
        self._history = history
        self._files: List[Dict[str, Any]] = []
        self._category_filter: Optional[str] = None
        self._tag_filters: List[str] = []
        self._search_query: Optional[str] = None
        self._sort_by = 'processed_at'
        self._sort_order = 'desc'

    @property
    def database(self) -> Database:
        """Get database connection."""
        if self._database is None:
            self._database = AppState.instance().database
        return self._database

    @property
    def history(self) -> OperationHistory:
        """Get operation history."""
        if self._history is None:
            self._history = OperationHistory(self.database)
        return self._history

    @property
    def files(self) -> List[Dict[str, Any]]:
        """Get processed files."""
        return self._files.copy()

    @property
    def category_filter(self) -> Optional[str]:
        """Get category filter."""
        return self._category_filter

    @property
    def tag_filters(self) -> List[str]:
        """Get tag filters."""
        return self._tag_filters.copy()

    @property
    def search_query(self) -> Optional[str]:
        """Get search query."""
        return self._search_query

    @property
    def sort_by(self) -> str:
        """Get sort field."""
        return self._sort_by

    @property
    def sort_order(self) -> str:
        """Get sort order."""
        return self._sort_order

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        last_op = self.database.get_last_operation()
        return last_op is not None

    def set_category_filter(self, category: Optional[str]):
        """Set category filter."""
        self._category_filter = category
        self._refresh_files()
        self.filter_changed.emit()

    def set_tag_filter(self, tag: str):
        """Add tag to filters."""
        if tag not in self._tag_filters:
            self._tag_filters.append(tag)
            self._refresh_files()
            self.filter_changed.emit()

    def remove_tag_filter(self, tag: str):
        """Remove tag from filters."""
        if tag in self._tag_filters:
            self._tag_filters.remove(tag)
            self._refresh_files()
            self.filter_changed.emit()

    def clear_filters(self):
        """Clear all filters."""
        self._category_filter = None
        self._tag_filters = []
        self._search_query = None
        self._refresh_files()
        self.filter_changed.emit()

    def search(self, query: str):
        """Search files by text."""
        self._search_query = query if query else None
        self._refresh_files()

    def set_sort(self, field: str, order: str = 'asc'):
        """Set sort options."""
        self._sort_by = field
        self._sort_order = order
        self._refresh_files()

    def _refresh_files(self):
        """Refresh file list from database."""
        try:
            # Build query based on filters
            files = self.database.query_files(
                category=self._category_filter,
                tags=self._tag_filters if self._tag_filters else None,
                text_search=self._search_query,
                limit=100
            )
            self._files = files
        except Exception:
            self._files = []

        self.data_changed.emit()

    def undo(self) -> bool:
        """Undo last operation."""
        try:
            return self.history.undo_last()
        except Exception:
            return False

    def undo_batch(self, batch_id: str) -> bool:
        """Undo entire batch."""
        try:
            return self.history.undo_batch(batch_id)
        except Exception:
            return False

    def export_json(self, output_path: Path) -> bool:
        """Export results to JSON."""
        import json
        try:
            with open(output_path, 'w') as f:
                json.dump(self._files, f, indent=2, default=str)
            return True
        except Exception:
            return False

    def export_csv(self, output_path: Path) -> bool:
        """Export results to CSV."""
        import csv
        try:
            if not self._files:
                return False

            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._files[0].keys())
                writer.writeheader()
                writer.writerows(self._files)
            return True
        except Exception:
            return False

    def refresh(self):
        """Refresh data."""
        self._refresh_files()
