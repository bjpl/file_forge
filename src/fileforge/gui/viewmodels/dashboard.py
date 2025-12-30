"""Dashboard ViewModel."""
from typing import Optional, List, Dict, Any
from PySide6.QtCore import QObject, Signal

from ...storage.database import Database
from ..state import AppState


class DashboardViewModel(QObject):
    """ViewModel for Dashboard view.

    Exposes:
    - Statistics (total files, faces, objects)
    - Recent activity
    - Quick actions
    - Processing status
    """

    data_changed = Signal()

    def __init__(self, database: Optional[Database] = None):
        super().__init__()
        self._database = database
        self._total_files = 0
        self._total_faces = 0
        self._total_objects = 0
        self._database_size = 0.0
        self._recent_activity: List[Dict[str, Any]] = []
        self._is_processing = False

        # Quick actions configuration
        self._quick_actions = [
            {'id': 'scan', 'label': 'Scan Folder', 'icon': 'folder-search'},
            {'id': 'organize', 'label': 'Organize Files', 'icon': 'folder-move'},
            {'id': 'query', 'label': 'Search Files', 'icon': 'search'},
            {'id': 'watch', 'label': 'Watch Folder', 'icon': 'eye'},
        ]

    @property
    def database(self) -> Database:
        """Get database connection."""
        if self._database is None:
            self._database = AppState.instance().database
        return self._database

    @property
    def total_files(self) -> int:
        """Get total file count."""
        return self._total_files

    @property
    def total_faces(self) -> int:
        """Get total face count."""
        return self._total_faces

    @property
    def total_objects(self) -> int:
        """Get total detected objects."""
        return self._total_objects

    @property
    def database_size(self) -> float:
        """Get database size in MB."""
        return self._database_size

    @property
    def recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity list."""
        return self._recent_activity.copy()

    @property
    def quick_actions(self) -> List[Dict[str, str]]:
        """Get quick action definitions."""
        return self._quick_actions.copy()

    @property
    def is_processing(self) -> bool:
        """Check if processing is active."""
        return AppState.instance().is_processing

    def refresh(self):
        """Refresh data from database."""
        try:
            stats = self.database.get_stats()
            self._total_files = stats.get('total_files', 0)
            self._total_faces = stats.get('total_faces', 0)
            self._total_objects = stats.get('detected_objects', 0)
            self._database_size = stats.get('database_size_mb', 0.0)

            # Get recent operations
            operations = self.database.list_operations(limit=10)
            self._recent_activity = [
                {
                    'id': op.get('id'),
                    'type': op.get('operation_type'),
                    'file': op.get('source_path'),
                    'timestamp': op.get('created_at'),
                }
                for op in operations
            ]
        except Exception:
            # Handle database errors gracefully
            pass

        self.data_changed.emit()
