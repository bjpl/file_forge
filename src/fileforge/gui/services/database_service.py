"""Database Service - wraps database operations."""
from pathlib import Path
from typing import Optional, List, Dict, Any
from PySide6.QtCore import QObject, Signal

from ..state import AppState
from ...storage.database import Database
from ...config import Settings, load_config


class DatabaseService(QObject):
    """Service for database operations.

    Wraps the Database class with caching and
    Qt signals for reactive updates.
    """

    data_changed = Signal()
    query_complete = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, database: Optional[Database] = None,
                 config: Optional[Settings] = None):
        super().__init__()
        self._config = config or load_config()
        self._database = database
        self._is_connected = False
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 60  # seconds

        # Connect on init
        self._connect()

    def _connect(self):
        """Connect to database."""
        try:
            if self._database is None:
                db_path = Path(self._config.database.path).expanduser()
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self._database = Database(db_path)
            self._is_connected = True
        except Exception as e:
            self._is_connected = False
            self.error_occurred.emit(str(e))

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._is_connected

    @property
    def cache_size(self) -> int:
        """Get cache size."""
        return len(self._cache)

    @property
    def database(self) -> Optional[Database]:
        """Get underlying database."""
        return self._database

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Statistics dictionary
        """
        cache_key = 'stats'
        if cache_key in self._cache:
            return self._cache[cache_key]

        stats = {
            'total_files': 0,
            'total_faces': 0,
            'detected_objects': 0,
            'database_size_mb': 0.0
        }

        try:
            if self._database:
                db_stats = self._database.get_stats()
                stats.update(db_stats)
        except Exception as e:
            self.error_occurred.emit(str(e))

        self._cache[cache_key] = stats
        return stats

    def query_files(self, **kwargs) -> List[Dict[str, Any]]:
        """Query files from database.

        Args:
            **kwargs: Query parameters (tag, category, text_search, etc.)

        Returns:
            List of matching files
        """
        results = []

        try:
            if self._database:
                results = self._database.query_files(**kwargs)
            self.query_complete.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

        return results

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search files by text.

        Args:
            query: Search query

        Returns:
            Matching files
        """
        results = []

        try:
            if self._database:
                results = self._database.query_files(text_search=query)
            self.query_complete.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

        return results

    def get_face_clusters(self) -> List[Dict[str, Any]]:
        """Get face clusters.

        Returns:
            List of face clusters
        """
        cache_key = 'face_clusters'
        if cache_key in self._cache:
            return self._cache[cache_key]

        clusters = []

        try:
            if self._database:
                clusters = self._database.get_face_clusters()
        except Exception as e:
            self.error_occurred.emit(str(e))

        self._cache[cache_key] = clusters
        return clusters

    def get_recent_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent operations.

        Args:
            limit: Maximum number of operations

        Returns:
            List of recent operations
        """
        operations = []

        try:
            if self._database:
                operations = self._database.list_operations(limit=limit)
        except Exception as e:
            self.error_occurred.emit(str(e))

        return operations

    def invalidate_cache(self):
        """Invalidate all cached data."""
        self._cache = {}
        self.data_changed.emit()

    def refresh(self):
        """Refresh data from database."""
        self.invalidate_cache()
        self.get_stats()  # Repopulate cache
