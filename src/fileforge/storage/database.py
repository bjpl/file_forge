"""
FileForge Database Module

Provides SQLite-based persistent storage for file metadata, operations,
detections, and processing runs with full ACID transaction support.
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database manager for FileForge.

    Handles all persistent storage including:
    - File metadata and categorization
    - Operation journaling for undo/redo
    - Object detection results
    - Face detection and clustering
    - NSFW content detection
    - Processing errors and runs
    - Text extraction results
    """

    # Current schema version
    SCHEMA_VERSION = 1

    def __init__(self, db_path: str, wal_mode: bool = True):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            wal_mode: Enable Write-Ahead Logging for better concurrency
        """
        self.db_path = db_path
        self.wal_mode = wal_mode

        # Ensure parent directory exists (not for in-memory)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")

        # Set WAL mode if requested
        if wal_mode:
            self.conn.execute("PRAGMA journal_mode = WAL")

        # Set cache size for performance
        self.conn.execute("PRAGMA cache_size = -64000")  # 64MB

    def initialize(self):
        """Create all database tables and indexes."""
        # Set busy timeout to handle concurrent access
        self.conn.execute("PRAGMA busy_timeout = 5000")  # 5 seconds

        cursor = self.conn.cursor()

        # Schema version table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Files table - main file metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_hash TEXT,
                file_type TEXT,
                original_name TEXT,
                suggested_name TEXT,
                category TEXT,
                content_text TEXT,
                summary TEXT,
                tags TEXT,
                metadata TEXT,
                confidence REAL,
                processed_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_duplicate_of INTEGER,
                FOREIGN KEY (is_duplicate_of) REFERENCES files(id)
            )
        """)

        # Operations table - journal for undo/redo
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                operation_type TEXT NOT NULL,
                source_path TEXT,
                dest_path TEXT,
                old_value TEXT,
                new_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                undone INTEGER DEFAULT 0,
                undone_at TIMESTAMP
            )
        """)

        # Detected objects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detected_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL,
                bbox TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
        """)

        # Extracted text table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_text (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                text TEXT,
                confidence REAL,
                page_num INTEGER,
                bbox TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
        """)

        # Faces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                embedding BLOB,
                cluster_id INTEGER,
                cluster_name TEXT,
                confidence REAL,
                bbox TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
        """)

        # NSFW detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nsfw_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                is_nsfw INTEGER NOT NULL,
                confidence REAL,
                scores TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
        """)

        # Processing errors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                error_type TEXT,
                error_message TEXT,
                traceback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
        """)

        # Processing runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type TEXT NOT NULL,
                parameters TEXT,
                status TEXT,
                files_processed INTEGER DEFAULT 0,
                files_succeeded INTEGER DEFAULT 0,
                files_failed INTEGER DEFAULT 0,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_category ON files(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_batch ON operations(batch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_type ON operations(operation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detected_objects_file ON detected_objects(file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detected_objects_label ON detected_objects(label)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_file ON faces(file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nsfw_file ON nsfw_detections(file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_errors_file ON processing_errors(file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_errors_stage ON processing_errors(stage)")

        # Set schema version
        cursor.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (self.SCHEMA_VERSION,))

        self._commit()

    def get_tables(self) -> List[str]:
        """Get list of all tables in database."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_indexes(self) -> List[str]:
        """Get list of all indexes in database."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_journal_mode(self) -> str:
        """Get current journal mode."""
        cursor = self.conn.execute("PRAGMA journal_mode")
        return cursor.fetchone()[0].lower()

    def get_pragma(self, name: str) -> Any:
        """Get value of a PRAGMA setting."""
        cursor = self.conn.execute(f"PRAGMA {name}")
        result = cursor.fetchone()
        return result[0] if result else None

    def get_schema_version(self) -> int:
        """Get current schema version."""
        try:
            cursor = self.conn.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else 0
        except sqlite3.OperationalError:
            return 0

    def migrate_schema(self):
        """Migrate database schema to latest version."""
        current_version = self.get_schema_version()

        if current_version < self.SCHEMA_VERSION:
            # Future migrations would go here
            # For now, just reinitialize
            self.initialize()

    # File operations

    def upsert_file(self, data: Dict[str, Any]) -> int:
        """
        Insert or update file metadata.

        Args:
            data: File metadata dictionary

        Returns:
            File ID
        """
        # Serialize JSON fields if they're not already strings
        if 'tags' in data and not isinstance(data['tags'], str):
            data['tags'] = json.dumps(data['tags'])
        if 'metadata' in data and not isinstance(data['metadata'], str):
            data['metadata'] = json.dumps(data['metadata'])

        # Use datetime.now() with microseconds for precise timestamps
        current_timestamp = datetime.now().isoformat()

        cursor = self.conn.execute("""
            INSERT INTO files (
                file_path, file_hash, file_type, original_name, suggested_name,
                category, content_text, summary, tags, metadata, confidence,
                processed_at, is_duplicate_of, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                file_hash = excluded.file_hash,
                file_type = excluded.file_type,
                original_name = excluded.original_name,
                suggested_name = excluded.suggested_name,
                category = excluded.category,
                content_text = excluded.content_text,
                summary = excluded.summary,
                tags = excluded.tags,
                metadata = excluded.metadata,
                confidence = excluded.confidence,
                processed_at = excluded.processed_at,
                is_duplicate_of = excluded.is_duplicate_of,
                updated_at = excluded.updated_at
        """, (
            data.get('file_path'),
            data.get('file_hash'),
            data.get('file_type'),
            data.get('original_name'),
            data.get('suggested_name'),
            data.get('category'),
            data.get('content_text'),
            data.get('summary'),
            data.get('tags'),
            data.get('metadata'),
            data.get('confidence'),
            data.get('processed_at'),
            data.get('is_duplicate_of'),
            current_timestamp
        ))

        # Only commit if not in a transaction
        self._commit()

        # Get the file_id
        cursor = self.conn.execute(
            "SELECT id FROM files WHERE file_path = ?",
            (data.get('file_path'),)
        )
        return cursor.fetchone()[0]

    def get_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Get file by path."""
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE file_path = ?",
            (path,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_file_by_hash(self, hash: str) -> Optional[Dict[str, Any]]:
        """Get file by hash."""
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE file_hash = ? LIMIT 1",
            (hash,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_file_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get file by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (file_id,)
        )
        row = cursor.fetchone()
        result = self._row_to_dict(row) if row else None
        if result:
            result['file_id'] = result['id']
        return result

    def delete_file(self, file_id: int):
        """Delete file and all associated data."""
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self._commit()

    def query_files(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query files with filters.

        Args:
            filters: Dictionary of field:value filters
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of file dictionaries
        """
        where_clauses = []
        params = []

        for key, value in filters.items():
            if key == 'text_search':
                where_clauses.append("content_text LIKE ?")
                params.append(f'%{value}%')
            elif key == 'tag':
                where_clauses.append("tags LIKE ?")
                params.append(f'%{value}%')
            else:
                where_clauses.append(f"{key} = ?")
                params.append(value)

        query = "SELECT * FROM files"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY id"

        if limit is not None:
            query += f" LIMIT {limit}"
        if offset is not None:
            query += f" OFFSET {offset}"

        cursor = self.conn.execute(query, params)
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Operation journal

    def add_operation(self, data: Dict[str, Any]) -> int:
        """Add operation to journal."""
        cursor = self.conn.execute("""
            INSERT INTO operations (
                batch_id, operation_type, source_path, dest_path,
                old_value, new_value
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            data.get('batch_id'),
            data.get('operation_type'),
            data.get('source_path'),
            data.get('dest_path'),
            data.get('old_value'),
            data.get('new_value')
        ))

        self._commit()
        return cursor.lastrowid

    def get_operation(self, op_id: int) -> Optional[Dict[str, Any]]:
        """Get operation by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM operations WHERE id = ?",
            (op_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_operations_by_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get all operations in a batch."""
        cursor = self.conn.execute(
            "SELECT * FROM operations WHERE batch_id = ? ORDER BY id",
            (batch_id,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_operations_by_type(self, op_type: str) -> List[Dict[str, Any]]:
        """Get all operations of a type."""
        cursor = self.conn.execute(
            "SELECT * FROM operations WHERE operation_type = ? ORDER BY id",
            (op_type,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def mark_operation_undone(self, op_id: int):
        """Mark operation as undone."""
        self.conn.execute("""
            UPDATE operations
            SET undone = 1, undone_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (op_id,))
        self._commit()

    def get_undoable_operations(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get undoable operations in a batch."""
        cursor = self.conn.execute("""
            SELECT * FROM operations
            WHERE batch_id = ? AND undone = 0
            ORDER BY id DESC
        """, (batch_id,))
        results = [self._row_to_dict(row) for row in cursor.fetchall()]
        # Add operation_id field
        for result in results:
            result['operation_id'] = result['id']
        return results

    def get_last_operation(self) -> Optional[Dict[str, Any]]:
        """Get the most recent undoable operation."""
        cursor = self.conn.execute("""
            SELECT * FROM operations
            WHERE undone = 0
            ORDER BY id DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def list_operations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent operations."""
        cursor = self.conn.execute("""
            SELECT * FROM operations
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Object detection

    def add_detected_objects(self, file_id: int, objects: List[Dict[str, Any]]):
        """Add detected objects for a file."""
        for obj in objects:
            bbox = obj.get('bbox')
            if bbox and not isinstance(bbox, str):
                bbox = json.dumps(bbox)

            self.conn.execute("""
                INSERT INTO detected_objects (file_id, label, confidence, bbox)
                VALUES (?, ?, ?, ?)
            """, (file_id, obj.get('label'), obj.get('confidence'), bbox))

        self._commit()

    def get_detected_objects(self, file_id: int) -> List[Dict[str, Any]]:
        """Get all detected objects for a file."""
        cursor = self.conn.execute(
            "SELECT * FROM detected_objects WHERE file_id = ?",
            (file_id,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def query_files_by_object(self, label: str) -> List[Dict[str, Any]]:
        """Query files containing a specific object."""
        cursor = self.conn.execute("""
            SELECT DISTINCT f.*
            FROM files f
            JOIN detected_objects o ON f.id = o.file_id
            WHERE o.label = ?
        """, (label,))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Face detection

    def add_face(self, file_id: int, data: Dict[str, Any]) -> int:
        """Add detected face."""
        embedding = data.get('embedding')
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tobytes()

        bbox = data.get('bbox')
        if bbox and not isinstance(bbox, str):
            bbox = json.dumps(bbox)

        cursor = self.conn.execute("""
            INSERT INTO faces (
                file_id, embedding, cluster_id, cluster_name, confidence, bbox
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file_id,
            embedding,
            data.get('cluster_id'),
            data.get('cluster_name'),
            data.get('confidence'),
            bbox
        ))

        self._commit()
        return cursor.lastrowid

    def get_face(self, face_id: int) -> Optional[Dict[str, Any]]:
        """Get face by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM faces WHERE id = ?",
            (face_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_face_embedding(self, face_id: int) -> Optional[np.ndarray]:
        """Get face embedding as numpy array."""
        cursor = self.conn.execute(
            "SELECT embedding FROM faces WHERE id = ?",
            (face_id,)
        )
        row = cursor.fetchone()
        if row and row[0]:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def update_face_cluster(self, face_id: int, cluster_id: int, cluster_name: str):
        """Update face cluster assignment."""
        self.conn.execute("""
            UPDATE faces
            SET cluster_id = ?, cluster_name = ?
            WHERE id = ?
        """, (cluster_id, cluster_name, face_id))
        self._commit()

    def get_faces_by_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get all faces in a cluster."""
        cursor = self.conn.execute(
            "SELECT * FROM faces WHERE cluster_id = ?",
            (cluster_id,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_all_face_embeddings(self) -> List[Dict[str, Any]]:
        """Get all face embeddings."""
        cursor = self.conn.execute("SELECT id, embedding FROM faces WHERE embedding IS NOT NULL")
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'embedding': np.frombuffer(row[1], dtype=np.float32) if row[1] else None
            })
        return results

    # NSFW detection

    def add_nsfw_detection(self, file_id: int, data: Dict[str, Any]) -> int:
        """Add NSFW detection result."""
        scores = data.get('scores')
        if scores and not isinstance(scores, str):
            scores = json.dumps(scores)

        cursor = self.conn.execute("""
            INSERT INTO nsfw_detections (file_id, is_nsfw, confidence, scores)
            VALUES (?, ?, ?, ?)
        """, (
            file_id,
            1 if data.get('is_nsfw') else 0,
            data.get('confidence'),
            scores
        ))

        self._commit()
        return cursor.lastrowid

    def get_nsfw_detection(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get NSFW detection for file."""
        cursor = self.conn.execute(
            "SELECT * FROM nsfw_detections WHERE file_id = ? LIMIT 1",
            (file_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def query_nsfw_files(self, is_nsfw: bool) -> List[Dict[str, Any]]:
        """Query files by NSFW status."""
        cursor = self.conn.execute("""
            SELECT DISTINCT f.*
            FROM files f
            JOIN nsfw_detections n ON f.id = n.file_id
            WHERE n.is_nsfw = ?
        """, (1 if is_nsfw else 0,))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Processing errors

    def add_processing_error(self, file_id: int, data: Dict[str, Any]) -> int:
        """Add processing error."""
        cursor = self.conn.execute("""
            INSERT INTO processing_errors (
                file_id, stage, error_type, error_message, traceback
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            file_id,
            data.get('stage'),
            data.get('error_type'),
            data.get('error_message'),
            data.get('traceback')
        ))

        self._commit()
        return cursor.lastrowid

    def get_processing_errors(self, file_id: int) -> List[Dict[str, Any]]:
        """Get all processing errors for a file."""
        cursor = self.conn.execute(
            "SELECT * FROM processing_errors WHERE file_id = ? ORDER BY created_at",
            (file_id,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_errors_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        """Get all errors for a processing stage."""
        cursor = self.conn.execute(
            "SELECT * FROM processing_errors WHERE stage = ? ORDER BY created_at",
            (stage,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Processing runs

    def create_processing_run(self, data: Dict[str, Any]) -> int:
        """Create a new processing run."""
        parameters = data.get('parameters')
        if parameters and not isinstance(parameters, str):
            parameters = json.dumps(parameters)

        cursor = self.conn.execute("""
            INSERT INTO processing_runs (run_type, parameters, status)
            VALUES (?, ?, ?)
        """, (
            data.get('run_type'),
            parameters,
            data.get('status', 'running')
        ))

        self._commit()
        return cursor.lastrowid

    def update_processing_run(self, run_id: int, data: Dict[str, Any]):
        """Update processing run."""
        updates = []
        params = []

        for key, value in data.items():
            if key == 'parameters' and not isinstance(value, str):
                value = json.dumps(value)
            updates.append(f"{key} = ?")
            params.append(value)

        params.append(run_id)

        query = f"UPDATE processing_runs SET {', '.join(updates)} WHERE id = ?"
        self.conn.execute(query, params)
        self._commit()

    def get_processing_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get processing run by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM processing_runs WHERE id = ?",
            (run_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_recent_processing_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processing runs."""
        cursor = self.conn.execute(
            "SELECT * FROM processing_runs ORDER BY started_at DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Statistics

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}

        # Total files
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM files")
        stats['total_files'] = cursor.fetchone()[0]

        # Files by type
        cursor = self.conn.execute("""
            SELECT file_type, COUNT(*) as count
            FROM files
            WHERE file_type IS NOT NULL
            GROUP BY file_type
        """)
        stats['by_type'] = {row[0]: row[1] for row in cursor.fetchall()}

        # Files by category
        cursor = self.conn.execute("""
            SELECT category, COUNT(*) as count
            FROM files
            WHERE category IS NOT NULL
            GROUP BY category
        """)
        stats['by_category'] = {row[0]: row[1] for row in cursor.fetchall()}

        # Total detected objects
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM detected_objects")
        stats['total_detected_objects'] = cursor.fetchone()[0]

        # Total faces
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM faces")
        stats['total_faces'] = cursor.fetchone()[0]

        return stats

    # Transaction support

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Usage:
            with db.transaction():
                db.upsert_file(...)
                db.add_operation(...)
        """
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    # Helper methods

    def _commit(self):
        """Commit if not in a transaction context."""
        # Check if we're in a transaction context (Python 3.6+)
        # For older versions or if not in transaction, commit
        try:
            if hasattr(self.conn, 'in_transaction') and self.conn.in_transaction:
                # We're in a transaction context, don't commit
                return
        except Exception:
            pass
        # Not in transaction, safe to commit
        self.conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary."""
        if not row:
            return None

        result = dict(row)

        # Don't parse JSON fields automatically - let the caller do it
        # This matches the test expectations where tags/metadata are JSON strings

        # Convert boolean fields
        if 'undone' in result:
            result['undone'] = bool(result['undone'])

        if 'is_nsfw' in result:
            result['is_nsfw'] = bool(result['is_nsfw'])

        return result

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
