"""
FileForge Discovery Module

Handles file system discovery, hashing, and processing queue management.
"""

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Generator, List, Optional, Dict
import os
import hashlib


class FileType(Enum):
    """Supported file types for processing."""
    IMAGE = "image"
    DOCUMENT = "document"
    PDF = "pdf"
    TEXT = "text"
    CODE = "code"
    UNKNOWN = "unknown"


# Extension to FileType mapping
EXTENSION_MAP = {
    # Images
    '.jpg': FileType.IMAGE,
    '.jpeg': FileType.IMAGE,
    '.png': FileType.IMAGE,
    '.gif': FileType.IMAGE,
    '.webp': FileType.IMAGE,
    '.bmp': FileType.IMAGE,
    '.tiff': FileType.IMAGE,
    '.tif': FileType.IMAGE,
    '.svg': FileType.IMAGE,
    # Documents
    '.pdf': FileType.DOCUMENT,
    '.docx': FileType.DOCUMENT,
    '.doc': FileType.DOCUMENT,
    '.odt': FileType.DOCUMENT,
    # Text
    '.txt': FileType.TEXT,
    '.md': FileType.TEXT,
    '.markdown': FileType.TEXT,
    '.rst': FileType.TEXT,
    # Code
    '.py': FileType.CODE,
    '.js': FileType.CODE,
    '.java': FileType.CODE,
    '.cpp': FileType.CODE,
    '.c': FileType.CODE,
    '.ts': FileType.CODE,
}


def get_file_type(extension: str) -> FileType:
    """
    Get FileType for a file extension.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        FileType enum value, UNKNOWN if extension not mapped
    """
    if not extension.startswith('.'):
        extension = f'.{extension}'
    return EXTENSION_MAP.get(extension.lower(), FileType.UNKNOWN)


@dataclass
class DiscoveredFile:
    """Represents a discovered file with metadata."""
    path: Path
    file_type: FileType
    size: int
    hash: str
    modified_time: datetime
    needs_processing: bool
    priority: int
    metadata: Optional[Dict] = None


class FileDiscovery:
    """
    Discovers files in directory structure and generates DiscoveredFile objects.
    """

    def __init__(self, config, database=None):
        """
        Initialize file discovery.

        Args:
            config: Configuration object with scanning settings
            database: Optional database for checking existing files
        """
        self.config = config
        self.database = database
        self._stats = {
            'total_files': 0,
            'by_type': {},
            'total_size': 0,
            'needs_processing': 0
        }

    def discover(self, root_path: Path) -> Generator[DiscoveredFile, None, None]:
        """
        Discover files in directory and yield DiscoveredFile objects.

        Args:
            root_path: Root directory to scan

        Yields:
            DiscoveredFile objects for each discovered file

        Raises:
            FileNotFoundError: If root_path does not exist
        """
        root_path = Path(root_path)
        if not root_path.exists():
            raise FileNotFoundError(f"Path does not exist: {root_path}")

        # Reset stats for new discovery
        self._stats = {
            'total_files': 0,
            'by_type': {},
            'total_size': 0,
            'needs_processing': 0
        }

        # Use os.scandir for performance
        if self.config.scanning.recursive:
            yield from self._discover_recursive(root_path)
        else:
            yield from self._discover_single_level(root_path)

    def _discover_recursive(self, root_path: Path) -> Generator[DiscoveredFile, None, None]:
        """Recursively discover files in directory tree."""
        for entry in os.scandir(root_path):
            if entry.is_file():
                file_path = Path(entry.path)
                discovered = self._process_file(file_path)
                if discovered:
                    yield discovered
            elif entry.is_dir():
                dir_path = Path(entry.path)
                # Skip hidden directories and excluded paths
                if not entry.name.startswith('.') and not self._should_skip(dir_path):
                    yield from self._discover_recursive(dir_path)

    def _discover_single_level(self, root_path: Path) -> Generator[DiscoveredFile, None, None]:
        """Discover files in single directory level only."""
        for entry in os.scandir(root_path):
            if entry.is_file():
                file_path = Path(entry.path)
                discovered = self._process_file(file_path)
                if discovered:
                    yield discovered

    def _process_file(self, path: Path) -> Optional[DiscoveredFile]:
        """
        Process a single file and create DiscoveredFile if valid.

        Args:
            path: File path to process

        Returns:
            DiscoveredFile or None if file should be skipped
        """
        # Skip hidden files if configured
        if not getattr(self.config.scanning, 'include_hidden', False) and path.name.startswith('.'):
            return None

        # Skip excluded paths
        if self._should_skip(path):
            return None

        # Get file extension
        extension = path.suffix.lower()

        # Filter by extensions if specified
        # Try both attribute names, preferring 'allowed_extensions' over 'extensions'
        allowed_exts = None
        if hasattr(self.config.scanning, 'allowed_extensions'):
            ext_val = self.config.scanning.allowed_extensions
            # Check if it's a real list/tuple (not a MagicMock)
            if isinstance(ext_val, (list, tuple)):
                allowed_exts = ext_val

        if allowed_exts is None and hasattr(self.config.scanning, 'extensions'):
            ext_val = self.config.scanning.extensions
            if isinstance(ext_val, (list, tuple)):
                allowed_exts = ext_val

        # Apply extension filtering if we have a valid list
        if allowed_exts:
            # Case-insensitive comparison: normalize allowed extensions to lowercase
            allowed_exts_lower = [str(ext).lower() for ext in allowed_exts]
            if extension not in allowed_exts_lower:
                return None

        # Get file stats
        try:
            stat = path.stat()
            size = stat.st_size
            modified_time = datetime.fromtimestamp(stat.st_mtime)
        except (OSError, PermissionError):
            return None

        # Check file size limit
        max_size_bytes = self.config.scanning.max_size_mb * 1024 * 1024
        if size > max_size_bytes:
            return None

        # Determine file type
        file_type = get_file_type(extension)

        # Compute hash
        try:
            file_hash = self._compute_hash(path)
        except (OSError, PermissionError):
            return None

        # Check if needs processing
        needs_processing = self._check_needs_processing(path, file_hash)

        # Calculate priority (simple: larger files = higher priority)
        priority = min(size // 1024, 100)  # KB-based priority, capped at 100

        # Create discovered file
        discovered = DiscoveredFile(
            path=path,
            file_type=file_type,
            size=size,
            hash=file_hash,
            modified_time=modified_time,
            needs_processing=needs_processing,
            priority=priority
        )

        # Update stats
        self._stats['total_files'] += 1
        self._stats['total_size'] += size
        file_type_str = file_type.value
        self._stats['by_type'][file_type_str] = self._stats['by_type'].get(file_type_str, 0) + 1
        if needs_processing:
            self._stats['needs_processing'] += 1

        return discovered

    def _should_skip(self, path: Path) -> bool:
        """
        Check if path should be skipped based on exclusions.

        Args:
            path: Path to check

        Returns:
            True if path should be skipped
        """
        if not self.config.scanning.exclusions:
            return False

        path_str = str(path)
        for exclusion in self.config.scanning.exclusions:
            # Support both exact matches and pattern matching
            if exclusion in path_str or path.name == exclusion:
                return True

        return False

    def _compute_hash(self, path: Path) -> str:
        """
        Compute SHA-256 hash of file using chunked reading.

        Args:
            path: File path to hash

        Returns:
            Hexadecimal hash string
        """
        sha256 = hashlib.sha256()
        chunk_size = 8192  # 8KB chunks

        with open(path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)

        return sha256.hexdigest()

    def _check_needs_processing(self, path: Path, hash: str) -> bool:
        """
        Check if file needs processing based on database records.

        Args:
            path: File path
            hash: File hash

        Returns:
            True if file needs processing (new or changed)
        """
        if not self.database:
            # No database, assume all files need processing
            return True

        # Check if file exists in database by hash
        existing = self.database.get_file_by_hash(hash)

        if not existing:
            # New file (hash not in database), needs processing
            return True

        # File with this hash exists in database, already processed
        return False

    def get_stats(self) -> dict:
        """
        Get discovery statistics.

        Returns:
            Dictionary with total_files, by_type, and total_size
        """
        return self._stats


class ProcessingQueue:
    """
    Queue for managing discovered files and batch processing.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize processing queue.

        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queue: List[DiscoveredFile] = []

    def add(self, file: DiscoveredFile):
        """
        Add file to queue if space available.

        Args:
            file: DiscoveredFile to add
        """
        if len(self._queue) < self.max_size:
            self._queue.append(file)

    def get_batch(self, batch_size: int, file_type: FileType = None) -> List[DiscoveredFile]:
        """
        Get batch of files from queue.

        Args:
            batch_size: Number of files to retrieve
            file_type: Optional file type filter

        Returns:
            List of DiscoveredFile objects (up to batch_size)
        """
        if file_type:
            # Filter by file type
            filtered = [f for f in self._queue if f.file_type == file_type]
            return filtered[:batch_size]

        # Return batch without filtering
        return self._queue[:batch_size]

    def prioritize(self):
        """Sort queue by priority (higher priority first)."""
        self._queue.sort(key=lambda f: f.priority, reverse=True)

    def __len__(self):
        """Return queue length."""
        return len(self._queue)
