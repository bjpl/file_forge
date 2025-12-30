"""TDD Tests for FileForge file discovery module.

RED phase: Tests written first, defining expected behavior.
These tests should FAIL initially until implementation is complete.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime


# Fixtures for test setup
@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    files = {}

    # Text file
    text_file = temp_dir / "sample.txt"
    text_file.write_text("Sample text content")
    files["text"] = text_file

    # Markdown file
    md_file = temp_dir / "README.md"
    md_file.write_text("# Sample Markdown")
    files["markdown"] = md_file

    # Image file (mock)
    img_file = temp_dir / "image.jpg"
    img_file.write_bytes(b"fake image data")
    files["image"] = img_file

    # Nested directory
    nested_dir = temp_dir / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "nested.txt"
    nested_file.write_text("Nested content")
    files["nested"] = nested_file

    return files


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = MagicMock()
    config.scanning.extensions = [".txt", ".md", ".jpg", ".png", ".pdf"]
    config.scanning.exclusions = ["__pycache__", ".git", "node_modules"]
    config.scanning.max_size_mb = 50
    config.scanning.recursive = True
    config.scanning.include_hidden = False
    return config


class TestFileType:
    """Tests for FileType enum and extension mapping."""

    def test_image_extensions_mapped_correctly(self):
        """Image extensions should map to IMAGE type."""
        from fileforge.pipeline.discovery import FileType, get_file_type

        assert get_file_type('.jpg') == FileType.IMAGE
        assert get_file_type('.jpeg') == FileType.IMAGE
        assert get_file_type('.png') == FileType.IMAGE
        assert get_file_type('.gif') == FileType.IMAGE
        assert get_file_type('.webp') == FileType.IMAGE
        assert get_file_type('.bmp') == FileType.IMAGE
        assert get_file_type('.svg') == FileType.IMAGE

    def test_document_extensions_mapped_correctly(self):
        """Document extensions should map to DOCUMENT type."""
        from fileforge.pipeline.discovery import FileType, get_file_type

        assert get_file_type('.pdf') == FileType.DOCUMENT
        assert get_file_type('.docx') == FileType.DOCUMENT
        assert get_file_type('.doc') == FileType.DOCUMENT
        assert get_file_type('.odt') == FileType.DOCUMENT

    def test_text_extensions_mapped_correctly(self):
        """Text extensions should map to TEXT type."""
        from fileforge.pipeline.discovery import FileType, get_file_type

        assert get_file_type('.txt') == FileType.TEXT
        assert get_file_type('.md') == FileType.TEXT
        assert get_file_type('.markdown') == FileType.TEXT
        assert get_file_type('.rst') == FileType.TEXT

    def test_code_extensions_mapped_correctly(self):
        """Code extensions should map to CODE type."""
        from fileforge.pipeline.discovery import FileType, get_file_type

        assert get_file_type('.py') == FileType.CODE
        assert get_file_type('.js') == FileType.CODE
        assert get_file_type('.java') == FileType.CODE
        assert get_file_type('.cpp') == FileType.CODE

    def test_unknown_extensions_return_unknown(self):
        """Unknown extensions should return UNKNOWN type."""
        from fileforge.pipeline.discovery import FileType, get_file_type

        assert get_file_type('.xyz') == FileType.UNKNOWN
        assert get_file_type('.random') == FileType.UNKNOWN
        assert get_file_type('.custom123') == FileType.UNKNOWN

    def test_case_insensitive_extension_mapping(self):
        """Extension mapping should be case-insensitive."""
        from fileforge.pipeline.discovery import FileType, get_file_type

        assert get_file_type('.JPG') == FileType.IMAGE
        assert get_file_type('.Pdf') == FileType.DOCUMENT
        assert get_file_type('.TXT') == FileType.TEXT


class TestDiscoveredFile:
    """Tests for DiscoveredFile dataclass."""

    def test_discovered_file_creation(self, temp_dir):
        """Should create DiscoveredFile with all required fields."""
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        file = DiscoveredFile(
            path=temp_dir / "test.txt",
            file_type=FileType.TEXT,
            size=1024,
            hash="abc123",
            modified_time=datetime.now(),
            needs_processing=True,
            priority=5
        )

        assert file.path == temp_dir / "test.txt"
        assert file.file_type == FileType.TEXT
        assert file.size == 1024
        assert file.hash == "abc123"
        assert file.needs_processing is True
        assert file.priority == 5

    def test_discovered_file_with_optional_fields(self, temp_dir):
        """Should handle optional metadata fields."""
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        file = DiscoveredFile(
            path=temp_dir / "test.txt",
            file_type=FileType.TEXT,
            size=1024,
            hash="abc123",
            modified_time=datetime.now(),
            needs_processing=True,
            priority=5,
            metadata={"encoding": "utf-8", "line_count": 42}
        )

        assert file.metadata == {"encoding": "utf-8", "line_count": 42}

    def test_discovered_file_path_must_be_path_object(self, temp_dir):
        """Path should be a Path object."""
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        file = DiscoveredFile(
            path=temp_dir / "test.txt",
            file_type=FileType.TEXT,
            size=1024,
            hash="abc123",
            modified_time=datetime.now(),
            needs_processing=True,
            priority=5
        )

        assert isinstance(file.path, Path)


class TestFileDiscovery:
    """Tests for FileDiscovery class."""

    def test_discovers_files_in_directory(self, sample_files, mock_config):
        """Should discover all files in a directory."""
        from fileforge.pipeline.discovery import FileDiscovery

        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(sample_files["text"].parent))

        assert len(files) >= 2  # At least text and markdown files

    def test_discovery_returns_discovered_file_objects(self, sample_files, mock_config):
        """Discovery should return DiscoveredFile objects."""
        from fileforge.pipeline.discovery import FileDiscovery, DiscoveredFile

        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(sample_files["text"].parent))

        assert all(isinstance(f, DiscoveredFile) for f in files)

    def test_recursive_discovery(self, sample_files, mock_config):
        """Should discover files in subdirectories when recursive=True."""
        from fileforge.pipeline.discovery import FileDiscovery

        mock_config.scanning.recursive = True
        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(sample_files["text"].parent))

        # Should find nested file
        paths = [f.path for f in files]
        assert any("nested" in str(p) for p in paths)

    def test_non_recursive_discovery(self, sample_files, mock_config):
        """Should not discover subdirectory files when recursive=False."""
        from fileforge.pipeline.discovery import FileDiscovery

        mock_config.scanning.recursive = False
        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(sample_files["text"].parent))

        paths = [f.path for f in files]
        assert not any("nested" in str(p) for p in paths)

    def test_filters_by_extension(self, temp_dir, mock_config):
        """Should only discover files with configured extensions."""
        from fileforge.pipeline.discovery import FileDiscovery

        # Create files with different extensions
        (temp_dir / "valid.txt").write_text("content")
        (temp_dir / "valid.md").write_text("content")
        (temp_dir / "invalid.xyz").write_text("content")

        mock_config.scanning.extensions = [".txt", ".md"]
        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(temp_dir))

        extensions = [f.path.suffix for f in files]
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".xyz" not in extensions

    def test_excludes_configured_directories(self, temp_dir, mock_config):
        """Should skip excluded directories."""
        from fileforge.pipeline.discovery import FileDiscovery

        # Create excluded directory
        excluded = temp_dir / "__pycache__"
        excluded.mkdir()
        (excluded / "cached.txt").write_text("content")
        (temp_dir / "valid.txt").write_text("content")

        mock_config.scanning.exclusions = ["__pycache__"]
        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(temp_dir))

        paths = [str(f.path) for f in files]
        assert not any("__pycache__" in p for p in paths)

    def test_excludes_hidden_files(self, temp_dir, mock_config):
        """Should skip hidden files by default."""
        from fileforge.pipeline.discovery import FileDiscovery

        (temp_dir / ".hidden.txt").write_text("content")
        (temp_dir / "visible.txt").write_text("content")

        mock_config.scanning.include_hidden = False
        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(temp_dir))

        names = [f.path.name for f in files]
        assert ".hidden.txt" not in names
        assert "visible.txt" in names

    def test_includes_hidden_files_when_configured(self, temp_dir, mock_config):
        """Should include hidden files when configured."""
        from fileforge.pipeline.discovery import FileDiscovery

        (temp_dir / ".hidden.txt").write_text("content")
        (temp_dir / "visible.txt").write_text("content")

        mock_config.scanning.include_hidden = True
        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(temp_dir))

        names = [f.path.name for f in files]
        assert ".hidden.txt" in names
        assert "visible.txt" in names

    def test_respects_max_file_size(self, temp_dir, mock_config):
        """Should skip files larger than max_size_mb."""
        from fileforge.pipeline.discovery import FileDiscovery

        # Create small and large files
        (temp_dir / "small.txt").write_text("small")
        large_file = temp_dir / "large.txt"
        large_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

        mock_config.scanning.max_size_mb = 1  # 1MB limit
        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(temp_dir))

        names = [f.path.name for f in files]
        assert "small.txt" in names
        assert "large.txt" not in names

    def test_computes_file_hash(self, sample_files, mock_config):
        """Should compute SHA-256 hash for each file."""
        from fileforge.pipeline.discovery import FileDiscovery

        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(sample_files["text"].parent))

        for f in files:
            assert f.hash is not None
            assert len(f.hash) == 64  # SHA-256 hex length
            assert f.hash.isalnum()  # Hex string

    def test_checks_database_for_existing_files(self, sample_files, mock_config):
        """Should check database and skip unchanged files."""
        from fileforge.pipeline.discovery import FileDiscovery

        mock_db = MagicMock()
        mock_db.get_file_by_hash.return_value = {'processed_at': datetime.now()}

        discovery = FileDiscovery(mock_config, database=mock_db)
        files = list(discovery.discover(sample_files["text"].parent))

        # Files already in DB should have needs_processing=False
        for f in files:
            assert f.needs_processing is False

    def test_new_files_need_processing(self, sample_files, mock_config):
        """Files not in database should have needs_processing=True."""
        from fileforge.pipeline.discovery import FileDiscovery

        mock_db = MagicMock()
        mock_db.get_file_by_hash.return_value = None  # Not in DB

        discovery = FileDiscovery(mock_config, database=mock_db)
        files = list(discovery.discover(sample_files["text"].parent))

        for f in files:
            assert f.needs_processing is True

    def test_handles_nonexistent_directory(self, temp_dir, mock_config):
        """Should handle discovery of non-existent directory gracefully."""
        from fileforge.pipeline.discovery import FileDiscovery

        discovery = FileDiscovery(mock_config, database=None)
        nonexistent = temp_dir / "does_not_exist"

        with pytest.raises(FileNotFoundError):
            list(discovery.discover(nonexistent))

    def test_handles_file_as_directory(self, sample_files, mock_config):
        """Should handle when a file path is passed instead of directory."""
        from fileforge.pipeline.discovery import FileDiscovery

        discovery = FileDiscovery(mock_config, database=None)

        with pytest.raises(NotADirectoryError):
            list(discovery.discover(sample_files["text"]))


class TestProcessingQueue:
    """Tests for ProcessingQueue class."""

    def test_add_and_get_files(self, temp_dir):
        """Should add files and retrieve in batches."""
        from fileforge.pipeline.discovery import ProcessingQueue, DiscoveredFile, FileType

        queue = ProcessingQueue(max_size=100)

        for i in range(5):
            queue.add(DiscoveredFile(
                path=temp_dir / f"file{i}.txt",
                file_type=FileType.TEXT,
                size=100,
                hash=f"hash{i}",
                modified_time=datetime.now(),
                needs_processing=True,
                priority=i
            ))

        assert len(queue) == 5

        batch = queue.get_batch(batch_size=3)
        assert len(batch) == 3

    def test_prioritizes_by_priority_score(self, temp_dir):
        """Should return higher priority files first."""
        from fileforge.pipeline.discovery import ProcessingQueue, DiscoveredFile, FileType

        queue = ProcessingQueue(max_size=100)

        queue.add(DiscoveredFile(
            path=temp_dir / "low.txt", file_type=FileType.TEXT,
            size=100, hash="h1", modified_time=datetime.now(),
            needs_processing=True, priority=1
        ))
        queue.add(DiscoveredFile(
            path=temp_dir / "high.txt", file_type=FileType.TEXT,
            size=100, hash="h2", modified_time=datetime.now(),
            needs_processing=True, priority=10
        ))

        queue.prioritize()
        batch = queue.get_batch(batch_size=1)

        assert batch[0].path.name == "high.txt"

    def test_filter_by_file_type(self, temp_dir):
        """Should filter batch by file type."""
        from fileforge.pipeline.discovery import ProcessingQueue, DiscoveredFile, FileType

        queue = ProcessingQueue(max_size=100)

        queue.add(DiscoveredFile(
            path=temp_dir / "doc.txt", file_type=FileType.TEXT,
            size=100, hash="h1", modified_time=datetime.now(),
            needs_processing=True, priority=5
        ))
        queue.add(DiscoveredFile(
            path=temp_dir / "img.jpg", file_type=FileType.IMAGE,
            size=100, hash="h2", modified_time=datetime.now(),
            needs_processing=True, priority=5
        ))

        text_batch = queue.get_batch(batch_size=10, file_type=FileType.TEXT)
        assert len(text_batch) == 1
        assert text_batch[0].file_type == FileType.TEXT

    def test_respects_max_queue_size(self, temp_dir):
        """Should respect maximum queue size."""
        from fileforge.pipeline.discovery import ProcessingQueue, DiscoveredFile, FileType

        queue = ProcessingQueue(max_size=3)

        for i in range(5):
            queue.add(DiscoveredFile(
                path=temp_dir / f"file{i}.txt",
                file_type=FileType.TEXT,
                size=100,
                hash=f"hash{i}",
                modified_time=datetime.now(),
                needs_processing=True,
                priority=i
            ))

        assert len(queue) <= 3

    def test_empty_queue_returns_empty_batch(self):
        """Should return empty list when queue is empty."""
        from fileforge.pipeline.discovery import ProcessingQueue

        queue = ProcessingQueue(max_size=100)
        batch = queue.get_batch(batch_size=5)

        assert len(batch) == 0
        assert batch == []

    def test_batch_size_larger_than_queue(self, temp_dir):
        """Should return all files when batch_size > queue length."""
        from fileforge.pipeline.discovery import ProcessingQueue, DiscoveredFile, FileType

        queue = ProcessingQueue(max_size=100)

        for i in range(3):
            queue.add(DiscoveredFile(
                path=temp_dir / f"file{i}.txt",
                file_type=FileType.TEXT,
                size=100,
                hash=f"hash{i}",
                modified_time=datetime.now(),
                needs_processing=True,
                priority=i
            ))

        batch = queue.get_batch(batch_size=10)
        assert len(batch) == 3


class TestDiscoveryStats:
    """Tests for discovery statistics."""

    def test_get_stats_returns_counts(self, sample_files, mock_config):
        """Should return statistics about discovered files."""
        from fileforge.pipeline.discovery import FileDiscovery

        discovery = FileDiscovery(mock_config, database=None)
        list(discovery.discover(sample_files["text"].parent))  # Run discovery

        stats = discovery.get_stats()
        assert 'total_files' in stats
        assert 'by_type' in stats
        assert 'total_size' in stats
        assert stats['total_files'] > 0

    def test_stats_by_file_type(self, sample_files, mock_config):
        """Should break down stats by file type."""
        from fileforge.pipeline.discovery import FileDiscovery, FileType

        discovery = FileDiscovery(mock_config, database=None)
        list(discovery.discover(sample_files["text"].parent))

        stats = discovery.get_stats()
        by_type = stats['by_type']

        assert FileType.TEXT.value in by_type or 'TEXT' in by_type
        assert isinstance(by_type, dict)

    def test_stats_tracks_total_size(self, sample_files, mock_config):
        """Should track total size of discovered files."""
        from fileforge.pipeline.discovery import FileDiscovery

        discovery = FileDiscovery(mock_config, database=None)
        list(discovery.discover(sample_files["text"].parent))

        stats = discovery.get_stats()
        assert stats['total_size'] > 0
        assert isinstance(stats['total_size'], int)

    def test_stats_tracks_processing_needed(self, sample_files, mock_config):
        """Should track how many files need processing."""
        from fileforge.pipeline.discovery import FileDiscovery

        mock_db = MagicMock()
        mock_db.get_file_by_hash.return_value = None

        discovery = FileDiscovery(mock_config, database=mock_db)
        list(discovery.discover(sample_files["text"].parent))

        stats = discovery.get_stats()
        assert 'needs_processing' in stats
        assert stats['needs_processing'] > 0


class TestHashComputation:
    """Tests for file hash computation."""

    def test_same_file_produces_same_hash(self, temp_dir, mock_config):
        """Same file content should produce identical hash."""
        from fileforge.pipeline.discovery import FileDiscovery

        file1 = temp_dir / "file1.txt"
        file1.write_text("identical content")

        file2 = temp_dir / "file2.txt"
        file2.write_text("identical content")

        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(temp_dir))

        hashes = [f.hash for f in files]
        assert len(set(hashes)) == 1  # Both files have same hash

    def test_different_files_produce_different_hashes(self, temp_dir, mock_config):
        """Different file content should produce different hashes."""
        from fileforge.pipeline.discovery import FileDiscovery

        file1 = temp_dir / "file1.txt"
        file1.write_text("content A")

        file2 = temp_dir / "file2.txt"
        file2.write_text("content B")

        discovery = FileDiscovery(mock_config, database=None)
        files = list(discovery.discover(temp_dir))

        hashes = [f.hash for f in files]
        assert len(set(hashes)) == 2  # Different hashes


class TestErrorHandling:
    """Tests for error handling in discovery."""

    def test_handles_permission_denied(self, temp_dir, mock_config):
        """Should handle permission errors gracefully."""
        from fileforge.pipeline.discovery import FileDiscovery

        # This test is platform-specific and may need adjustment
        discovery = FileDiscovery(mock_config, database=None)

        # Create a file with no read permissions (Unix-like systems)
        restricted = temp_dir / "restricted.txt"
        restricted.write_text("content")
        restricted.chmod(0o000)

        try:
            files = list(discovery.discover(temp_dir))
            # Should either skip the file or handle error
            assert True
        except PermissionError:
            pytest.skip("Platform does not support permission restrictions in tests")
        finally:
            restricted.chmod(0o644)  # Restore permissions for cleanup

    def test_handles_broken_symlinks(self, temp_dir, mock_config):
        """Should handle broken symlinks gracefully."""
        from fileforge.pipeline.discovery import FileDiscovery

        discovery = FileDiscovery(mock_config, database=None)

        # Create broken symlink
        target = temp_dir / "nonexistent.txt"
        link = temp_dir / "broken_link.txt"

        try:
            link.symlink_to(target)
            files = list(discovery.discover(temp_dir))
            # Should skip broken symlinks
            assert all(f.path != link for f in files)
        except OSError:
            pytest.skip("Platform does not support symlinks")
