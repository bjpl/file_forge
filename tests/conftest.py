"""Shared pytest fixtures for FileForge tests."""
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    files = {}
    # Create sample text file
    text_file = temp_dir / "sample.txt"
    text_file.write_text("This is sample text content for testing.")
    files["text"] = text_file

    # Create sample markdown
    md_file = temp_dir / "sample.md"
    md_file.write_text("# Heading\n\nSome markdown content.")
    files["markdown"] = md_file

    # Create nested structure
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    nested_file = subdir / "nested.txt"
    nested_file.write_text("Nested file content")
    files["nested"] = nested_file

    return files


@pytest.fixture
def mock_config():
    """Create mock configuration for tests."""
    from unittest.mock import MagicMock
    config = MagicMock()
    config.database.path = ":memory:"
    config.scanning.recursive = True
    config.scanning.extensions = [".txt", ".md", ".pdf", ".docx"]
    config.scanning.max_size_mb = 100
    config.scanning.exclusions = ["__pycache__", ".git", "node_modules"]
    config.processing.batch_size = 10
    config.logging.level = "DEBUG"
    return config


@pytest.fixture
def temp_db(temp_dir):
    """Create temporary database for tests."""
    return temp_dir / "test.db"
