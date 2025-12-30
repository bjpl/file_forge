"""Pytest configuration for GUI tests.

Provides fixtures for Qt testing with mocking support
for headless environments.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Check if we're in a headless environment
HEADLESS = True
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication
    # Try to create app - will fail in headless
    if not QCoreApplication.instance():
        try:
            app = QApplication([])
            HEADLESS = False
        except Exception:
            HEADLESS = True
except ImportError:
    HEADLESS = True


if HEADLESS:
    # Mock PySide6 modules for headless testing
    mock_qt = MagicMock()
    mock_qt.QObject = MagicMock
    mock_qt.Signal = MagicMock(return_value=MagicMock())
    mock_qt.Slot = MagicMock(return_value=lambda x: x)
    mock_qt.Qt = MagicMock()
    mock_qt.QSize = MagicMock()
    mock_qt.QSettings = MagicMock
    mock_qt.QThreadPool = MagicMock()
    mock_qt.QRunnable = MagicMock
    mock_qt.QPropertyAnimation = MagicMock()
    mock_qt.QEasingCurve = MagicMock()

    mock_widgets = MagicMock()
    mock_widgets.QWidget = MagicMock
    mock_widgets.QMainWindow = MagicMock
    mock_widgets.QApplication = MagicMock()
    mock_widgets.QVBoxLayout = MagicMock()
    mock_widgets.QHBoxLayout = MagicMock()
    mock_widgets.QGridLayout = MagicMock()
    mock_widgets.QFormLayout = MagicMock()
    mock_widgets.QLabel = MagicMock()
    mock_widgets.QPushButton = MagicMock()
    mock_widgets.QFrame = MagicMock
    mock_widgets.QListWidget = MagicMock()
    mock_widgets.QListWidgetItem = MagicMock()
    mock_widgets.QStackedWidget = MagicMock()
    mock_widgets.QStatusBar = MagicMock()
    mock_widgets.QMenuBar = MagicMock()
    mock_widgets.QMenu = MagicMock()
    mock_widgets.QToolBar = MagicMock()
    mock_widgets.QScrollArea = MagicMock()
    mock_widgets.QSplitter = MagicMock()
    mock_widgets.QTreeView = MagicMock()
    mock_widgets.QFileSystemModel = MagicMock()
    mock_widgets.QComboBox = MagicMock()
    mock_widgets.QLineEdit = MagicMock()
    mock_widgets.QTextEdit = MagicMock()
    mock_widgets.QProgressBar = MagicMock()
    mock_widgets.QGroupBox = MagicMock()
    mock_widgets.QSpinBox = MagicMock()
    mock_widgets.QDoubleSpinBox = MagicMock()
    mock_widgets.QCheckBox = MagicMock()
    mock_widgets.QMessageBox = MagicMock()
    mock_widgets.QFileDialog = MagicMock()
    mock_widgets.QSpacerItem = MagicMock()
    mock_widgets.QSizePolicy = MagicMock()
    mock_widgets.QAbstractItemView = MagicMock()

    mock_gui = MagicMock()
    mock_gui.QAction = MagicMock()
    mock_gui.QKeySequence = MagicMock()
    mock_gui.QIcon = MagicMock()
    mock_gui.QDragEnterEvent = MagicMock()
    mock_gui.QDropEvent = MagicMock()

    sys.modules['PySide6'] = MagicMock()
    sys.modules['PySide6.QtCore'] = mock_qt
    sys.modules['PySide6.QtWidgets'] = mock_widgets
    sys.modules['PySide6.QtGui'] = mock_gui


@pytest.fixture(scope='session', autouse=True)
def setup_qt_env():
    """Setup Qt environment for testing."""
    if not HEADLESS:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QCoreApplication
        if not QCoreApplication.instance():
            app = QApplication([])
            yield app
            app.quit()
        else:
            yield QCoreApplication.instance()
    else:
        yield None


@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    config = MagicMock()
    config.database = MagicMock()
    config.database.path = '~/.fileforge/fileforge.db'
    config.database.wal_mode = True
    config.scanning = MagicMock()
    config.scanning.recursive = True
    config.scanning.max_size_mb = 100
    config.processing = MagicMock()
    config.processing.workers = 4
    config.processing.batch_size = 10
    config.processing.timeout = 300
    config.ocr = MagicMock()
    config.ocr.engine = 'paddleocr'
    config.ocr.gpu_enabled = True
    config.llm = MagicMock()
    config.llm.model = 'llama3.2'
    config.llm.base_url = 'http://localhost:11434'
    config.llm.temperature = 0.7
    return config


@pytest.fixture
def mock_database():
    """Provide mock database."""
    db = MagicMock()
    db.get_stats.return_value = {
        'total_files': 100,
        'total_faces': 25,
        'detected_objects': 50,
        'database_size_mb': 15.5
    }
    db.list_operations.return_value = []
    db.query_files.return_value = []
    db.get_face_clusters.return_value = []
    db.get_last_operation.return_value = None
    return db


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory."""
    return tmp_path


@pytest.fixture
def sample_files(tmp_path):
    """Create sample test files."""
    files = []

    # Create image file
    img_file = tmp_path / "test_image.jpg"
    img_file.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)
    files.append(img_file)

    # Create document file
    doc_file = tmp_path / "test_doc.pdf"
    doc_file.write_bytes(b'%PDF-1.4' + b'\x00' * 100)
    files.append(doc_file)

    # Create text file
    txt_file = tmp_path / "test_text.txt"
    txt_file.write_text("Sample text content")
    files.append(txt_file)

    return files
