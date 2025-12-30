"""FileForge GUI Application Entry Point."""
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from .main_window import MainWindow
from .state import AppState
from ..config import load_config
from ..storage.database import Database


def run_gui(config_path: Path = None):
    """Run the FileForge GUI application.

    Args:
        config_path: Optional path to configuration file
    """
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("FileForge")
    app.setOrganizationName("FileForge")
    app.setOrganizationDomain("fileforge.local")

    # Load configuration
    config = load_config(config_path)

    # Initialize database
    db_path = Path(config.database.path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    database = Database(db_path)

    # Initialize app state
    state = AppState.instance()
    state._config = config
    state._database = database

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    return app.exec()


def main():
    """Main entry point."""
    sys.exit(run_gui())


if __name__ == '__main__':
    main()
