"""Main Window for FileForge GUI."""
from pathlib import Path
from typing import Optional, List, Dict
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QStatusBar, QMenuBar, QMenu,
    QLabel, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent

from .state import AppState
from .widgets import Sidebar
from .views import (
    DashboardView, FileBrowserView, ProcessingView,
    ResultsView, SettingsView
)


class MainWindow(QMainWindow):
    """Main application window.

    Features:
    - Sidebar navigation
    - Stacked content views
    - Dark/light theme support
    - Drag-drop file handling
    - Keyboard shortcuts
    """

    view_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_view = 'dashboard'
        self._theme = 'system'
        self._views: Dict[str, QWidget] = {}
        self._settings = QSettings('FileForge', 'FileForge')

        self._setup_window()
        self._setup_ui()
        self._setup_menus()
        self._setup_shortcuts()
        self._load_settings()
        self._connect_signals()

    def _setup_window(self):
        """Setup window properties."""
        self.setWindowTitle('FileForge - Intelligent File Organization')
        self.setMinimumSize(800, 600)
        self.resize(1200, 800)
        self.setAcceptDrops(True)

    def _setup_ui(self):
        """Setup main UI layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        main_layout.addWidget(self._sidebar)

        # Content area
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Stacked widget for views
        self._content_stack = QStackedWidget()
        content_layout.addWidget(self._content_stack)

        main_layout.addLayout(content_layout)

        # Create views
        self._create_views()

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)

    def _create_views(self):
        """Create all application views."""
        # Dashboard
        self._views['dashboard'] = DashboardView()
        self._content_stack.addWidget(self._views['dashboard'])

        # File Browser
        self._views['browser'] = FileBrowserView()
        self._content_stack.addWidget(self._views['browser'])

        # Processing
        self._views['processing'] = ProcessingView()
        self._content_stack.addWidget(self._views['processing'])

        # Results
        self._views['results'] = ResultsView()
        self._content_stack.addWidget(self._views['results'])

        # Settings
        self._views['settings'] = SettingsView()
        self._content_stack.addWidget(self._views['settings'])

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        open_action = QAction('&Open Folder...', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setObjectName('action_open')
        open_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction('&Quit', self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu('&View')

        dashboard_action = QAction('&Dashboard', self)
        dashboard_action.setShortcut('Ctrl+1')
        dashboard_action.triggered.connect(lambda: self.switch_view('dashboard'))
        view_menu.addAction(dashboard_action)

        browser_action = QAction('&File Browser', self)
        browser_action.setShortcut('Ctrl+2')
        browser_action.triggered.connect(lambda: self.switch_view('browser'))
        view_menu.addAction(browser_action)

        processing_action = QAction('&Processing', self)
        processing_action.setShortcut('Ctrl+3')
        processing_action.triggered.connect(lambda: self.switch_view('processing'))
        view_menu.addAction(processing_action)

        results_action = QAction('&Results', self)
        results_action.setShortcut('Ctrl+4')
        results_action.triggered.connect(lambda: self.switch_view('results'))
        view_menu.addAction(results_action)

        view_menu.addSeparator()

        refresh_action = QAction('&Refresh', self)
        refresh_action.setShortcut(QKeySequence.Refresh)
        refresh_action.triggered.connect(self._refresh_current_view)
        view_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu('&Help')

        about_action = QAction('&About FileForge', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Additional shortcuts handled via menu actions
        pass

    def _connect_signals(self):
        """Connect signals."""
        self._sidebar.navigation_requested.connect(self.switch_view)
        AppState.instance().view_changed.connect(self._on_app_view_changed)
        AppState.instance().progress_changed.connect(self._update_status_progress)

    def _load_settings(self):
        """Load window settings."""
        # Geometry
        geometry = self._settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)

        # Theme
        saved_theme = self._settings.value('theme', 'system')
        self.set_theme(saved_theme)

    def save_settings(self):
        """Save window settings."""
        self._settings.setValue('geometry', self.saveGeometry())
        self._settings.setValue('theme', self._theme)

    @property
    def sidebar(self) -> Sidebar:
        """Get sidebar widget."""
        return self._sidebar

    @property
    def content_stack(self) -> QStackedWidget:
        """Get content stack widget."""
        return self._content_stack

    @property
    def current_view(self) -> str:
        """Get current view ID."""
        return self._current_view

    @property
    def available_views(self) -> List[str]:
        """Get list of available view IDs."""
        return list(self._views.keys())

    @property
    def theme(self) -> str:
        """Get current theme."""
        return self._theme

    def switch_view(self, view_id: str):
        """Switch to a different view.

        Args:
            view_id: View identifier
        """
        if view_id in self._views:
            self._current_view = view_id
            self._content_stack.setCurrentWidget(self._views[view_id])
            self._sidebar.set_current(view_id)
            AppState.instance().set_current_view(view_id)
            self.view_changed.emit(view_id)

    def set_theme(self, theme: str):
        """Set application theme.

        Args:
            theme: Theme name ('light', 'dark', or 'system')
        """
        if theme not in ['light', 'dark', 'system']:
            return

        self._theme = theme
        AppState.instance().set_theme(theme)

        # Apply stylesheet based on theme
        if theme == 'dark':
            self._apply_dark_theme()
        elif theme == 'light':
            self._apply_light_theme()
        else:
            # System theme - use default
            self.setStyleSheet("")

    def _apply_dark_theme(self):
        """Apply dark theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QWidget#sidebar {
                background-color: #252525;
                border-right: 1px solid #333333;
            }
            QPushButton {
                background-color: #333333;
                color: #FFFFFF;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #404040;
            }
            QPushButton:pressed {
                background-color: #0078D4;
            }
            QStatusBar {
                background-color: #007ACC;
                color: #FFFFFF;
            }
        """)

    def _apply_light_theme(self):
        """Apply light theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
                color: #1E1E1E;
            }
            QWidget {
                background-color: #FFFFFF;
                color: #1E1E1E;
            }
            QWidget#sidebar {
                background-color: #F3F3F3;
                border-right: 1px solid #E1E1E1;
            }
            QPushButton {
                background-color: #F3F3F3;
                color: #1E1E1E;
                border: 1px solid #D1D1D1;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #E5E5E5;
            }
            QPushButton:pressed {
                background-color: #0078D4;
                color: #FFFFFF;
            }
            QStatusBar {
                background-color: #F3F3F3;
                border-top: 1px solid #E1E1E1;
            }
        """)

    def _open_folder(self):
        """Open folder dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Scan",
            str(Path.home())
        )
        if folder:
            # Switch to browser and navigate
            self.switch_view('browser')
            if hasattr(self._views['browser'], 'navigate_to'):
                self._views['browser'].navigate_to(Path(folder))

    def _refresh_current_view(self):
        """Refresh current view."""
        current = self._views.get(self._current_view)
        if current and hasattr(current, 'refresh'):
            current.refresh()

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About FileForge",
            "FileForge - Intelligent File Organization\n\n"
            "Version 1.0.0\n\n"
            "Organize your files using AI-powered analysis "
            "including OCR, face recognition, and content understanding."
        )

    def _on_app_view_changed(self, view_id: str):
        """Handle app state view change."""
        if view_id != self._current_view:
            self.switch_view(view_id)

    def _update_status_progress(self, progress: int):
        """Update status bar with progress."""
        if progress > 0 and progress < 100:
            self._status_label.setText(f"Processing... {progress}%")
        elif progress >= 100:
            self._status_label.setText("Ready")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        paths = [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]

        if paths:
            # Add to file selection
            AppState.instance().select_files(paths)
            # Switch to processing view
            self.switch_view('processing')

    def closeEvent(self, event):
        """Handle window close."""
        self.save_settings()
        super().closeEvent(event)
