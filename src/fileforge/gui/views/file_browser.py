"""File Browser View."""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QComboBox, QLineEdit, QToolBar, QSplitter,
    QTreeView, QFileSystemModel, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon

from ..viewmodels import FileBrowserViewModel


class BreadcrumbBar(QWidget):
    """Breadcrumb navigation bar."""

    path_clicked = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(4)

    def set_path(self, path: Path):
        """Update breadcrumb display."""
        # Clear existing
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Build breadcrumbs
        parts = path.parts
        for i, part in enumerate(parts):
            btn = QPushButton(part if part != '/' else 'Root')
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn_path = Path(*parts[:i+1])
            btn.clicked.connect(lambda checked, p=btn_path: self.path_clicked.emit(p))
            btn.setStyleSheet("""
                QPushButton {
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: rgba(0, 0, 0, 0.05);
                }
            """)
            self._layout.addWidget(btn)

            if i < len(parts) - 1:
                sep = QLabel(">")
                sep.setStyleSheet("color: #999;")
                self._layout.addWidget(sep)

        self._layout.addStretch()


class FileBrowserView(QWidget):
    """File browser view for navigating and selecting files.

    Features:
    - Directory tree navigation
    - File list with filtering
    - Multiple view modes (list/grid/details)
    - Multi-select support
    - Drag-drop handling
    """

    files_selected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewmodel = FileBrowserViewModel()
        self._setup_ui()
        self._connect_signals()
        self.refresh()

    def _setup_ui(self):
        """Setup file browser UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #F9F9F9;
                border-bottom: 1px solid #E1E1E1;
                padding: 4px;
            }
        """)

        # Back button
        back_btn = QPushButton("←")
        back_btn.setFixedSize(32, 32)
        back_btn.clicked.connect(self._go_back)
        toolbar.addWidget(back_btn)

        # Up button
        up_btn = QPushButton("↑")
        up_btn.setFixedSize(32, 32)
        up_btn.clicked.connect(self._go_up)
        toolbar.addWidget(up_btn)

        # Refresh button
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedSize(32, 32)
        refresh_btn.clicked.connect(self.refresh)
        toolbar.addWidget(refresh_btn)

        toolbar.addSeparator()

        # Breadcrumb
        self._breadcrumb = BreadcrumbBar()
        toolbar.addWidget(self._breadcrumb)

        toolbar.addSeparator()

        # Filter dropdown
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(['All Files', 'Images', 'Documents', 'Text'])
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self._filter_combo)

        # View mode buttons
        list_btn = QPushButton("☰")
        list_btn.setFixedSize(32, 32)
        list_btn.setCheckable(True)
        list_btn.setChecked(True)
        list_btn.clicked.connect(lambda: self._set_view_mode('list'))
        toolbar.addWidget(list_btn)

        grid_btn = QPushButton("⊞")
        grid_btn.setFixedSize(32, 32)
        grid_btn.setCheckable(True)
        grid_btn.clicked.connect(lambda: self._set_view_mode('grid'))
        toolbar.addWidget(grid_btn)

        layout.addWidget(toolbar)

        # Main content with splitter
        splitter = QSplitter(Qt.Horizontal)

        # Directory tree
        self._tree_model = QFileSystemModel()
        self._tree_model.setRootPath("")
        self._tree_model.setFilter(
            self._tree_model.filter() | self._tree_model.Dirs
        )

        self._tree_view = QTreeView()
        self._tree_view.setModel(self._tree_model)
        self._tree_view.setRootIndex(self._tree_model.index(str(Path.home())))
        self._tree_view.hideColumn(1)  # Size
        self._tree_view.hideColumn(2)  # Type
        self._tree_view.hideColumn(3)  # Modified
        self._tree_view.setHeaderHidden(True)
        self._tree_view.setFixedWidth(200)
        self._tree_view.clicked.connect(self._on_tree_click)
        splitter.addWidget(self._tree_view)

        # File list
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setSpacing(2)
        self._file_list.itemSelectionChanged.connect(self._on_selection_changed)
        self._file_list.itemDoubleClicked.connect(self._on_item_double_click)
        splitter.addWidget(self._file_list)

        splitter.setSizes([200, 600])
        layout.addWidget(splitter)

        # Status bar
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(8, 4, 8, 4)
        self._status_label = QLabel("0 items")
        status_layout.addWidget(self._status_label)
        status_layout.addStretch()
        self._selection_label = QLabel("")
        status_layout.addWidget(self._selection_label)

        status_widget = QWidget()
        status_widget.setLayout(status_layout)
        status_widget.setStyleSheet("""
            QWidget {
                background-color: #F3F3F3;
                border-top: 1px solid #E1E1E1;
            }
        """)
        layout.addWidget(status_widget)

    def _connect_signals(self):
        """Connect viewmodel signals."""
        self._viewmodel.files_changed.connect(self._update_file_list)
        self._viewmodel.path_changed.connect(self._update_breadcrumb)
        self._breadcrumb.path_clicked.connect(self.navigate_to)

    def navigate_to(self, path: Path):
        """Navigate to a directory."""
        self._viewmodel.navigate_to(path)

    def _go_back(self):
        """Go to previous directory."""
        # Could implement history
        self._viewmodel.navigate_up()

    def _go_up(self):
        """Go to parent directory."""
        self._viewmodel.navigate_up()

    def _on_tree_click(self, index):
        """Handle tree item click."""
        path = Path(self._tree_model.filePath(index))
        if path.is_dir():
            self._viewmodel.navigate_to(path)

    def _on_filter_changed(self, text: str):
        """Handle filter change."""
        filter_map = {
            'All Files': 'all',
            'Images': 'images',
            'Documents': 'documents',
            'Text': 'text'
        }
        self._viewmodel.set_filter(filter_map.get(text, 'all'))

    def _set_view_mode(self, mode: str):
        """Set view mode."""
        self._viewmodel.set_view_mode(mode)
        if mode == 'grid':
            self._file_list.setViewMode(QListWidget.IconMode)
            self._file_list.setGridSize(QSize(100, 100))
        else:
            self._file_list.setViewMode(QListWidget.ListMode)
            self._file_list.setGridSize(QSize(-1, -1))

    def _update_file_list(self):
        """Update file list display."""
        self._file_list.clear()

        for file_info in self._viewmodel.files:
            item = QListWidgetItem()
            name = file_info.get('name', '')
            is_dir = file_info.get('is_dir', False)

            # Icon prefix
            icon = "📁" if is_dir else "📄"
            item.setText(f"{icon} {name}")
            item.setData(Qt.UserRole, file_info.get('path'))
            self._file_list.addItem(item)

        self._status_label.setText(f"{len(self._viewmodel.files)} items")

    def _update_breadcrumb(self, path: Path):
        """Update breadcrumb display."""
        self._breadcrumb.set_path(path)

    def _on_selection_changed(self):
        """Handle selection change."""
        selected = []
        for item in self._file_list.selectedItems():
            path = item.data(Qt.UserRole)
            if path:
                selected.append(Path(path))

        self._viewmodel.select_files(selected)
        self._selection_label.setText(f"{len(selected)} selected")
        self.files_selected.emit(selected)

    def _on_item_double_click(self, item: QListWidgetItem):
        """Handle double-click on item."""
        path = item.data(Qt.UserRole)
        if path:
            p = Path(path)
            if p.is_dir():
                self._viewmodel.navigate_to(p)

    def refresh(self):
        """Refresh file browser."""
        self._viewmodel.refresh()
        self._update_breadcrumb(self._viewmodel.current_path)
