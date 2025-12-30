"""Results Gallery View."""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QComboBox, QLineEdit, QToolBar, QScrollArea,
    QGridLayout, QFrame, QMenu, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QAction

from ..viewmodels import ResultsViewModel


class ResultCard(QFrame):
    """Card widget for displaying a result item."""

    clicked = Signal(dict)

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self._data = data
        self.setObjectName("result_card")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(200, 240)
        self.setStyleSheet("""
            QFrame#result_card {
                background-color: white;
                border: 1px solid #E1E1E1;
                border-radius: 8px;
            }
            QFrame#result_card:hover {
                border-color: #0078D4;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Thumbnail placeholder
        thumb = QLabel("📄")
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setStyleSheet("font-size: 48px; padding: 16px;")
        thumb.setFixedHeight(120)
        layout.addWidget(thumb)

        # File name
        name = QLabel(data.get('name', 'Unknown'))
        name.setStyleSheet("font-weight: 500; font-size: 13px;")
        name.setWordWrap(True)
        name.setMaximumHeight(40)
        layout.addWidget(name)

        # Category tag
        category = data.get('category', 'other')
        tag = QLabel(category.title())
        tag.setStyleSheet("""
            background-color: #E1F5FE;
            color: #0277BD;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
        """)
        tag.setFixedHeight(20)
        layout.addWidget(tag)

        layout.addStretch()

    def mousePressEvent(self, event):
        """Handle click."""
        self.clicked.emit(self._data)


class ResultsView(QWidget):
    """Results gallery view for browsing processed files.

    Features:
    - Grid/list view of processed files
    - Filtering by category and tags
    - Full-text search
    - Undo operations
    - Export functionality
    """

    file_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewmodel = ResultsViewModel()
        self._setup_ui()
        self._connect_signals()
        self.refresh()

    def _setup_ui(self):
        """Setup results UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QHBoxLayout()
        header = QLabel("Results Gallery")
        header.setStyleSheet("font-size: 24px; font-weight: 600;")
        header_layout.addWidget(header)
        header_layout.addStretch()

        # Export button
        export_btn = QPushButton("Export")
        export_menu = QMenu(export_btn)
        export_json = QAction("Export as JSON", self)
        export_json.triggered.connect(lambda: self._export('json'))
        export_menu.addAction(export_json)
        export_csv = QAction("Export as CSV", self)
        export_csv.triggered.connect(lambda: self._export('csv'))
        export_menu.addAction(export_csv)
        export_btn.setMenu(export_menu)
        header_layout.addWidget(export_btn)

        # Undo button
        self._undo_btn = QPushButton("↩ Undo Last")
        self._undo_btn.clicked.connect(self._undo_last)
        self._undo_btn.setEnabled(False)
        header_layout.addWidget(self._undo_btn)

        layout.addLayout(header_layout)

        # Filter toolbar
        filter_layout = QHBoxLayout()

        # Search
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search files...")
        self._search_input.textChanged.connect(self._on_search)
        self._search_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #E1E1E1;
                border-radius: 4px;
            }
            QLineEdit:focus {
                border-color: #0078D4;
            }
        """)
        filter_layout.addWidget(self._search_input)

        # Category filter
        filter_layout.addWidget(QLabel("Category:"))
        self._category_combo = QComboBox()
        self._category_combo.addItems(['All', 'Images', 'Documents', 'Text', 'Other'])
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        filter_layout.addWidget(self._category_combo)

        # Sort
        filter_layout.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(['Date (Newest)', 'Date (Oldest)', 'Name', 'Category'])
        self._sort_combo.currentTextChanged.connect(self._on_sort_changed)
        filter_layout.addWidget(self._sort_combo)

        # Clear filters
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self._clear_filters)
        filter_layout.addWidget(clear_btn)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Results grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(16)
        self._grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll.setWidget(self._grid_widget)
        layout.addWidget(scroll)

        # Status bar
        status_layout = QHBoxLayout()
        self._count_label = QLabel("0 results")
        status_layout.addWidget(self._count_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

    def _connect_signals(self):
        """Connect viewmodel signals."""
        self._viewmodel.data_changed.connect(self._update_display)
        self._viewmodel.filter_changed.connect(self._update_display)

    def _update_display(self):
        """Update results display."""
        # Clear existing cards
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add result cards
        files = self._viewmodel.files
        cols = 4  # Cards per row
        for i, file_data in enumerate(files):
            row = i // cols
            col = i % cols
            card = ResultCard(file_data)
            card.clicked.connect(self._on_card_click)
            self._grid_layout.addWidget(card, row, col)

        # Update count
        self._count_label.setText(f"{len(files)} result{'s' if len(files) != 1 else ''}")

        # Update undo button
        self._undo_btn.setEnabled(self._viewmodel.can_undo)

    def _on_search(self, text: str):
        """Handle search input."""
        self._viewmodel.search(text)

    def _on_category_changed(self, text: str):
        """Handle category filter change."""
        category = None if text == 'All' else text.lower()
        self._viewmodel.set_category_filter(category)

    def _on_sort_changed(self, text: str):
        """Handle sort change."""
        sort_map = {
            'Date (Newest)': ('processed_at', 'desc'),
            'Date (Oldest)': ('processed_at', 'asc'),
            'Name': ('name', 'asc'),
            'Category': ('category', 'asc')
        }
        field, order = sort_map.get(text, ('processed_at', 'desc'))
        self._viewmodel.set_sort(field, order)

    def _clear_filters(self):
        """Clear all filters."""
        self._search_input.clear()
        self._category_combo.setCurrentIndex(0)
        self._viewmodel.clear_filters()

    def _on_card_click(self, data: dict):
        """Handle card click."""
        self.file_selected.emit(data)

    def _undo_last(self):
        """Undo last operation."""
        if self._viewmodel.undo():
            self.refresh()

    def _export(self, format: str):
        """Export results."""
        if format == 'json':
            path, _ = QFileDialog.getSaveFileName(
                self, "Export as JSON",
                str(Path.home() / "fileforge_results.json"),
                "JSON Files (*.json)"
            )
            if path:
                self._viewmodel.export_json(Path(path))

        elif format == 'csv':
            path, _ = QFileDialog.getSaveFileName(
                self, "Export as CSV",
                str(Path.home() / "fileforge_results.csv"),
                "CSV Files (*.csv)"
            )
            if path:
                self._viewmodel.export_csv(Path(path))

    def refresh(self):
        """Refresh results."""
        self._viewmodel.refresh()
