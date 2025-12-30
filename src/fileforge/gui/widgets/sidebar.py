"""Sidebar Navigation Widget."""
from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFrame, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Signal, Qt, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon


class NavButton(QPushButton):
    """Navigation button with icon and label."""

    def __init__(self, icon_name: str, label: str, view_id: str, parent=None):
        super().__init__(parent)
        self.view_id = view_id
        self._label_text = label
        self._is_selected = False

        self.setText(label)
        self.setCheckable(True)
        self.setMinimumHeight(44)
        self.setCursor(Qt.PointingHandCursor)

        # Style
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 10px 16px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
            QPushButton:checked {
                background-color: rgba(0, 120, 212, 0.1);
                color: #0078D4;
                font-weight: 500;
            }
        """)

    def set_selected(self, selected: bool):
        """Set selection state."""
        self._is_selected = selected
        self.setChecked(selected)


class Sidebar(QWidget):
    """Sidebar navigation component.

    Provides vertical navigation with icons and labels.
    Supports collapsing to icon-only mode.
    """

    navigation_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_collapsed = False
        self._current_item = 'dashboard'
        self._expanded_width = 220
        self._collapsed_width = 60

        self._nav_items = [
            {'id': 'dashboard', 'label': 'Dashboard', 'icon': 'home'},
            {'id': 'browser', 'label': 'File Browser', 'icon': 'folder'},
            {'id': 'processing', 'label': 'Processing', 'icon': 'play'},
            {'id': 'results', 'label': 'Results', 'icon': 'image'},
            {'id': 'settings', 'label': 'Settings', 'icon': 'settings'},
        ]

        self._buttons: Dict[str, NavButton] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Setup sidebar UI."""
        self.setFixedWidth(self._expanded_width)
        self.setObjectName("sidebar")

        # Apply styling
        self.setStyleSheet("""
            QWidget#sidebar {
                background-color: #F3F3F3;
                border-right: 1px solid #E1E1E1;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(4)

        # App branding
        brand_layout = QHBoxLayout()
        brand_label = QLabel("FileForge")
        brand_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #0078D4;
                padding: 8px;
            }
        """)
        brand_layout.addWidget(brand_label)
        brand_layout.addStretch()
        layout.addLayout(brand_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #E1E1E1;")
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        layout.addSpacing(8)

        # Navigation buttons
        for item in self._nav_items:
            btn = NavButton(item['icon'], item['label'], item['id'])
            btn.clicked.connect(lambda checked, vid=item['id']: self._on_nav_click(vid))
            self._buttons[item['id']] = btn
            layout.addWidget(btn)

        # Spacer
        layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Collapse button
        self._collapse_btn = QPushButton("«")
        self._collapse_btn.setFixedSize(32, 32)
        self._collapse_btn.clicked.connect(self._toggle_collapse)
        self._collapse_btn.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)
        layout.addWidget(self._collapse_btn, alignment=Qt.AlignCenter)

        # Set initial selection
        self._update_selection()

    @property
    def nav_items(self) -> List[Dict[str, Any]]:
        """Get navigation items."""
        return self._nav_items.copy()

    @property
    def current_item(self) -> str:
        """Get current selected item."""
        return self._current_item

    @property
    def is_collapsed(self) -> bool:
        """Check if sidebar is collapsed."""
        return self._is_collapsed

    def navigate_to(self, view_id: str):
        """Navigate to a view.

        Args:
            view_id: View identifier
        """
        if view_id in self._buttons:
            self._current_item = view_id
            self._update_selection()
            self.navigation_requested.emit(view_id)

    def set_current(self, view_id: str):
        """Set current view without emitting signal.

        Args:
            view_id: View identifier
        """
        if view_id in self._buttons:
            self._current_item = view_id
            self._update_selection()

    def collapse(self):
        """Collapse sidebar to icon-only mode."""
        self._is_collapsed = True
        self.setFixedWidth(self._collapsed_width)
        self._collapse_btn.setText("»")

        # Hide labels
        for btn in self._buttons.values():
            btn.setText("")

    def expand(self):
        """Expand sidebar to full mode."""
        self._is_collapsed = False
        self.setFixedWidth(self._expanded_width)
        self._collapse_btn.setText("«")

        # Show labels
        for item in self._nav_items:
            if item['id'] in self._buttons:
                self._buttons[item['id']].setText(item['label'])

    def _toggle_collapse(self):
        """Toggle collapse state."""
        if self._is_collapsed:
            self.expand()
        else:
            self.collapse()

    def _on_nav_click(self, view_id: str):
        """Handle navigation click."""
        self.navigate_to(view_id)

    def _update_selection(self):
        """Update button selection states."""
        for view_id, btn in self._buttons.items():
            btn.set_selected(view_id == self._current_item)
