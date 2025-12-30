"""Dashboard View."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGridLayout, QPushButton, QScrollArea
)
from PySide6.QtCore import Qt, Signal

from ..viewmodels import DashboardViewModel


class StatCard(QFrame):
    """Statistics card widget."""

    def __init__(self, title: str, value: str, icon: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("stat_card")
        self.setStyleSheet("""
            QFrame#stat_card {
                background-color: #FFFFFF;
                border: 1px solid #E1E1E1;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        self.setMinimumSize(180, 100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(title_label)

        # Value
        self._value_label = QLabel(value)
        self._value_label.setStyleSheet("font-size: 28px; font-weight: 600; color: #0078D4;")
        layout.addWidget(self._value_label)

    def set_value(self, value: str):
        """Update the displayed value."""
        self._value_label.setText(value)


class QuickActionButton(QPushButton):
    """Quick action button with icon."""

    def __init__(self, label: str, icon: str = None, parent=None):
        super().__init__(label, parent)
        self.setMinimumSize(140, 80)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #F3F3F3;
                border: 1px solid #E1E1E1;
                border-radius: 8px;
                padding: 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #E5E5E5;
                border-color: #0078D4;
            }
            QPushButton:pressed {
                background-color: #0078D4;
                color: white;
            }
        """)


class DashboardView(QWidget):
    """Dashboard view showing statistics and quick actions.

    Displays:
    - Key statistics (files, faces, objects, db size)
    - Quick action buttons
    - Recent activity
    """

    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewmodel = DashboardViewModel()
        self._setup_ui()
        self._connect_signals()
        self.refresh()

    def _setup_ui(self):
        """Setup dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)

        # Header
        header = QLabel("Dashboard")
        header.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout.addWidget(header)

        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        self._stat_files = StatCard("Total Files", "0")
        stats_layout.addWidget(self._stat_files)

        self._stat_faces = StatCard("Faces Detected", "0")
        stats_layout.addWidget(self._stat_faces)

        self._stat_objects = StatCard("Objects Found", "0")
        stats_layout.addWidget(self._stat_objects)

        self._stat_db = StatCard("Database Size", "0 MB")
        stats_layout.addWidget(self._stat_db)

        stats_layout.addStretch()
        layout.addLayout(stats_layout)

        # Quick actions section
        actions_label = QLabel("Quick Actions")
        actions_label.setStyleSheet("font-size: 18px; font-weight: 500; margin-top: 16px;")
        layout.addWidget(actions_label)

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(12)

        scan_btn = QuickActionButton("Scan Folder")
        scan_btn.clicked.connect(lambda: self.action_requested.emit('scan'))
        actions_layout.addWidget(scan_btn)

        organize_btn = QuickActionButton("Organize Files")
        organize_btn.clicked.connect(lambda: self.action_requested.emit('organize'))
        actions_layout.addWidget(organize_btn)

        search_btn = QuickActionButton("Search Files")
        search_btn.clicked.connect(lambda: self.action_requested.emit('search'))
        actions_layout.addWidget(search_btn)

        watch_btn = QuickActionButton("Watch Folder")
        watch_btn.clicked.connect(lambda: self.action_requested.emit('watch'))
        actions_layout.addWidget(watch_btn)

        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        # Recent activity section
        activity_label = QLabel("Recent Activity")
        activity_label.setStyleSheet("font-size: 18px; font-weight: 500; margin-top: 16px;")
        layout.addWidget(activity_label)

        self._activity_container = QVBoxLayout()
        self._activity_container.setSpacing(8)
        layout.addLayout(self._activity_container)

        # Spacer
        layout.addStretch()

    def _connect_signals(self):
        """Connect viewmodel signals."""
        self._viewmodel.data_changed.connect(self._update_display)

    def _update_display(self):
        """Update display from viewmodel."""
        self._stat_files.set_value(str(self._viewmodel.total_files))
        self._stat_faces.set_value(str(self._viewmodel.total_faces))
        self._stat_objects.set_value(str(self._viewmodel.total_objects))
        self._stat_db.set_value(f"{self._viewmodel.database_size:.1f} MB")

        # Update activity
        self._update_activity()

    def _update_activity(self):
        """Update recent activity list."""
        # Clear existing
        while self._activity_container.count():
            item = self._activity_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add activity items
        for activity in self._viewmodel.recent_activity[:5]:
            label = QLabel(f"• {activity.get('type', 'Unknown')} - {activity.get('file', 'N/A')}")
            label.setStyleSheet("color: #666; font-size: 13px;")
            self._activity_container.addWidget(label)

        if not self._viewmodel.recent_activity:
            no_activity = QLabel("No recent activity")
            no_activity.setStyleSheet("color: #999; font-style: italic;")
            self._activity_container.addWidget(no_activity)

    def refresh(self):
        """Refresh dashboard data."""
        self._viewmodel.refresh()
