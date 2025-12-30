"""Settings View."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QStackedWidget, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QGroupBox, QFormLayout,
    QMessageBox, QFileDialog, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from pathlib import Path

from ..viewmodels import SettingsViewModel


class SettingWidget(QWidget):
    """Base widget for a single setting."""

    value_changed = Signal(str, object)

    def __init__(self, setting: dict, parent=None):
        super().__init__(parent)
        self._setting = setting
        self._key = setting['key']

    def get_value(self):
        """Get current value."""
        raise NotImplementedError

    def set_value(self, value):
        """Set current value."""
        raise NotImplementedError


class StringSetting(SettingWidget):
    """String setting with line edit."""

    def __init__(self, setting: dict, parent=None):
        super().__init__(setting, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._input = QLineEdit()
        self._input.textChanged.connect(
            lambda v: self.value_changed.emit(self._key, v)
        )
        layout.addWidget(self._input)

    def get_value(self):
        return self._input.text()

    def set_value(self, value):
        self._input.setText(str(value) if value else '')


class IntSetting(SettingWidget):
    """Integer setting with spin box."""

    def __init__(self, setting: dict, parent=None):
        super().__init__(setting, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._input = QSpinBox()
        self._input.setMinimum(setting.get('min', 0))
        self._input.setMaximum(setting.get('max', 999999))
        self._input.valueChanged.connect(
            lambda v: self.value_changed.emit(self._key, v)
        )
        layout.addWidget(self._input)

    def get_value(self):
        return self._input.value()

    def set_value(self, value):
        self._input.setValue(int(value) if value else 0)


class FloatSetting(SettingWidget):
    """Float setting with double spin box."""

    def __init__(self, setting: dict, parent=None):
        super().__init__(setting, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._input = QDoubleSpinBox()
        self._input.setMinimum(setting.get('min', 0.0))
        self._input.setMaximum(setting.get('max', 100.0))
        self._input.setDecimals(2)
        self._input.setSingleStep(0.1)
        self._input.valueChanged.connect(
            lambda v: self.value_changed.emit(self._key, v)
        )
        layout.addWidget(self._input)

    def get_value(self):
        return self._input.value()

    def set_value(self, value):
        self._input.setValue(float(value) if value else 0.0)


class BoolSetting(SettingWidget):
    """Boolean setting with checkbox."""

    def __init__(self, setting: dict, parent=None):
        super().__init__(setting, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._input = QCheckBox()
        self._input.stateChanged.connect(
            lambda s: self.value_changed.emit(self._key, s == Qt.Checked)
        )
        layout.addWidget(self._input)

    def get_value(self):
        return self._input.isChecked()

    def set_value(self, value):
        self._input.setChecked(bool(value))


class ChoiceSetting(SettingWidget):
    """Choice setting with combo box."""

    def __init__(self, setting: dict, parent=None):
        super().__init__(setting, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._input = QComboBox()
        self._input.addItems(setting.get('choices', []))
        self._input.currentTextChanged.connect(
            lambda v: self.value_changed.emit(self._key, v)
        )
        layout.addWidget(self._input)

    def get_value(self):
        return self._input.currentText()

    def set_value(self, value):
        index = self._input.findText(str(value) if value else '')
        if index >= 0:
            self._input.setCurrentIndex(index)


class PathSetting(SettingWidget):
    """Path setting with browse button."""

    def __init__(self, setting: dict, parent=None):
        super().__init__(setting, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._input = QLineEdit()
        self._input.textChanged.connect(
            lambda v: self.value_changed.emit(self._key, v)
        )
        layout.addWidget(self._input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory",
            self._input.text() or str(Path.home())
        )
        if path:
            self._input.setText(path)

    def get_value(self):
        return self._input.text()

    def set_value(self, value):
        self._input.setText(str(value) if value else '')


class SettingsView(QWidget):
    """Settings view for configuration.

    Features:
    - Categorized settings with sidebar
    - Type-appropriate input widgets
    - Validation before save
    - Reset to defaults
    """

    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewmodel = SettingsViewModel()
        self._widgets: dict = {}
        self._setup_ui()
        self._connect_signals()
        self._load_settings()

    def _setup_ui(self):
        """Setup settings UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Section list
        self._section_list = QListWidget()
        self._section_list.setFixedWidth(180)
        self._section_list.setStyleSheet("""
            QListWidget {
                background-color: #F9F9F9;
                border-right: 1px solid #E1E1E1;
                padding: 8px;
            }
            QListWidget::item {
                padding: 12px 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #E1F5FE;
                color: #0078D4;
            }
            QListWidget::item:hover {
                background-color: #F0F0F0;
            }
        """)
        self._section_list.currentRowChanged.connect(self._on_section_changed)
        layout.addWidget(self._section_list)

        # Content area
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(16)

        # Header
        self._header = QLabel("Settings")
        self._header.setStyleSheet("font-size: 24px; font-weight: 600;")
        content_layout.addWidget(self._header)

        # Settings stack
        self._content_stack = QStackedWidget()
        content_layout.addWidget(self._content_stack)

        # Build sections
        self._build_sections()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(reset_btn)

        self._save_btn = QPushButton("Save Changes")
        self._save_btn.clicked.connect(self._save_settings)
        self._save_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
        """)
        btn_layout.addWidget(self._save_btn)

        content_layout.addLayout(btn_layout)
        layout.addLayout(content_layout)

    def _build_sections(self):
        """Build settings sections."""
        for section in self._viewmodel.sections:
            # Add to list
            item = QListWidgetItem(section['name'])
            self._section_list.addItem(item)

            # Create section widget
            section_widget = self._create_section_widget(section)
            self._content_stack.addWidget(section_widget)

        if self._section_list.count() > 0:
            self._section_list.setCurrentRow(0)

    def _create_section_widget(self, section: dict) -> QWidget:
        """Create widget for a settings section."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        form_layout = QFormLayout(container)
        form_layout.setSpacing(16)
        form_layout.setContentsMargins(0, 0, 0, 0)

        for setting in section['settings']:
            label = QLabel(setting['label'])
            label.setStyleSheet("font-size: 13px;")

            widget = self._create_setting_widget(setting)
            if widget:
                self._widgets[setting['key']] = widget
                form_layout.addRow(label, widget)

        scroll.setWidget(container)
        return scroll

    def _create_setting_widget(self, setting: dict) -> SettingWidget:
        """Create appropriate widget for setting type."""
        setting_type = setting.get('type', 'str')

        widget_map = {
            'str': StringSetting,
            'int': IntSetting,
            'float': FloatSetting,
            'bool': BoolSetting,
            'choice': ChoiceSetting,
            'path': PathSetting,
        }

        widget_class = widget_map.get(setting_type, StringSetting)
        widget = widget_class(setting)
        widget.value_changed.connect(self._on_value_changed)
        return widget

    def _connect_signals(self):
        """Connect viewmodel signals."""
        self._viewmodel.settings_saved.connect(self._on_saved)
        self._viewmodel.settings_reset.connect(self._load_settings)

    def _on_section_changed(self, index: int):
        """Handle section change."""
        self._content_stack.setCurrentIndex(index)
        if index >= 0 and index < len(self._viewmodel.sections):
            self._header.setText(self._viewmodel.sections[index]['name'])

    def _on_value_changed(self, key: str, value):
        """Handle value change."""
        self._viewmodel.update_setting(key, value)

    def _load_settings(self):
        """Load current settings into widgets."""
        for key, widget in self._widgets.items():
            value = self._viewmodel.get_setting(key)
            widget.set_value(value)

    def _save_settings(self):
        """Save settings."""
        if self._viewmodel.save():
            QMessageBox.information(
                self, "Settings Saved",
                "Your settings have been saved successfully."
            )
            self.settings_changed.emit()
        else:
            QMessageBox.warning(
                self, "Save Failed",
                "Failed to save settings. Please check the values."
            )

    def _reset_defaults(self):
        """Reset to default settings."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._viewmodel.reset_to_defaults()

    def _on_saved(self):
        """Handle save complete."""
        pass

    def refresh(self):
        """Refresh settings view."""
        self._load_settings()
