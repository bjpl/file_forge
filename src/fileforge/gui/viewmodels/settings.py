"""Settings ViewModel."""
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PySide6.QtCore import QObject, Signal

from ..state import AppState
from ...config import Settings, load_config


class SettingsViewModel(QObject):
    """ViewModel for Settings view.

    Manages:
    - Configuration sections
    - Setting updates with validation
    - Save/reset functionality
    - Theme preferences
    """

    setting_changed = Signal(str, object)  # key, value
    settings_saved = Signal()
    settings_reset = Signal()

    def __init__(self, config: Optional[Settings] = None):
        super().__init__()
        self._config = config or load_config()
        self._modified_settings: Dict[str, Any] = {}

    @property
    def config(self) -> Settings:
        """Get configuration."""
        return self._config

    @property
    def sections(self) -> List[Dict[str, Any]]:
        """Get settings sections."""
        return [
            {
                'name': 'Database',
                'icon': 'database',
                'settings': [
                    {'key': 'database.path', 'label': 'Database Path', 'type': 'path'},
                    {'key': 'database.wal_mode', 'label': 'WAL Mode', 'type': 'bool'},
                ]
            },
            {
                'name': 'Scanning',
                'icon': 'folder-search',
                'settings': [
                    {'key': 'scanning.recursive', 'label': 'Recursive Scan', 'type': 'bool'},
                    {'key': 'scanning.max_size_mb', 'label': 'Max File Size (MB)', 'type': 'int', 'min': 1, 'max': 2000},
                ]
            },
            {
                'name': 'Processing',
                'icon': 'cpu',
                'settings': [
                    {'key': 'processing.workers', 'label': 'Worker Threads', 'type': 'int', 'min': 1, 'max': 32},
                    {'key': 'processing.batch_size', 'label': 'Batch Size', 'type': 'int', 'min': 1, 'max': 100},
                    {'key': 'processing.timeout', 'label': 'Timeout (seconds)', 'type': 'int', 'min': 30, 'max': 3600},
                ]
            },
            {
                'name': 'OCR',
                'icon': 'text',
                'settings': [
                    {'key': 'ocr.engine', 'label': 'OCR Engine', 'type': 'choice', 'choices': ['paddleocr', 'tesseract']},
                    {'key': 'ocr.gpu_enabled', 'label': 'GPU Acceleration', 'type': 'bool'},
                ]
            },
            {
                'name': 'LLM',
                'icon': 'brain',
                'settings': [
                    {'key': 'llm.model', 'label': 'Model', 'type': 'str'},
                    {'key': 'llm.base_url', 'label': 'Ollama URL', 'type': 'str'},
                    {'key': 'llm.temperature', 'label': 'Temperature', 'type': 'float', 'min': 0.0, 'max': 2.0},
                ]
            },
            {
                'name': 'Appearance',
                'icon': 'palette',
                'settings': [
                    {'key': 'appearance.theme', 'label': 'Theme', 'type': 'choice', 'choices': ['system', 'light', 'dark']},
                ]
            },
        ]

    def get_setting(self, key: str) -> Any:
        """Get setting value by dot-notation key."""
        # Check modified first
        if key in self._modified_settings:
            return self._modified_settings[key]

        # Handle appearance.theme specially
        if key == 'appearance.theme':
            return AppState.instance().theme

        # Parse nested key
        parts = key.split('.')
        obj = self._config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    def update_setting(self, key: str, value: Any):
        """Update setting value."""
        # Validate first
        is_valid, _ = self.validate_setting(key, value)
        if not is_valid:
            return

        self._modified_settings[key] = value

        # Handle theme specially
        if key == 'appearance.theme':
            AppState.instance().set_theme(value)

        self.setting_changed.emit(key, value)

    def validate_setting(self, key: str, value: Any) -> Tuple[bool, List[str]]:
        """Validate a setting value."""
        errors = []

        # Find setting definition
        for section in self.sections:
            for setting in section['settings']:
                if setting['key'] == key:
                    setting_type = setting.get('type')

                    if setting_type == 'int':
                        if not isinstance(value, int):
                            errors.append(f"{key} must be an integer")
                        elif 'min' in setting and value < setting['min']:
                            errors.append(f"{key} must be >= {setting['min']}")
                        elif 'max' in setting and value > setting['max']:
                            errors.append(f"{key} must be <= {setting['max']}")

                    elif setting_type == 'float':
                        if not isinstance(value, (int, float)):
                            errors.append(f"{key} must be a number")
                        elif 'min' in setting and value < setting['min']:
                            errors.append(f"{key} must be >= {setting['min']}")
                        elif 'max' in setting and value > setting['max']:
                            errors.append(f"{key} must be <= {setting['max']}")

                    elif setting_type == 'choice':
                        if value not in setting.get('choices', []):
                            errors.append(f"{key} must be one of {setting['choices']}")

                    break

        return len(errors) == 0, errors

    def save(self) -> bool:
        """Save settings to config file."""
        try:
            # Apply modified settings to config
            for key, value in self._modified_settings.items():
                parts = key.split('.')
                if len(parts) == 2:
                    section, setting = parts
                    if hasattr(self._config, section):
                        section_obj = getattr(self._config, section)
                        if hasattr(section_obj, setting):
                            setattr(section_obj, setting, value)

            # Save to file
            config_path = Path.home() / '.fileforge' / 'config.toml'
            self._config.save_toml(config_path)

            self._modified_settings = {}
            self.settings_saved.emit()
            return True
        except Exception:
            return False

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self._config = Settings()
        self._modified_settings = {}
        AppState.instance().set_theme('system')
        self.settings_reset.emit()

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return len(self._modified_settings) > 0
