"""Config Service - wraps configuration management."""
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
from PySide6.QtCore import QObject, Signal

from ...config import Settings, load_config


class ConfigService(QObject):
    """Service for configuration management.

    Wraps Settings with validation and
    Qt signals for reactive updates.
    """

    config_changed = Signal(str, object)  # key, value
    config_saved = Signal()
    validation_error = Signal(str, list)  # key, errors

    def __init__(self, config: Optional[Settings] = None):
        super().__init__()
        self._config = config or load_config()
        self._modified: Dict[str, Any] = {}

    @property
    def config(self) -> Settings:
        """Get configuration."""
        return self._config

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as dictionary.

        Returns:
            Dictionary of all settings
        """
        return {
            'database': {
                'path': self._config.database.path,
                'wal_mode': self._config.database.wal_mode,
            },
            'scanning': {
                'recursive': self._config.scanning.recursive,
                'max_size_mb': self._config.scanning.max_size_mb,
            },
            'processing': {
                'workers': self._config.processing.workers,
                'batch_size': self._config.processing.batch_size,
                'timeout': self._config.processing.timeout,
            },
            'ocr': {
                'engine': self._config.ocr.engine,
                'gpu_enabled': self._config.ocr.gpu_enabled,
            },
            'llm': {
                'model': self._config.llm.model,
                'base_url': self._config.llm.base_url,
                'temperature': self._config.llm.temperature,
            },
        }

    def get(self, key: str) -> Any:
        """Get setting value by dot-notation key.

        Args:
            key: Setting key (e.g., 'scanning.recursive')

        Returns:
            Setting value
        """
        # Check modified first
        if key in self._modified:
            return self._modified[key]

        # Parse nested key
        parts = key.split('.')
        obj = self._config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    def update(self, key: str, value: Any):
        """Update setting value.

        Args:
            key: Setting key
            value: New value
        """
        self._modified[key] = value
        self.config_changed.emit(key, value)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all settings.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for key, value in self._modified.items():
            is_valid, key_errors = self._validate_setting(key, value)
            if not is_valid:
                errors.extend(key_errors)

        return len(errors) == 0, errors

    def _validate_setting(self, key: str, value: Any) -> Tuple[bool, List[str]]:
        """Validate a single setting.

        Args:
            key: Setting key
            value: Setting value

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validation rules by key
        if key == 'processing.workers':
            if not isinstance(value, int) or value < 1 or value > 32:
                errors.append(f"{key} must be an integer between 1 and 32")

        elif key == 'processing.batch_size':
            if not isinstance(value, int) or value < 1 or value > 100:
                errors.append(f"{key} must be an integer between 1 and 100")

        elif key == 'processing.timeout':
            if not isinstance(value, int) or value < 30 or value > 3600:
                errors.append(f"{key} must be an integer between 30 and 3600")

        elif key == 'scanning.max_size_mb':
            if not isinstance(value, int) or value < 1 or value > 2000:
                errors.append(f"{key} must be an integer between 1 and 2000")

        elif key == 'llm.temperature':
            if not isinstance(value, (int, float)) or value < 0 or value > 2:
                errors.append(f"{key} must be a number between 0 and 2")

        elif key == 'ocr.engine':
            if value not in ['paddleocr', 'tesseract']:
                errors.append(f"{key} must be 'paddleocr' or 'tesseract'")

        if errors:
            self.validation_error.emit(key, errors)

        return len(errors) == 0, errors

    def save(self) -> bool:
        """Save configuration to file.

        Returns:
            True if successful
        """
        is_valid, errors = self.validate()
        if not is_valid:
            return False

        try:
            # Apply modified settings to config
            for key, value in self._modified.items():
                parts = key.split('.')
                if len(parts) == 2:
                    section, setting = parts
                    if hasattr(self._config, section):
                        section_obj = getattr(self._config, section)
                        if hasattr(section_obj, setting):
                            setattr(section_obj, setting, value)

            # Save to file
            config_path = Path.home() / '.fileforge' / 'config.toml'
            config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config.save_toml(config_path)

            self._modified = {}
            self.config_saved.emit()
            return True

        except Exception:
            return False

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self._config = Settings()
        self._modified = {}

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return len(self._modified) > 0
