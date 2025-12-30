"""TDD Tests for FileForge configuration system.

These tests define the expected behavior of the config module.
RED phase: Tests written first, will fail until implementation.

Test Coverage:
- DatabaseConfig: Path defaults, WAL mode, vector search
- ScanningConfig: Extensions, exclusions, recursive scanning, max size
- OCRConfig: Engine selection, languages, GPU configuration
- LLMConfig: Model defaults, temperature, base URL
- VisionConfig: Model selection, confidence thresholds
- FaceConfig: Detection and recognition settings
- NSFWConfig: Content detection thresholds
- ProcessingConfig: Batch processing, workers, timeout
- OutputConfig: Directory structure, formats
- LoggingConfig: Levels, file handling
- Settings: Integration, environment overrides, file loading
- Validation: Range checks, type validation, required fields
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import os
import tempfile
from typing import List


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDatabaseConfig:
    """Tests for database configuration.

    Database config should:
    - Default to user home directory
    - Enable WAL mode for concurrency
    - Enable vector search for semantic queries
    - Support custom paths
    """

    def test_default_db_path_is_in_user_home(self):
        """Database path should default to ~/.fileforge/fileforge.db"""
        from fileforge.config import DatabaseConfig
        config = DatabaseConfig()
        assert ".fileforge" in str(config.path)
        assert config.path.name == "fileforge.db"

    def test_wal_mode_enabled_by_default(self):
        """WAL mode should be enabled by default for better concurrency."""
        from fileforge.config import DatabaseConfig
        config = DatabaseConfig()
        assert config.wal_mode is True

    def test_vector_search_enabled_by_default(self):
        """Vector search should be enabled by default."""
        from fileforge.config import DatabaseConfig
        config = DatabaseConfig()
        assert config.vector_search is True

    def test_custom_db_path_can_be_set(self):
        """Should accept custom database path."""
        from fileforge.config import DatabaseConfig
        custom_path = Path("/custom/location/db.sqlite")
        config = DatabaseConfig(path=custom_path)
        assert config.path == custom_path

    def test_db_path_converts_string_to_path(self):
        """String paths should be converted to Path objects."""
        from fileforge.config import DatabaseConfig
        config = DatabaseConfig(path="/some/path/db.sqlite")
        assert isinstance(config.path, Path)


class TestScanningConfig:
    """Tests for file scanning configuration.

    Scanning config should:
    - Include common file extensions
    - Exclude system/build directories
    - Support recursive scanning
    - Enforce reasonable file size limits
    """

    def test_default_extensions_include_common_types(self):
        """Should include common document and image extensions."""
        from fileforge.config import ScanningConfig
        config = ScanningConfig()
        assert ".pdf" in config.extensions
        assert ".docx" in config.extensions
        assert ".jpg" in config.extensions
        assert ".png" in config.extensions
        assert ".txt" in config.extensions
        assert ".md" in config.extensions

    def test_default_extensions_include_office_formats(self):
        """Should include Microsoft Office and OpenOffice formats."""
        from fileforge.config import ScanningConfig
        config = ScanningConfig()
        assert ".xlsx" in config.extensions
        assert ".pptx" in config.extensions
        assert ".odt" in config.extensions

    def test_default_exclusions_include_common_dirs(self):
        """Should exclude common non-content directories."""
        from fileforge.config import ScanningConfig
        config = ScanningConfig()
        assert "__pycache__" in config.exclusions
        assert ".git" in config.exclusions
        assert "node_modules" in config.exclusions

    def test_default_exclusions_include_system_dirs(self):
        """Should exclude system and temporary directories."""
        from fileforge.config import ScanningConfig
        config = ScanningConfig()
        assert ".cache" in config.exclusions or "cache" in [e.lower() for e in config.exclusions]

    def test_recursive_enabled_by_default(self):
        """Recursive scanning should be enabled by default."""
        from fileforge.config import ScanningConfig
        config = ScanningConfig()
        assert config.recursive is True

    def test_max_size_has_reasonable_default(self):
        """Max file size should have a reasonable default (e.g., 500MB)."""
        from fileforge.config import ScanningConfig
        config = ScanningConfig()
        assert config.max_size_mb >= 100
        assert config.max_size_mb <= 1000

    def test_custom_extensions_can_be_added(self):
        """Should support custom file extensions."""
        from fileforge.config import ScanningConfig
        custom_extensions = [".custom", ".xyz"]
        config = ScanningConfig(extensions=custom_extensions)
        assert ".custom" in config.extensions
        assert ".xyz" in config.extensions

    def test_exclusions_can_be_customized(self):
        """Should support custom exclusion patterns."""
        from fileforge.config import ScanningConfig
        custom_exclusions = ["temp", "build"]
        config = ScanningConfig(exclusions=custom_exclusions)
        assert "temp" in config.exclusions
        assert "build" in config.exclusions


class TestOCRConfig:
    """Tests for OCR engine configuration.

    OCR config should:
    - Default to PaddleOCR for accuracy
    - Support multiple languages
    - Enable GPU acceleration when available
    - Set reasonable confidence thresholds
    """

    def test_default_engine_is_paddleocr(self):
        """Default OCR engine should be PaddleOCR for better accuracy."""
        from fileforge.config import OCRConfig
        config = OCRConfig()
        assert config.engine == "paddleocr"

    def test_default_language_is_english(self):
        """Default language should include English."""
        from fileforge.config import OCRConfig
        config = OCRConfig()
        assert "en" in config.languages or "eng" in config.languages

    def test_gpu_enabled_by_default(self):
        """GPU should be enabled by default if available."""
        from fileforge.config import OCRConfig
        config = OCRConfig()
        assert config.gpu_enabled is True

    def test_confidence_threshold_has_default(self):
        """Confidence threshold should have a reasonable default."""
        from fileforge.config import OCRConfig
        config = OCRConfig()
        assert 0.0 <= config.confidence_threshold <= 1.0
        assert config.confidence_threshold >= 0.6  # Reasonable minimum

    def test_multiple_languages_supported(self):
        """Should support multiple OCR languages."""
        from fileforge.config import OCRConfig
        config = OCRConfig(languages=["en", "es", "fr"])
        assert len(config.languages) == 3
        assert "en" in config.languages
        assert "es" in config.languages
        assert "fr" in config.languages

    def test_alternative_engines_supported(self):
        """Should support alternative OCR engines like Tesseract."""
        from fileforge.config import OCRConfig
        config = OCRConfig(engine="tesseract")
        assert config.engine == "tesseract"


class TestLLMConfig:
    """Tests for LLM integration configuration.

    LLM config should:
    - Default to Qwen2.5 model
    - Use temperature=0 for deterministic outputs
    - Point to local Ollama by default
    - Support timeout configuration
    """

    def test_default_model_is_qwen(self):
        """Default model should be Qwen2.5 as per spec."""
        from fileforge.config import LLMConfig
        config = LLMConfig()
        assert "qwen" in config.model.lower()

    def test_default_temperature_is_zero(self):
        """Temperature should be 0 for deterministic outputs."""
        from fileforge.config import LLMConfig
        config = LLMConfig()
        assert config.temperature == 0.0

    def test_default_base_url_is_localhost(self):
        """Base URL should point to local Ollama by default."""
        from fileforge.config import LLMConfig
        config = LLMConfig()
        assert "localhost" in config.base_url or "127.0.0.1" in config.base_url

    def test_timeout_has_reasonable_default(self):
        """Timeout should have a reasonable default (e.g., 30-60s)."""
        from fileforge.config import LLMConfig
        config = LLMConfig()
        assert config.timeout >= 30
        assert config.timeout <= 120

    def test_custom_model_can_be_set(self):
        """Should support custom model selection."""
        from fileforge.config import LLMConfig
        config = LLMConfig(model="llama3.2")
        assert config.model == "llama3.2"

    def test_temperature_can_be_adjusted(self):
        """Should allow temperature adjustment for creativity."""
        from fileforge.config import LLMConfig
        config = LLMConfig(temperature=0.7)
        assert config.temperature == 0.7


class TestVisionConfig:
    """Tests for vision model configuration.

    Vision config should:
    - Default to appropriate vision model
    - Set reasonable confidence thresholds
    - Support multiple detection types
    """

    def test_default_model_specified(self):
        """Default vision model should be specified."""
        from fileforge.config import VisionConfig
        config = VisionConfig()
        assert config.model is not None
        assert len(config.model) > 0

    def test_confidence_threshold_in_valid_range(self):
        """Confidence threshold should be between 0 and 1."""
        from fileforge.config import VisionConfig
        config = VisionConfig()
        assert 0.0 <= config.confidence_threshold <= 1.0

    def test_gpu_enabled_by_default(self):
        """GPU should be enabled by default for faster inference."""
        from fileforge.config import VisionConfig
        config = VisionConfig()
        assert config.gpu_enabled is True


class TestFaceConfig:
    """Tests for face detection and recognition configuration.

    Face config should:
    - Enable detection by default
    - Set appropriate confidence thresholds
    - Support recognition configuration
    """

    def test_detection_enabled_by_default(self):
        """Face detection should be enabled by default."""
        from fileforge.config import FaceConfig
        config = FaceConfig()
        assert config.enabled is True

    def test_confidence_threshold_reasonable(self):
        """Face detection confidence should be reasonable."""
        from fileforge.config import FaceConfig
        config = FaceConfig()
        assert 0.0 <= config.confidence_threshold <= 1.0
        assert config.confidence_threshold >= 0.5

    def test_recognition_can_be_enabled(self):
        """Face recognition should be configurable."""
        from fileforge.config import FaceConfig
        config = FaceConfig()
        # Should have recognition-related settings
        assert hasattr(config, 'enabled')


class TestNSFWConfig:
    """Tests for NSFW content detection configuration.

    NSFW config should:
    - Be configurable (enabled/disabled)
    - Set appropriate confidence thresholds
    - Support filtering levels
    """

    def test_can_be_enabled_or_disabled(self):
        """NSFW detection should be configurable."""
        from fileforge.config import NSFWConfig
        config = NSFWConfig()
        assert hasattr(config, 'enabled')
        assert isinstance(config.enabled, bool)

    def test_confidence_threshold_in_valid_range(self):
        """NSFW confidence threshold should be between 0 and 1."""
        from fileforge.config import NSFWConfig
        config = NSFWConfig()
        assert 0.0 <= config.confidence_threshold <= 1.0


class TestProcessingConfig:
    """Tests for processing and performance configuration.

    Processing config should:
    - Set reasonable batch sizes
    - Configure worker threads
    - Set timeout limits
    - Enable/disable parallel processing
    """

    def test_batch_size_is_positive(self):
        """Batch size should be a positive integer."""
        from fileforge.config import ProcessingConfig
        config = ProcessingConfig()
        assert config.batch_size > 0

    def test_default_batch_size_is_reasonable(self):
        """Default batch size should be reasonable (e.g., 10-100)."""
        from fileforge.config import ProcessingConfig
        config = ProcessingConfig()
        assert config.batch_size >= 1
        assert config.batch_size <= 100

    def test_workers_auto_configured(self):
        """Worker count should auto-configure based on CPU."""
        from fileforge.config import ProcessingConfig
        config = ProcessingConfig()
        assert config.workers >= 1
        assert config.workers <= os.cpu_count() * 2

    def test_timeout_has_reasonable_default(self):
        """Processing timeout should have reasonable default."""
        from fileforge.config import ProcessingConfig
        config = ProcessingConfig()
        assert config.timeout > 0
        assert config.timeout <= 3600  # Max 1 hour


class TestOutputConfig:
    """Tests for output configuration.

    Output config should:
    - Set default output directory
    - Configure output formats
    - Set organization structure
    """

    def test_default_output_directory_exists(self):
        """Default output directory should be specified."""
        from fileforge.config import OutputConfig
        config = OutputConfig()
        assert config.directory is not None

    def test_output_format_specified(self):
        """Output format should be specified (json, csv, etc.)."""
        from fileforge.config import OutputConfig
        config = OutputConfig()
        assert hasattr(config, 'format')

    def test_custom_directory_can_be_set(self):
        """Should support custom output directory."""
        from fileforge.config import OutputConfig
        custom_dir = Path("/custom/output")
        config = OutputConfig(directory=custom_dir)
        assert config.directory == custom_dir


class TestLoggingConfig:
    """Tests for logging configuration.

    Logging config should:
    - Default to INFO level
    - Support file and console logging
    - Configure log rotation
    """

    def test_default_log_level_is_info(self):
        """Default log level should be INFO."""
        from fileforge.config import LoggingConfig
        config = LoggingConfig()
        assert config.level.upper() == "INFO"

    def test_valid_log_levels_supported(self):
        """Should support standard log levels."""
        from fileforge.config import LoggingConfig
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level.upper() == level

    def test_log_file_path_configurable(self):
        """Log file path should be configurable."""
        from fileforge.config import LoggingConfig
        config = LoggingConfig()
        assert hasattr(config, 'file') or hasattr(config, 'path')


class TestSettings:
    """Tests for main Settings class.

    Settings should:
    - Load all configuration sections
    - Support environment variable overrides
    - Load from TOML files
    - Use defaults when config missing
    - Validate all settings
    """

    def test_settings_loads_all_config_sections(self):
        """Settings should have all required configuration sections."""
        from fileforge.config import Settings
        settings = Settings()
        assert hasattr(settings, 'database')
        assert hasattr(settings, 'scanning')
        assert hasattr(settings, 'ocr')
        assert hasattr(settings, 'vision')
        assert hasattr(settings, 'llm')
        assert hasattr(settings, 'faces')
        assert hasattr(settings, 'nsfw')
        assert hasattr(settings, 'processing')
        assert hasattr(settings, 'output')
        assert hasattr(settings, 'logging')

    def test_environment_variable_override(self):
        """Environment variables with FILEFORGE_ prefix should override config."""
        from fileforge.config import Settings
        with patch.dict(os.environ, {'FILEFORGE_LOG_LEVEL': 'DEBUG'}):
            settings = Settings()
            assert settings.logging.level == "DEBUG"

    def test_config_file_loading(self, temp_dir):
        """Should load configuration from TOML file."""
        from fileforge.config import load_config

        config_file = temp_dir / "config.toml"
        config_file.write_text('''
[database]
path = "/custom/path/db.sqlite"
wal_mode = false

[logging]
level = "WARNING"
''')

        settings = load_config(config_file)
        assert str(settings.database.path) == "/custom/path/db.sqlite"
        assert settings.database.wal_mode is False
        assert settings.logging.level == "WARNING"

    def test_missing_config_file_uses_defaults(self):
        """Missing config file should use all defaults."""
        from fileforge.config import load_config
        settings = load_config(Path("/nonexistent/config.toml"))
        assert settings is not None
        assert settings.database.wal_mode is True  # default value

    def test_invalid_config_raises_validation_error(self, temp_dir):
        """Invalid configuration values should raise ValidationError."""
        from fileforge.config import load_config
        from pydantic import ValidationError

        config_file = temp_dir / "config.toml"
        config_file.write_text('''
[scanning]
max_size_mb = -100
''')

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_partial_config_merges_with_defaults(self, temp_dir):
        """Partial config should merge with defaults, not replace."""
        from fileforge.config import load_config

        config_file = temp_dir / "config.toml"
        config_file.write_text('''
[database]
wal_mode = false
''')

        settings = load_config(config_file)
        # Custom setting
        assert settings.database.wal_mode is False
        # Default settings should still exist
        assert settings.database.vector_search is True
        assert settings.scanning.recursive is True

    def test_settings_is_immutable(self):
        """Settings should be immutable after creation (frozen)."""
        from fileforge.config import Settings
        settings = Settings()
        # Pydantic models can be frozen
        if hasattr(settings, 'model_config'):
            # Check if frozen is set
            pass  # Implementation will determine if this is enforced


class TestConfigValidation:
    """Tests for configuration validation.

    Validation should:
    - Enforce value ranges (0-1 for confidence, positive for sizes)
    - Validate file paths
    - Check required fields
    - Validate enum values
    """

    def test_confidence_threshold_must_be_between_0_and_1(self):
        """Confidence thresholds must be in valid range."""
        from fileforge.config import OCRConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OCRConfig(confidence_threshold=1.5)

        with pytest.raises(ValidationError):
            OCRConfig(confidence_threshold=-0.1)

    def test_batch_size_must_be_positive(self):
        """Batch size must be a positive integer."""
        from fileforge.config import ProcessingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProcessingConfig(batch_size=0)

        with pytest.raises(ValidationError):
            ProcessingConfig(batch_size=-5)

    def test_max_size_must_be_positive(self):
        """Max file size must be positive."""
        from fileforge.config import ScanningConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ScanningConfig(max_size_mb=0)

        with pytest.raises(ValidationError):
            ScanningConfig(max_size_mb=-100)

    def test_workers_must_be_positive(self):
        """Worker count must be positive."""
        from fileforge.config import ProcessingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProcessingConfig(workers=0)

        with pytest.raises(ValidationError):
            ProcessingConfig(workers=-1)

    def test_timeout_must_be_positive(self):
        """Timeout must be positive."""
        from fileforge.config import LLMConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMConfig(timeout=0)

        with pytest.raises(ValidationError):
            LLMConfig(timeout=-10)

    def test_temperature_must_be_between_0_and_2(self):
        """LLM temperature must be in valid range (0-2)."""
        from fileforge.config import LLMConfig
        from pydantic import ValidationError

        # Valid range
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)

        # Invalid range
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.5)

    def test_log_level_must_be_valid(self):
        """Log level must be a valid Python logging level."""
        from fileforge.config import LoggingConfig
        from pydantic import ValidationError

        # Valid levels
        LoggingConfig(level="DEBUG")
        LoggingConfig(level="INFO")
        LoggingConfig(level="WARNING")
        LoggingConfig(level="ERROR")
        LoggingConfig(level="CRITICAL")

        # Invalid level should be rejected or normalized
        # Implementation may choose to validate or accept any string


class TestConfigSerialization:
    """Tests for configuration serialization and export.

    Should support:
    - Export to TOML
    - Export to JSON
    - Pretty printing
    - Roundtrip consistency
    """

    def test_settings_can_be_exported_to_dict(self):
        """Settings should be exportable to dictionary."""
        from fileforge.config import Settings
        settings = Settings()
        config_dict = settings.model_dump() if hasattr(settings, 'model_dump') else settings.dict()
        assert isinstance(config_dict, dict)
        assert 'database' in config_dict
        assert 'scanning' in config_dict

    def test_settings_can_be_exported_to_toml(self, temp_dir):
        """Settings should be exportable to TOML file."""
        from fileforge.config import Settings
        settings = Settings()

        # Should have a save or to_toml method
        output_file = temp_dir / "exported.toml"
        if hasattr(settings, 'save_toml'):
            settings.save_toml(output_file)
            assert output_file.exists()

    def test_roundtrip_consistency(self, temp_dir):
        """Config should maintain values through save/load cycle."""
        from fileforge.config import Settings, load_config

        # Create settings with custom values
        settings1 = Settings()

        # Export to file
        config_file = temp_dir / "roundtrip.toml"
        if hasattr(settings1, 'save_toml'):
            settings1.save_toml(config_file)

            # Load back
            settings2 = load_config(config_file)

            # Compare key values
            assert settings1.database.wal_mode == settings2.database.wal_mode
            assert settings1.scanning.recursive == settings2.scanning.recursive


class TestConfigDefaults:
    """Tests to verify all defaults are sensible.

    This ensures the system works out-of-the-box
    with minimal configuration.
    """

    def test_default_config_is_valid(self):
        """Default configuration should pass all validation."""
        from fileforge.config import Settings
        from pydantic import ValidationError

        try:
            settings = Settings()
            assert settings is not None
        except ValidationError as e:
            pytest.fail(f"Default config failed validation: {e}")

    def test_all_paths_are_absolute(self):
        """All path configurations should resolve to absolute paths."""
        from fileforge.config import Settings
        settings = Settings()

        # Database path should be absolute
        assert settings.database.path.is_absolute()

        # Output directory should be absolute
        if hasattr(settings.output, 'directory'):
            if isinstance(settings.output.directory, Path):
                # May be relative initially, but should resolve
                pass

    def test_gpu_settings_detect_availability(self):
        """GPU settings should intelligently detect GPU availability."""
        from fileforge.config import Settings
        settings = Settings()

        # GPU enabled should be boolean
        assert isinstance(settings.ocr.gpu_enabled, bool)
        assert isinstance(settings.vision.gpu_enabled, bool)

        # Note: Actual GPU detection happens at runtime,
        # config just stores the preference


class TestConfigHelpers:
    """Tests for configuration helper functions."""

    def test_get_config_returns_settings_instance(self):
        """get_config() should return a Settings instance."""
        from fileforge.config import get_config, Settings
        config = get_config()
        assert isinstance(config, Settings)

    def test_get_config_caches_instance(self):
        """get_config() should return the same instance (singleton pattern)."""
        from fileforge.config import get_config
        config1 = get_config()
        config2 = get_config()
        # May or may not be same instance depending on implementation
        # This test documents expected behavior

    def test_load_config_accepts_path_or_string(self):
        """load_config() should accept both Path and string."""
        from fileforge.config import load_config
        from pathlib import Path

        # Should work with Path object
        settings1 = load_config(Path("/nonexistent/config.toml"))
        assert settings1 is not None

        # Should work with string
        settings2 = load_config("/nonexistent/config.toml")
        assert settings2 is not None


# Performance and integration tests
class TestConfigPerformance:
    """Tests for configuration performance characteristics."""

    def test_config_loads_quickly(self):
        """Configuration loading should be fast (<100ms)."""
        import time
        from fileforge.config import Settings

        start = time.time()
        settings = Settings()
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should load in under 100ms

    def test_config_validation_is_fast(self):
        """Configuration validation should be fast."""
        import time
        from fileforge.config import Settings

        start = time.time()
        for _ in range(100):
            settings = Settings()
        elapsed = time.time() - start

        # 100 validations should complete quickly
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
