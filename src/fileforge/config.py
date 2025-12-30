"""Configuration management for FileForge using Pydantic Settings with TOML support.

This module provides a comprehensive configuration system that supports:
- TOML configuration files
- Environment variable overrides (FILEFORGE_ prefix)
- Command-line config file override (--config flag)
- Default configuration generation
- Validation via Pydantic models
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


# Enums for constrained configuration values
class OCREngine(str, Enum):
    """Supported OCR engines."""
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"


class VisionModel(str, Enum):
    """Supported vision models."""
    YOLO_WORLD = "yolo-world"


class FaceDetector(str, Enum):
    """Supported face detection models."""
    RETINAFACE = "retinaface"
    MTCNN = "mtcnn"


class FaceEmbeddingModel(str, Enum):
    """Supported face embedding models."""
    FACENET512 = "Facenet512"
    FACENET = "Facenet"
    ARCFACE = "ArcFace"


class ReportFormat(str, Enum):
    """Supported report output formats."""
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "markdown"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Configuration section models
class DatabaseConfig(BaseModel):
    """Database configuration."""
    model_config = ConfigDict(extra="forbid")

    path: Path = Field(
        default_factory=lambda: Path.home() / ".fileforge" / "fileforge.db",
        description="SQLite database file path"
    )
    wal_mode: bool = Field(
        default=True,
        description="Enable Write-Ahead Logging for better concurrency"
    )
    vector_search: bool = Field(
        default=True,
        description="Enable vector similarity search capabilities"
    )

    @field_validator("path", mode='before')
    @classmethod
    def convert_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class ScanningConfig(BaseModel):
    """File scanning configuration."""
    model_config = ConfigDict(extra="forbid")

    extensions: List[str] = Field(
        default=[
            ".pdf", ".docx", ".doc", ".xlsx", ".pptx", ".odt",
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
            ".txt", ".md", ".markdown"
        ],
        description="File extensions to scan"
    )
    exclusions: List[str] = Field(
        default=[
            "__pycache__", ".git", "node_modules", ".cache",
            "venv", ".venv", "__MACOSX", ".DS_Store"
        ],
        description="Directories and files to exclude from scanning"
    )
    recursive: bool = Field(
        default=True,
        description="Recursively scan subdirectories"
    )
    max_size_mb: int = Field(
        default=500,
        gt=0,
        le=2000,
        description="Maximum file size to process in megabytes"
    )


class OCRConfig(BaseModel):
    """OCR processing configuration."""
    model_config = ConfigDict(extra="forbid")

    engine: str = Field(
        default="paddleocr",
        description="OCR engine to use"
    )
    languages: List[str] = Field(
        default=["en"],
        description="Languages for OCR recognition (ISO 639-1 codes)"
    )
    gpu_enabled: bool = Field(
        default=True,
        description="Use GPU acceleration if available"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for OCR results"
    )


class VisionConfig(BaseModel):
    """Vision/object detection configuration."""
    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="llava:7b",
        description="Vision model for object detection"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for object detection"
    )
    gpu_enabled: bool = Field(
        default=True,
        description="Use GPU acceleration if available"
    )


class LLMConfig(BaseModel):
    """LLM processing configuration."""
    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="qwen2.5:14b",
        description="LLM model identifier"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM generation"
    )
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    timeout: int = Field(
        default=60,
        gt=0,
        le=300,
        description="Request timeout in seconds"
    )


class FaceConfig(BaseModel):
    """Face detection and recognition configuration."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable face detection"
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for face detection"
    )
    recognition_enabled: bool = Field(
        default=False,
        description="Enable face recognition (requires additional setup)"
    )


class NSFWConfig(BaseModel):
    """NSFW content detection configuration."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable NSFW content detection"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to flag as NSFW"
    )


class ProcessingConfig(BaseModel):
    """Processing pipeline configuration."""
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Batch size for processing operations"
    )
    workers: int = Field(
        default_factory=lambda: max(1, os.cpu_count() or 1),
        gt=0,
        description="Number of worker threads/processes"
    )
    timeout: int = Field(
        default=300,
        gt=0,
        le=3600,
        description="Processing timeout in seconds"
    )

    @field_validator('workers', mode='before')
    @classmethod
    def validate_workers(cls, v):
        """Ensure workers is within reasonable bounds."""
        if v is None:
            v = max(1, os.cpu_count() or 1)
        cpu_count = os.cpu_count() or 1
        return min(v, cpu_count * 2)


class OutputConfig(BaseModel):
    """Output and export configuration."""
    model_config = ConfigDict(extra="forbid")

    directory: Path = Field(
        default_factory=lambda: Path.home() / ".fileforge" / "output",
        description="Directory for output files"
    )
    format: str = Field(
        default="json",
        description="Output format (json, csv, etc.)"
    )

    @field_validator('directory', mode='before')
    @classmethod
    def convert_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class OrganizationRule(BaseModel):
    """Organization rule definition."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Rule name")
    pattern: str = Field(description="File matching pattern")
    destination: str = Field(description="Destination path template")
    conditions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional conditions (metadata-based)"
    )


class OrganizationConfig(BaseModel):
    """File organization configuration."""
    model_config = ConfigDict(extra="forbid")

    rules: List[OrganizationRule] = Field(
        default_factory=list,
        description="Organization rules"
    )
    naming_templates: Dict[str, str] = Field(
        default={
            "photo": "{date:%Y-%m-%d}_{original_name}",
            "document": "{category}/{date:%Y}/{original_name}",
            "video": "Videos/{date:%Y-%m}/{original_name}"
        },
        description="Naming templates by file type"
    )
    default_category: str = Field(
        default="Uncategorized",
        description="Default category for unmatched files"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    model_config = ConfigDict(extra="forbid")

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    file: Optional[Path] = Field(
        default=None,
        description="Log file path (None for console only)"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper in valid_levels:
            return v_upper
        return "INFO"

    @field_validator('file', mode='before')
    @classmethod
    def convert_path(cls, v):
        """Convert string paths to Path objects."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return Path(v)
        return v


# Main Settings class
class Settings(BaseSettings):
    """Main FileForge settings combining all configuration sections.

    Supports:
    - Loading from TOML file (~/.fileforge/config.toml by default)
    - Environment variable overrides (FILEFORGE_ prefix)
    - Command-line config file override
    """
    model_config = SettingsConfigDict(
        env_prefix="FILEFORGE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid"
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scanning: ScanningConfig = Field(default_factory=ScanningConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    faces: FaceConfig = Field(default_factory=FaceConfig)
    nsfw: NSFWConfig = Field(default_factory=NSFWConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    organization: OrganizationConfig = Field(default_factory=OrganizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def __init__(self, **data):
        """Initialize settings with support for flat environment variables."""
        import os

        # Handle flat env variables
        if 'FILEFORGE_LOG_LEVEL' in os.environ:
            if 'logging' not in data:
                data['logging'] = {}
            if isinstance(data.get('logging'), dict):
                data['logging']['level'] = os.environ['FILEFORGE_LOG_LEVEL']

        super().__init__(**data)

    def save_toml(self, path) -> None:
        """Save settings to a TOML file.

        Args:
            path: Path where TOML file should be saved (str or Path)
        """
        try:
            import tomli_w
        except ImportError:
            raise ImportError(
                "tomli_w is required for saving TOML files. "
                "Install with: pip install tomli-w"
            )

        path = Path(path) if isinstance(path, str) else path
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path objects
        config_dict = self.model_dump()
        config_dict = self._convert_paths_to_strings(config_dict)

        with open(path, "wb") as f:
            tomli_w.dump(config_dict, f)

    def _convert_paths_to_strings(self, obj):
        """Recursively convert Path objects to strings and remove None values for serialization."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            # Filter out None values as TOML doesn't support null/None
            return {
                k: self._convert_paths_to_strings(v)
                for k, v in obj.items()
                if v is not None
            }
        elif isinstance(obj, list):
            return [self._convert_paths_to_strings(item) for item in obj]
        return obj


# Configuration loading functions
DEFAULT_CONFIG_PATH = Path.home() / ".fileforge" / "config.toml"


def load_config(config_path=None) -> Settings:
    """Load configuration from TOML file with environment variable overrides.

    Args:
        config_path: Path to config file (str or Path). If file doesn't exist,
                     returns default settings.

    Returns:
        Settings: Validated configuration settings
    """
    # Convert to Path if string
    if config_path is not None:
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        config_path = config_path.expanduser().resolve()

    # If file doesn't exist, return defaults
    if config_path is None or not config_path.exists():
        return Settings()

    # Load TOML file
    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
    except Exception:
        # If loading fails, return defaults
        return Settings()

    # Create Settings instance (will apply env var overrides automatically)
    try:
        settings = Settings(**config_data)
    except Exception:
        # If validation fails, raise the error
        raise

    return settings


def create_default_config(config_path: Path) -> None:
    """Create default configuration file.

    Args:
        config_path: Path where to create the config file
    """
    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate default config from Settings defaults
    default_settings = Settings()

    # Convert to dict and write as TOML
    config_dict = default_settings.model_dump(mode="json")

    # Convert to TOML format
    toml_content = _dict_to_toml(config_dict)

    # Write to file
    with open(config_path, "w") as f:
        f.write("# FileForge Configuration File\n")
        f.write("# Generated automatically - edit as needed\n\n")
        f.write(toml_content)

    print(f"Created default configuration at: {config_path}")


def _dict_to_toml(data: Dict[str, Any], indent: int = 0) -> str:
    """Convert nested dictionary to TOML format.

    Args:
        data: Dictionary to convert
        indent: Current indentation level

    Returns:
        str: TOML-formatted string
    """
    lines = []
    indent_str = "  " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            # Section header
            if indent == 0:
                lines.append(f"\n[{key}]")
                lines.append(_dict_to_toml(value, indent + 1))
            else:
                lines.append(f"\n{indent_str}[{key}]")
                lines.append(_dict_to_toml(value, indent + 1))
        elif isinstance(value, list):
            # Array
            if value and isinstance(value[0], dict):
                # Array of tables
                for item in value:
                    lines.append(f"\n{indent_str}[[{key}]]")
                    lines.append(_dict_to_toml(item, indent + 1))
            else:
                # Simple array
                array_str = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
                lines.append(f"{indent_str}{key} = [{array_str}]")
        elif isinstance(value, str):
            lines.append(f'{indent_str}{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{indent_str}{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{indent_str}{key} = {value}")
        elif value is None:
            lines.append(f"{indent_str}# {key} = null")
        else:
            lines.append(f'{indent_str}{key} = "{value}"')

    return "\n".join(lines)


def get_config_path_from_args(args: Optional[List[str]] = None) -> Optional[Path]:
    """Extract config file path from command-line arguments.

    Args:
        args: Command-line arguments (default: sys.argv)

    Returns:
        Optional[Path]: Config file path if --config flag present, else None
    """
    import sys
    if args is None:
        args = sys.argv[1:]

    try:
        config_idx = args.index("--config")
        if config_idx + 1 < len(args):
            return Path(args[config_idx + 1])
    except (ValueError, IndexError):
        pass

    return None


# Global singleton instance
_config_instance: Optional[Settings] = None


def get_config() -> Settings:
    """Get the global configuration instance.

    Uses singleton pattern to ensure consistent configuration
    across the application.

    Returns:
        Settings instance

    Example:
        >>> config = get_config()
        >>> print(config.logging.level)
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Settings()
    return _config_instance


def reset_config() -> None:
    """Reset the global configuration instance.

    Useful for testing or when configuration needs to be reloaded.
    """
    global _config_instance
    _config_instance = None


# Convenience function to load config with CLI override support
def load_config_with_cli_override(args: Optional[List[str]] = None) -> Settings:
    """Load configuration with support for --config CLI flag override.

    Args:
        args: Command-line arguments (default: sys.argv)

    Returns:
        Settings: Validated configuration settings
    """
    config_path = get_config_path_from_args(args)
    return load_config(config_path)
