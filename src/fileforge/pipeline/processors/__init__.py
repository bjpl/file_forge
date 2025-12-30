"""
Processors module for FileForge pipeline.

Contains all stage processors including file detection, metadata extraction,
OCR, vision analysis, LLM integration, and storage.
"""

from typing import Dict, Optional, Type

from .document import DocumentProcessor, ExtractedContent, ProcessingError
from .text import TextProcessor
from .image import ImageProcessor

# Aliases for backward compatibility and common usage
PDFProcessor = DocumentProcessor  # PDF is handled by DocumentProcessor

# Processor registry
_PROCESSOR_REGISTRY: Dict[str, Type] = {}


def register_processor(extension: str, processor_class: Type) -> None:
    """Register a processor for a file extension.

    Args:
        extension: File extension (e.g., '.pdf')
        processor_class: Processor class to handle this extension
    """
    _PROCESSOR_REGISTRY[extension.lower()] = processor_class


def get_processor(extension: str) -> Optional[object]:
    """Get processor instance for a file extension.

    Args:
        extension: File extension (e.g., '.pdf')

    Returns:
        Processor instance or None if not found
    """
    processor_class = _PROCESSOR_REGISTRY.get(extension.lower())
    if processor_class:
        return processor_class()
    return None


# Register built-in processors
def _register_builtin_processors() -> None:
    """Register all built-in processors."""
    # Register DocumentProcessor for supported extensions
    for ext in DocumentProcessor.supported_extensions:
        register_processor(ext, DocumentProcessor)

    # Register TextProcessor for supported extensions
    for ext in TextProcessor.supported_extensions:
        register_processor(ext, TextProcessor)

    # Register ImageProcessor for supported extensions
    for ext in ImageProcessor.supported_extensions:
        register_processor(ext, ImageProcessor)


# Auto-register on import
_register_builtin_processors()


__all__ = [
    "DocumentProcessor",
    "TextProcessor",
    "ImageProcessor",
    "PDFProcessor",  # Alias for DocumentProcessor
    "ExtractedContent",
    "ProcessingError",
    "register_processor",
    "get_processor",
]
