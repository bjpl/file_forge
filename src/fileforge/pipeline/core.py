"""Core pipeline classes for FileForge."""
from pathlib import Path
from typing import List, Set


class Pipeline:
    """Main processing pipeline."""

    def __init__(self):
        self._processors = []
        self._supported_extensions: Set[str] = set()

    def register_processor(self, processor):
        """Register a processor with the pipeline.

        Args:
            processor: Processor instance to register
        """
        self._processors.append(processor)
        if hasattr(processor, 'supported_extensions'):
            self._supported_extensions.update(processor.supported_extensions)

    @property
    def supported_extensions(self) -> Set[str]:
        """Get all supported file extensions."""
        return self._supported_extensions
