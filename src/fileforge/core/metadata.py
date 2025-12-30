"""Metadata models for FileForge."""
from typing import Any, Dict


class FileMetadata:
    """Metadata for processed files."""

    def __init__(self, file_type: str = None, **kwargs):
        self.file_type = file_type
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_analysis(cls, analysis: Any) -> "FileMetadata":
        """Create metadata from analysis result.

        Args:
            analysis: Analysis result object

        Returns:
            FileMetadata instance
        """
        metadata = cls()
        if hasattr(analysis, 'metadata'):
            for key, value in analysis.metadata.items():
                setattr(metadata, key, value)
        return metadata
