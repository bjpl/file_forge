"""
Pipeline module for FileForge.

Provides the core pipeline orchestration, stage management, and processor coordination.
"""

from .orchestrator import (
    PipelineOrchestrator,
    Stage,
    PipelineContext,
    PipelineResult,
    FileResult,
    ExtractedContent,
    IntelligenceResult
)
from .discovery import (
    FileDiscovery,
    DiscoveredFile,
    FileType,
    ProcessingQueue
)

__all__ = [
    "PipelineOrchestrator",
    "Stage",
    "PipelineContext",
    "PipelineResult",
    "FileResult",
    "ExtractedContent",
    "IntelligenceResult",
    "FileDiscovery",
    "DiscoveredFile",
    "FileType",
    "ProcessingQueue",
]
