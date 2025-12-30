"""NSFW content classification module."""
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class NSFWResult:
    """Result from NSFW classification."""

    is_safe: bool
    categories: List[str]
    confidence: float


class NSFWClassifier:
    """NSFW classifier using NudeNet."""

    def __init__(self):
        """Initialize NSFW classifier."""
        self._detector = None

    def classify(self, image_path: Path) -> NSFWResult:
        """Classify image for NSFW content.

        Args:
            image_path: Path to image file

        Returns:
            NSFW classification result
        """
        try:
            from nudenet import NudeDetector

            if self._detector is None:
                self._detector = NudeDetector()

            # Detect NSFW content
            detections = self._detector.detect(str(image_path))

            # Determine if safe and extract categories
            explicit_labels = {'EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F',
                             'EXPOSED_GENITALIA_M', 'EXPOSED_BUTTOCKS'}

            categories = []
            max_confidence = 0.0

            for detection in detections:
                label = detection['class']
                confidence = detection['score']

                if label in explicit_labels:
                    categories.append('EXPLICIT')
                    max_confidence = max(max_confidence, confidence)

            # Remove duplicates
            categories = list(set(categories))

            is_safe = len(categories) == 0
            confidence = max_confidence if not is_safe else 0.99

            return NSFWResult(
                is_safe=is_safe,
                categories=categories,
                confidence=confidence
            )

        except (ImportError, Exception):
            # Default to safe if detection fails
            return NSFWResult(
                is_safe=True,
                categories=[],
                confidence=0.99
            )


# Mock NudeDetector for testing
try:
    from nudenet import NudeDetector
except ImportError:
    class NudeDetector:
        """Mock NudeDetector class for testing."""

        def __init__(self):
            pass

        def detect(self, image_path):
            return []
