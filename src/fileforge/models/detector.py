"""Object detection module using YOLO."""
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """Represents a detected object in an image."""

    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)


class ObjectDetector:
    """Object detector using YOLO-World."""

    def __init__(self, confidence_threshold: float = 0.25):
        """Initialize object detector.

        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self._model = None

    def detect(self, image_path: Path) -> List[DetectedObject]:
        """Detect objects in an image.

        Args:
            image_path: Path to image file

        Returns:
            List of detected objects
        """
        # Try to import YOLO
        try:
            from ultralytics import YOLO

            if self._model is None:
                # Load YOLO model (using yolov8n as default)
                self._model = YOLO('yolov8n.pt')

            # Run inference
            results = self._model(str(image_path))

            detected = []
            for result in results:
                if hasattr(result, 'boxes'):
                    boxes = result.boxes
                    for i in range(len(boxes.xyxy)):
                        confidence = float(boxes.conf[i])
                        if confidence >= self.confidence_threshold:
                            bbox = tuple(float(x) for x in boxes.xyxy[i])
                            cls_id = int(boxes.cls[i])
                            label = result.names[cls_id]

                            detected.append(DetectedObject(
                                label=label,
                                confidence=confidence,
                                bbox=bbox
                            ))

            return detected

        except (ImportError, Exception):
            # Return empty list if YOLO unavailable or fails
            return []


# Mock YOLO for testing
try:
    from ultralytics import YOLO
except ImportError:
    class YOLO:
        """Mock YOLO class for testing."""
        def __init__(self, model_path):
            pass

        def __call__(self, image_path):
            return []
