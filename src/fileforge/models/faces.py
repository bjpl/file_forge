"""Face detection and embedding module."""
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DetectedFace:
    """Represents a detected face with embedding."""

    embedding: np.ndarray  # 512-dimensional face embedding
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)


class FaceDetector:
    """Face detector using DeepFace."""

    def __init__(self):
        """Initialize face detector."""
        self._detector = None

    def detect(self, image_path: Path) -> List[DetectedFace]:
        """Detect faces in an image.

        Args:
            image_path: Path to image file

        Returns:
            List of detected faces
        """
        try:
            from deepface import DeepFace

            # Extract faces with embeddings
            faces = DeepFace.extract_faces(
                str(image_path),
                detector_backend='opencv',
                enforce_detection=False
            )

            detected = []
            for face_obj in faces:
                # Get embedding using Facenet512
                try:
                    embedding_result = DeepFace.represent(
                        face_obj['face'],
                        model_name='Facenet512',
                        enforce_detection=False
                    )

                    if embedding_result:
                        embedding = np.array(embedding_result[0]['embedding'], dtype=np.float32)

                        # Get bounding box
                        facial_area = face_obj.get('facial_area', {})
                        bbox = (
                            facial_area.get('x', 0),
                            facial_area.get('y', 0),
                            facial_area.get('x', 0) + facial_area.get('w', 0),
                            facial_area.get('y', 0) + facial_area.get('h', 0)
                        )

                        confidence = face_obj.get('confidence', 0.0)

                        detected.append(DetectedFace(
                            embedding=embedding,
                            confidence=confidence,
                            bbox=bbox
                        ))
                except Exception:
                    continue

            return detected

        except (ImportError, Exception):
            return []


class FaceClusterer:
    """Cluster similar faces using DBSCAN."""

    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        """Initialize face clusterer.

        Args:
            eps: Maximum distance between two samples
            min_samples: Minimum samples per cluster
        """
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, embeddings: List[np.ndarray]) -> List[int]:
        """Cluster face embeddings.

        Args:
            embeddings: List of face embeddings

        Returns:
            List of cluster labels (-1 for noise)
        """
        if not embeddings:
            return []

        try:
            from sklearn.cluster import DBSCAN

            # Convert to numpy array
            X = np.array(embeddings)

            # Cluster using DBSCAN
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
            labels = clustering.fit_predict(X)

            return labels.tolist()

        except (ImportError, Exception):
            # Return all as noise if clustering fails
            return [-1] * len(embeddings)


# Mock DeepFace for testing
try:
    from deepface import DeepFace
except ImportError:
    class DeepFace:
        """Mock DeepFace class for testing."""

        @staticmethod
        def extract_faces(img_path, **kwargs):
            return []

        @staticmethod
        def represent(img, **kwargs):
            return []
