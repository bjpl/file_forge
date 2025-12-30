"""Image quality assessment module."""
from pathlib import Path
import numpy as np


class QualityAssessor:
    """Assess image quality metrics."""

    def __init__(self):
        """Initialize quality assessor."""
        pass

    def calculate_sharpness(self, image_path: Path) -> float:
        """Calculate image sharpness using Laplacian variance.

        Args:
            image_path: Path to image file

        Returns:
            Sharpness score (higher = sharper)
        """
        try:
            import cv2

            # Load image in grayscale
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            # Calculate Laplacian
            laplacian = cv2.Laplacian(img, cv2.CV_64F)

            # Return variance as sharpness score
            return float(laplacian.var())

        except (ImportError, Exception):
            return 100.0  # Default sharpness

    def is_blurry(self, image_path: Path, threshold: float = 100.0) -> bool:
        """Detect if image is blurry.

        Args:
            image_path: Path to image file
            threshold: Sharpness threshold (lower = blurry)

        Returns:
            True if image is blurry
        """
        sharpness = self.calculate_sharpness(image_path)
        return sharpness < threshold

    def calculate_brightness(self, image_path: Path) -> float:
        """Calculate average brightness.

        Args:
            image_path: Path to image file

        Returns:
            Average brightness (0-255)
        """
        try:
            from PIL import Image
            import numpy as np

            # Load image
            img = Image.open(image_path).convert('L')  # Grayscale

            # Calculate mean brightness
            pixels = np.array(img)
            brightness = float(pixels.mean())

            return brightness

        except (ImportError, Exception):
            return 128.0  # Default mid-brightness
