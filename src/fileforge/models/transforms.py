"""Image transformation and preprocessing module."""
from pathlib import Path
from typing import Optional
import numpy as np


class ImageTransformer:
    """Transform and preprocess images."""

    def __init__(self):
        """Initialize image transformer."""
        pass

    def resize(
        self,
        image_path: Path,
        width: int,
        height: int
    ) -> Optional[np.ndarray]:
        """Resize image to target dimensions.

        Args:
            image_path: Path to image file
            width: Target width
            height: Target height

        Returns:
            Resized image as numpy array or None
        """
        try:
            from PIL import Image

            # Load and resize image
            img = Image.open(image_path)
            resized = img.resize((width, height))

            return np.array(resized)

        except (ImportError, Exception):
            return None

    def normalize(self, image_path: Path) -> Optional[np.ndarray]:
        """Normalize pixel values to [0, 1].

        Args:
            image_path: Path to image file

        Returns:
            Normalized image as numpy array or None
        """
        try:
            from PIL import Image

            # Load image
            img = Image.open(image_path)
            pixels = np.array(img, dtype=np.float32)

            # Normalize to [0, 1]
            normalized = pixels / 255.0

            return normalized

        except (ImportError, Exception):
            return None

    def augment(
        self,
        image_path: Path,
        rotation: float = 0,
        flip: bool = False
    ) -> Optional[np.ndarray]:
        """Apply data augmentation.

        Args:
            image_path: Path to image file
            rotation: Rotation angle in degrees
            flip: Whether to flip horizontally

        Returns:
            Augmented image as numpy array or None
        """
        try:
            from PIL import Image

            # Load image
            img = Image.open(image_path)

            # Apply rotation
            if rotation != 0:
                img = img.rotate(rotation, expand=True)

            # Apply flip
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            return np.array(img)

        except (ImportError, Exception):
            return None
