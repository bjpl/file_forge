"""Color analysis and palette generation module."""
from pathlib import Path
from typing import List, Tuple, Dict


class ColorAnalyzer:
    """Analyze colors in images."""

    def __init__(self):
        """Initialize color analyzer."""
        pass

    def extract_dominant_colors(
        self,
        image_path: Path,
        n_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image.

        Args:
            image_path: Path to image file
            n_colors: Number of colors to extract

        Returns:
            List of RGB color tuples
        """
        try:
            from PIL import Image
            import numpy as np
            from sklearn.cluster import KMeans

            # Load image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((150, 150))  # Resize for speed

            # Convert to numpy array
            pixels = np.array(img).reshape(-1, 3)

            # Cluster colors
            kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
            kmeans.fit(pixels)

            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)

            return [tuple(color) for color in colors]

        except (ImportError, Exception):
            # Return default colors if extraction fails
            return [(128, 128, 128)] * min(n_colors, 1)

    def generate_palette(self, image_path: Path) -> Dict[str, Tuple[int, int, int]]:
        """Generate color palette for image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with primary, secondary, and accent colors
        """
        colors = self.extract_dominant_colors(image_path, n_colors=5)

        palette = {
            'primary': colors[0] if len(colors) > 0 else (128, 128, 128),
            'secondary': colors[1] if len(colors) > 1 else (128, 128, 128),
            'accent': colors[2] if len(colors) > 2 else (128, 128, 128),
        }

        return palette
