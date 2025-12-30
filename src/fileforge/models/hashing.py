"""Perceptual image hashing module."""
from pathlib import Path


class ImageHasher:
    """Perceptual image hasher for duplicate detection."""

    def __init__(self, hash_size: int = 8):
        """Initialize image hasher.

        Args:
            hash_size: Size of hash (default: 8 for 64-bit hash)
        """
        self.hash_size = hash_size

    def compute_hash(self, image_path: Path) -> str:
        """Compute perceptual hash for image.

        Args:
            image_path: Path to image file

        Returns:
            Hexadecimal hash string
        """
        try:
            import imagehash
            from PIL import Image

            # Load image
            img = Image.open(image_path)

            # Compute average hash
            hash_value = imagehash.average_hash(img, hash_size=self.hash_size)

            return str(hash_value)

        except (ImportError, Exception):
            # Return placeholder hash if hashing fails
            return "0" * (self.hash_size ** 2 // 4)

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes.

        Args:
            hash1: First hash string
            hash2: Second hash string

        Returns:
            Hamming distance (number of differing bits)
        """
        try:
            import imagehash

            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)

            return h1 - h2

        except (ImportError, Exception):
            # Calculate manually
            if len(hash1) != len(hash2):
                return max(len(hash1), len(hash2))

            distance = 0
            for c1, c2 in zip(hash1, hash2):
                if c1 != c2:
                    # Count differing bits in hex digits
                    xor = int(c1, 16) ^ int(c2, 16)
                    distance += bin(xor).count('1')

            return distance
