"""
Hashing utilities for FileForge.

Provides file content hashing for deduplication, integrity checking,
and perceptual hashing for images.
"""

import hashlib
from pathlib import Path
from typing import Union, Optional

from .logging import get_logger

logger = get_logger(__name__)

# Default chunk size for reading files (1 MB)
DEFAULT_CHUNK_SIZE = 1024 * 1024


def compute_file_hash(
    path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    Compute cryptographic hash of a file using chunked reading.

    Efficiently handles large files by reading in chunks rather than
    loading the entire file into memory.

    Args:
        path: Path to the file
        algorithm: Hash algorithm (sha256, sha512, md5, etc.)
        chunk_size: Size of chunks to read (bytes)

    Returns:
        Hexadecimal hash string

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
        ValueError: If hash algorithm is not supported

    Example:
        >>> hash_value = compute_file_hash("document.pdf")
        >>> print(f"SHA-256: {hash_value}")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

        hash_value = hasher.hexdigest()
        logger.debug(f"Computed {algorithm} hash for {path.name}: {hash_value[:16]}...")
        return hash_value

    except PermissionError as e:
        logger.error(f"Permission denied reading file: {path}")
        raise
    except Exception as e:
        logger.error(f"Error computing hash for {path}: {e}")
        raise


def compute_content_hash(
    content: Union[str, bytes],
    algorithm: str = "sha256",
) -> str:
    """
    Compute cryptographic hash of content for deduplication.

    Useful for identifying duplicate content without comparing
    entire files byte-by-byte.

    Args:
        content: String or bytes to hash
        algorithm: Hash algorithm (sha256, sha512, md5, etc.)

    Returns:
        Hexadecimal hash string

    Raises:
        ValueError: If hash algorithm is not supported

    Example:
        >>> content = "Important document content"
        >>> hash1 = compute_content_hash(content)
        >>> hash2 = compute_content_hash(content)
        >>> assert hash1 == hash2  # Same content = same hash
    """
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    # Convert string to bytes if necessary
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    else:
        content_bytes = content

    hasher.update(content_bytes)
    hash_value = hasher.hexdigest()

    logger.debug(f"Computed {algorithm} content hash: {hash_value[:16]}...")
    return hash_value


def compute_quick_hash(
    path: Union[str, Path],
    sample_size: int = 8192,
) -> str:
    """
    Compute fast hash using file size and sample bytes.

    Much faster than full file hashing, useful for quick duplicate detection.
    Not cryptographically secure but sufficient for many deduplication scenarios.

    Args:
        path: Path to the file
        sample_size: Number of bytes to sample from beginning/end

    Returns:
        Hexadecimal hash string combining size and content samples

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read

    Example:
        >>> quick_hash = compute_quick_hash("large_video.mp4")
        >>> # Much faster than full hash for large files
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    file_size = path.stat().st_size
    hasher = hashlib.sha256()

    # Include file size in hash
    hasher.update(str(file_size).encode())

    try:
        with open(path, "rb") as f:
            # Sample from beginning
            beginning = f.read(sample_size)
            hasher.update(beginning)

            # Sample from end if file is large enough
            if file_size > sample_size * 2:
                f.seek(-sample_size, 2)  # Seek from end
                end = f.read(sample_size)
                hasher.update(end)

        hash_value = hasher.hexdigest()
        logger.debug(f"Computed quick hash for {path.name}: {hash_value[:16]}...")
        return hash_value

    except PermissionError as e:
        logger.error(f"Permission denied reading file: {path}")
        raise
    except Exception as e:
        logger.error(f"Error computing quick hash for {path}: {e}")
        raise


def compute_perceptual_hash(
    path: Union[str, Path],
    hash_size: int = 8,
) -> Optional[str]:
    """
    Compute perceptual hash for images (placeholder for future implementation).

    Perceptual hashing creates similar hashes for visually similar images,
    useful for finding duplicate or near-duplicate images.

    Args:
        path: Path to the image file
        hash_size: Size of the hash (larger = more precise)

    Returns:
        Perceptual hash string, or None if image processing is unavailable

    Note:
        This is a placeholder. Full implementation requires PIL/Pillow
        and imagehash libraries. Install with: pip install pillow imagehash

    Example:
        >>> # Future usage:
        >>> phash = compute_perceptual_hash("photo.jpg")
        >>> if phash:
        ...     print(f"pHash: {phash}")
    """
    logger.warning(
        "Perceptual hashing not yet implemented. "
        "Install PIL and imagehash for image similarity detection."
    )
    # TODO: Implement when PIL/imagehash are added as optional dependencies
    # try:
    #     import PIL.Image
    #     import imagehash
    #
    #     image = PIL.Image.open(path)
    #     phash = imagehash.phash(image, hash_size=hash_size)
    #     return str(phash)
    # except ImportError:
    #     logger.debug("PIL/imagehash not available for perceptual hashing")
    #     return None
    # except Exception as e:
    #     logger.error(f"Error computing perceptual hash: {e}")
    #     return None

    return None


def verify_file_hash(
    path: Union[str, Path],
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify a file's hash matches the expected value.

    Args:
        path: Path to the file
        expected_hash: Expected hash value (hexadecimal)
        algorithm: Hash algorithm used

    Returns:
        True if hash matches, False otherwise

    Example:
        >>> is_valid = verify_file_hash("download.zip", "abc123...")
        >>> if not is_valid:
        ...     print("File corrupted or tampered!")
    """
    try:
        actual_hash = compute_file_hash(path, algorithm=algorithm)
        matches = actual_hash.lower() == expected_hash.lower()

        if matches:
            logger.debug(f"Hash verification passed for {Path(path).name}")
        else:
            logger.warning(f"Hash mismatch for {Path(path).name}")

        return matches

    except Exception as e:
        logger.error(f"Hash verification failed: {e}")
        return False
