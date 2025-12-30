"""
Windows-specific utilities for FileForge.

Provides Windows file system features including long path handling,
file attributes, and NTFS Alternate Data Streams (ADS).
"""

import os
import platform
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any

from .logging import get_logger

logger = get_logger(__name__)


def is_windows() -> bool:
    """
    Check if running on Windows platform.

    Returns:
        True if Windows, False otherwise

    Example:
        >>> if is_windows():
        ...     print("Windows-specific features available")
    """
    return sys.platform == "win32" or platform.system() == "Windows"


def handle_long_path(path: Union[str, Path]) -> str:
    r"""
    Convert path to Windows long path format if needed.

    Windows has a 260-character path limit (MAX_PATH). The \\?\ prefix
    allows paths up to ~32,000 characters.

    Args:
        path: Path to convert

    Returns:
        Path with \\?\ prefix on Windows if not already present,
        original path on other platforms

    Example:
        >>> long_path = handle_long_path("C:/very/long/path/to/file.txt")
        >>> # Returns: \\\\?\\C:\\very\\long\\path\\to\\file.txt
    """
    if not is_windows():
        return str(path)

    path_str = str(path)

    # Already has long path prefix
    if path_str.startswith("\\\\?\\"):
        return path_str

    # UNC paths (network paths)
    if path_str.startswith("\\\\"):
        return f"\\\\?\\UNC\\{path_str[2:]}"

    # Convert to absolute path and add prefix
    abs_path = os.path.abspath(path_str)

    # Add long path prefix
    return f"\\\\?\\{abs_path}"


def get_file_attributes(path: Union[str, Path]) -> Optional[Dict[str, bool]]:
    """
    Get Windows file attributes.

    Retrieves attributes like hidden, system, archive, read-only, etc.
    Gracefully returns None on non-Windows platforms.

    Args:
        path: Path to the file or directory

    Returns:
        Dictionary of attribute names to boolean values, or None if unavailable

    Example:
        >>> attrs = get_file_attributes("important.doc")
        >>> if attrs and attrs["hidden"]:
        ...     print("File is hidden")
    """
    if not is_windows():
        logger.debug("File attributes only available on Windows")
        return None

    try:
        import ctypes
        from ctypes import wintypes

        # Windows API constants
        FILE_ATTRIBUTE_READONLY = 0x01
        FILE_ATTRIBUTE_HIDDEN = 0x02
        FILE_ATTRIBUTE_SYSTEM = 0x04
        FILE_ATTRIBUTE_DIRECTORY = 0x10
        FILE_ATTRIBUTE_ARCHIVE = 0x20
        FILE_ATTRIBUTE_NORMAL = 0x80
        FILE_ATTRIBUTE_TEMPORARY = 0x100
        FILE_ATTRIBUTE_COMPRESSED = 0x800
        FILE_ATTRIBUTE_ENCRYPTED = 0x4000

        # Get attributes via Windows API
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        attrs = kernel32.GetFileAttributesW(str(path))

        if attrs == -1:
            logger.error(f"Failed to get attributes for {path}")
            return None

        # Parse attributes
        return {
            "readonly": bool(attrs & FILE_ATTRIBUTE_READONLY),
            "hidden": bool(attrs & FILE_ATTRIBUTE_HIDDEN),
            "system": bool(attrs & FILE_ATTRIBUTE_SYSTEM),
            "directory": bool(attrs & FILE_ATTRIBUTE_DIRECTORY),
            "archive": bool(attrs & FILE_ATTRIBUTE_ARCHIVE),
            "normal": bool(attrs & FILE_ATTRIBUTE_NORMAL),
            "temporary": bool(attrs & FILE_ATTRIBUTE_TEMPORARY),
            "compressed": bool(attrs & FILE_ATTRIBUTE_COMPRESSED),
            "encrypted": bool(attrs & FILE_ATTRIBUTE_ENCRYPTED),
        }

    except ImportError:
        logger.debug("ctypes not available")
        return None
    except Exception as e:
        logger.error(f"Error getting file attributes: {e}")
        return None


def set_file_attributes(
    path: Union[str, Path],
    attributes: Dict[str, bool],
) -> bool:
    """
    Set Windows file attributes.

    Args:
        path: Path to the file or directory
        attributes: Dictionary mapping attribute names to desired values

    Returns:
        True if successful, False otherwise

    Example:
        >>> success = set_file_attributes(
        ...     "secret.txt",
        ...     {"hidden": True, "readonly": True}
        ... )
    """
    if not is_windows():
        logger.debug("File attributes only available on Windows")
        return False

    try:
        import ctypes
        from ctypes import wintypes

        # Windows API constants
        FILE_ATTRIBUTE_READONLY = 0x01
        FILE_ATTRIBUTE_HIDDEN = 0x02
        FILE_ATTRIBUTE_SYSTEM = 0x04
        FILE_ATTRIBUTE_ARCHIVE = 0x20
        FILE_ATTRIBUTE_NORMAL = 0x80

        # Build attribute mask
        attrs = 0
        if attributes.get("readonly"):
            attrs |= FILE_ATTRIBUTE_READONLY
        if attributes.get("hidden"):
            attrs |= FILE_ATTRIBUTE_HIDDEN
        if attributes.get("system"):
            attrs |= FILE_ATTRIBUTE_SYSTEM
        if attributes.get("archive"):
            attrs |= FILE_ATTRIBUTE_ARCHIVE

        # Normal is special - only set if no other attributes
        if attrs == 0 and attributes.get("normal"):
            attrs = FILE_ATTRIBUTE_NORMAL

        # Set attributes via Windows API
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        result = kernel32.SetFileAttributesW(str(path), attrs)

        if result == 0:
            logger.error(f"Failed to set attributes for {path}")
            return False

        logger.debug(f"Set attributes for {path}: {attributes}")
        return True

    except ImportError:
        logger.debug("ctypes not available")
        return False
    except Exception as e:
        logger.error(f"Error setting file attributes: {e}")
        return False


def read_ads(
    path: Union[str, Path],
    stream_name: str,
) -> Optional[bytes]:
    """
    Read NTFS Alternate Data Stream.

    NTFS ADS allows storing additional data streams alongside the main file.
    Useful for metadata, tags, or hidden information.

    Args:
        path: Path to the file
        stream_name: Name of the alternate data stream

    Returns:
        Stream content as bytes, or None if unavailable/error

    Example:
        >>> tags = read_ads("document.pdf", "tags")
        >>> if tags:
        ...     print(f"Tags: {tags.decode('utf-8')}")
    """
    if not is_windows():
        logger.debug("Alternate Data Streams only available on Windows NTFS")
        return None

    try:
        # ADS accessed via filename:streamname syntax
        ads_path = f"{path}:{stream_name}"

        with open(ads_path, "rb") as f:
            content = f.read()

        logger.debug(f"Read {len(content)} bytes from ADS '{stream_name}' on {Path(path).name}")
        return content

    except FileNotFoundError:
        logger.debug(f"ADS '{stream_name}' not found on {path}")
        return None
    except Exception as e:
        logger.error(f"Error reading ADS: {e}")
        return None


def write_ads(
    path: Union[str, Path],
    stream_name: str,
    data: Union[str, bytes],
) -> bool:
    """
    Write NTFS Alternate Data Stream.

    Args:
        path: Path to the file
        stream_name: Name of the alternate data stream
        data: Data to write (string or bytes)

    Returns:
        True if successful, False otherwise

    Example:
        >>> success = write_ads(
        ...     "document.pdf",
        ...     "tags",
        ...     "important,financial,2024"
        ... )
    """
    if not is_windows():
        logger.debug("Alternate Data Streams only available on Windows NTFS")
        return False

    try:
        # Convert string to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data

        # ADS accessed via filename:streamname syntax
        ads_path = f"{path}:{stream_name}"

        with open(ads_path, "wb") as f:
            f.write(data_bytes)

        logger.debug(f"Wrote {len(data_bytes)} bytes to ADS '{stream_name}' on {Path(path).name}")
        return True

    except Exception as e:
        logger.error(f"Error writing ADS: {e}")
        return False


def list_ads(path: Union[str, Path]) -> Optional[list[str]]:
    """
    List all Alternate Data Streams on a file.

    Args:
        path: Path to the file

    Returns:
        List of stream names, or None if unavailable/error

    Example:
        >>> streams = list_ads("document.pdf")
        >>> if streams:
        ...     print(f"Found streams: {', '.join(streams)}")
    """
    if not is_windows():
        logger.debug("Alternate Data Streams only available on Windows NTFS")
        return None

    try:
        import ctypes
        from ctypes import wintypes

        # This is a simplified version - full implementation requires
        # calling FindFirstStreamW/FindNextStreamW Win32 APIs
        logger.warning("ADS enumeration requires advanced Win32 API calls")
        logger.info("Use 'dir /r' command or PowerShell Get-Item -Stream * for full list")

        # TODO: Implement using FindFirstStreamW/FindNextStreamW
        return None

    except Exception as e:
        logger.error(f"Error listing ADS: {e}")
        return None


def delete_ads(
    path: Union[str, Path],
    stream_name: str,
) -> bool:
    """
    Delete an Alternate Data Stream.

    Args:
        path: Path to the file
        stream_name: Name of the stream to delete

    Returns:
        True if successful, False otherwise

    Example:
        >>> success = delete_ads("document.pdf", "old_tags")
    """
    if not is_windows():
        logger.debug("Alternate Data Streams only available on Windows NTFS")
        return False

    try:
        ads_path = f"{path}:{stream_name}"
        os.remove(ads_path)

        logger.debug(f"Deleted ADS '{stream_name}' from {Path(path).name}")
        return True

    except FileNotFoundError:
        logger.debug(f"ADS '{stream_name}' not found on {path}")
        return False
    except Exception as e:
        logger.error(f"Error deleting ADS: {e}")
        return False
