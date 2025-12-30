"""
FileForge utility modules.

Provides logging, hashing, Windows-specific features, and tag management.
"""

from .logging import (
    get_logger,
    setup_logging,
    set_log_level,
    get_log_file_path,
)
from .hashing import (
    compute_file_hash,
    compute_content_hash,
    compute_quick_hash,
    compute_perceptual_hash,
    verify_file_hash,
)
from .windows import (
    is_windows,
    handle_long_path,
    get_file_attributes,
    set_file_attributes,
    read_ads,
    write_ads,
    list_ads,
    delete_ads,
)
from .tags import (
    normalize_tag,
    merge_tags,
    apply_synonyms,
    build_tag_hierarchy,
    flatten_hierarchy,
    TagNormalizer,
)

__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    "set_log_level",
    "get_log_file_path",
    # Hashing
    "compute_file_hash",
    "compute_content_hash",
    "compute_quick_hash",
    "compute_perceptual_hash",
    "verify_file_hash",
    # Windows
    "is_windows",
    "handle_long_path",
    "get_file_attributes",
    "set_file_attributes",
    "read_ads",
    "write_ads",
    "list_ads",
    "delete_ads",
    # Tags
    "normalize_tag",
    "merge_tags",
    "apply_synonyms",
    "build_tag_hierarchy",
    "flatten_hierarchy",
    "TagNormalizer",
]
