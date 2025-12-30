"""
Tag management utilities for FileForge.

Provides tag normalization, merging, synonym handling, and hierarchical
organization for flexible metadata tagging.
"""

import re
from typing import List, Set, Dict, Optional, Tuple
from collections import defaultdict

from .logging import get_logger

logger = get_logger(__name__)


def normalize_tag(
    tag: str,
    lowercase: bool = True,
    replace_spaces: bool = True,
    remove_special: bool = False,
    max_length: Optional[int] = 50,
) -> str:
    """
    Normalize a single tag according to configuration.

    Args:
        tag: Tag to normalize
        lowercase: Convert to lowercase
        replace_spaces: Replace spaces with hyphens
        remove_special: Remove special characters except hyphens
        max_length: Maximum tag length (truncate if longer)

    Returns:
        Normalized tag string

    Example:
        >>> normalize_tag("  Important Document! ")
        'important-document!'
        >>> normalize_tag("Work/Project", remove_special=True)
        'work-project'
    """
    # Strip whitespace
    normalized = tag.strip()

    if not normalized:
        return ""

    # Lowercase
    if lowercase:
        normalized = normalized.lower()

    # Replace spaces
    if replace_spaces:
        normalized = normalized.replace(" ", "-")
        # Collapse multiple hyphens
        normalized = re.sub(r"-+", "-", normalized)

    # Remove special characters
    if remove_special:
        # Keep alphanumeric, hyphens, and underscores
        normalized = re.sub(r"[^a-zA-Z0-9\-_]", "", normalized)

    # Truncate if too long
    if max_length and len(normalized) > max_length:
        normalized = normalized[:max_length]

    # Remove leading/trailing hyphens
    normalized = normalized.strip("-")

    return normalized


def merge_tags(
    existing: List[str],
    new: List[str],
    normalize: bool = True,
) -> List[str]:
    """
    Merge tag lists without duplicates.

    Args:
        existing: Current tags
        new: New tags to add
        normalize: Whether to normalize tags before merging

    Returns:
        Merged list of unique tags

    Example:
        >>> existing = ["work", "important"]
        >>> new = ["Important", "urgent", "work"]
        >>> merge_tags(existing, new)
        ['work', 'important', 'urgent']
    """
    # Normalize if requested
    if normalize:
        existing = [normalize_tag(tag) for tag in existing]
        new = [normalize_tag(tag) for tag in new]

    # Filter empty tags
    existing = [tag for tag in existing if tag]
    new = [tag for tag in new if tag]

    # Use set for deduplication while preserving order
    seen = set()
    merged = []

    for tag in existing + new:
        if tag not in seen:
            seen.add(tag)
            merged.append(tag)

    logger.debug(f"Merged tags: {len(existing)} + {len(new)} = {len(merged)} unique")
    return merged


def apply_synonyms(
    tags: List[str],
    synonym_map: Dict[str, str],
    normalize: bool = True,
) -> List[str]:
    """
    Apply synonym mappings to normalize similar tags.

    Args:
        tags: List of tags to process
        synonym_map: Dictionary mapping synonyms to canonical forms
                     e.g., {"urgent": "important", "critical": "important"}
        normalize: Whether to normalize tags first

    Returns:
        List of tags with synonyms replaced

    Example:
        >>> tags = ["urgent", "work", "critical"]
        >>> synonyms = {"urgent": "important", "critical": "important"}
        >>> apply_synonyms(tags, synonyms)
        ['important', 'work']
    """
    if normalize:
        tags = [normalize_tag(tag) for tag in tags]
        synonym_map = {normalize_tag(k): normalize_tag(v) for k, v in synonym_map.items()}

    # Apply synonym replacements
    normalized = []
    seen = set()

    for tag in tags:
        canonical = synonym_map.get(tag, tag)
        if canonical and canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)

    logger.debug(f"Applied synonyms: {len(tags)} tags -> {len(normalized)} canonical")
    return normalized


def build_tag_hierarchy(
    tags: List[str],
    separator: str = "/",
) -> Dict[str, List[str]]:
    """
    Build hierarchical tag structure from slash-separated tags.

    Groups tags by their top-level category for organization.

    Args:
        tags: List of tags (may include hierarchical tags like "work/project")
        separator: Separator for hierarchical levels

    Returns:
        Dictionary mapping categories to child tags

    Example:
        >>> tags = ["work/project", "work/meeting", "personal/health", "urgent"]
        >>> hierarchy = build_tag_hierarchy(tags)
        >>> print(hierarchy)
        {
            'work': ['project', 'meeting'],
            'personal': ['health'],
            '_root': ['urgent']
        }
    """
    hierarchy = defaultdict(list)

    for tag in tags:
        if separator in tag:
            parts = tag.split(separator, 1)
            category = parts[0].strip()
            subcategory = parts[1].strip()
            hierarchy[category].append(subcategory)
        else:
            # Tags without hierarchy go to _root
            hierarchy["_root"].append(tag)

    # Convert defaultdict to regular dict and sort
    result = {}
    for category, children in sorted(hierarchy.items()):
        result[category] = sorted(list(set(children)))

    logger.debug(f"Built hierarchy with {len(result)} categories")
    return result


def flatten_hierarchy(
    hierarchy: Dict[str, List[str]],
    separator: str = "/",
) -> List[str]:
    """
    Flatten hierarchical tag structure back to list.

    Args:
        hierarchy: Dictionary from build_tag_hierarchy
        separator: Separator for hierarchical levels

    Returns:
        Flat list of tags

    Example:
        >>> hierarchy = {'work': ['project'], '_root': ['urgent']}
        >>> flatten_hierarchy(hierarchy)
        ['work/project', 'urgent']
    """
    tags = []

    for category, children in hierarchy.items():
        if category == "_root":
            tags.extend(children)
        else:
            for child in children:
                tags.append(f"{category}{separator}{child}")

    return sorted(tags)


class TagNormalizer:
    """
    Configurable tag normalizer with synonym and hierarchy support.

    Example:
        >>> normalizer = TagNormalizer(
        ...     synonyms={"urgent": "important", "critical": "important"},
        ...     common_tags=["work", "personal", "project"]
        ... )
        >>> tags = normalizer.normalize_tags(["Urgent", "Work/Project", "NEW TAG"])
        >>> print(tags)
        ['important', 'work/project', 'new-tag']
    """

    def __init__(
        self,
        synonyms: Optional[Dict[str, str]] = None,
        common_tags: Optional[List[str]] = None,
        lowercase: bool = True,
        replace_spaces: bool = True,
        remove_special: bool = False,
        max_length: Optional[int] = 50,
        separator: str = "/",
    ):
        """
        Initialize tag normalizer.

        Args:
            synonyms: Synonym mappings
            common_tags: List of commonly used tags for suggestions
            lowercase: Convert tags to lowercase
            replace_spaces: Replace spaces with hyphens
            remove_special: Remove special characters
            max_length: Maximum tag length
            separator: Hierarchy separator
        """
        self.synonyms = synonyms or {}
        self.common_tags = set(common_tags or [])
        self.lowercase = lowercase
        self.replace_spaces = replace_spaces
        self.remove_special = remove_special
        self.max_length = max_length
        self.separator = separator

        logger.debug(f"TagNormalizer initialized with {len(self.synonyms)} synonyms")

    def normalize_tags(
        self,
        tags: List[str],
        merge_existing: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Normalize a list of tags according to configured rules.

        Args:
            tags: Tags to normalize
            merge_existing: Optional existing tags to merge with

        Returns:
            Normalized and deduplicated tag list
        """
        # Normalize individual tags
        normalized = [
            normalize_tag(
                tag,
                lowercase=self.lowercase,
                replace_spaces=self.replace_spaces,
                remove_special=self.remove_special,
                max_length=self.max_length,
            )
            for tag in tags
        ]

        # Filter empty tags
        normalized = [tag for tag in normalized if tag]

        # Apply synonyms
        normalized = apply_synonyms(normalized, self.synonyms, normalize=False)

        # Merge with existing tags if provided
        if merge_existing:
            normalized = merge_tags(merge_existing, normalized, normalize=False)

        return normalized

    def add_synonym(self, synonym: str, canonical: str) -> None:
        """Add a new synonym mapping."""
        self.synonyms[synonym] = canonical
        logger.debug(f"Added synonym: {synonym} -> {canonical}")

    def add_common_tag(self, tag: str) -> None:
        """Add a tag to the common tags list."""
        normalized = normalize_tag(
            tag,
            lowercase=self.lowercase,
            replace_spaces=self.replace_spaces,
            remove_special=self.remove_special,
        )
        self.common_tags.add(normalized)
        logger.debug(f"Added common tag: {normalized}")

    def suggest_tags(
        self,
        partial: str,
        max_suggestions: int = 5,
    ) -> List[str]:
        """
        Suggest tags based on partial input.

        Args:
            partial: Partial tag text
            max_suggestions: Maximum suggestions to return

        Returns:
            List of suggested tags

        Example:
            >>> suggestions = normalizer.suggest_tags("wo")
            >>> print(suggestions)
            ['work', 'work/project']
        """
        partial_lower = partial.lower()
        suggestions = [
            tag for tag in self.common_tags
            if partial_lower in tag.lower()
        ]

        return sorted(suggestions[:max_suggestions])

    def get_hierarchy(self, tags: List[str]) -> Dict[str, List[str]]:
        """Build hierarchical structure from tags."""
        return build_tag_hierarchy(tags, separator=self.separator)

    def validate_tag(self, tag: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a tag and return normalized version.

        Args:
            tag: Tag to validate

        Returns:
            Tuple of (is_valid, normalized_tag or error_message)

        Example:
            >>> valid, result = normalizer.validate_tag("  Work/Project  ")
            >>> if valid:
            ...     print(f"Valid tag: {result}")
            ... else:
            ...     print(f"Invalid: {result}")
        """
        if not tag or not tag.strip():
            return False, "Tag cannot be empty"

        normalized = normalize_tag(
            tag,
            lowercase=self.lowercase,
            replace_spaces=self.replace_spaces,
            remove_special=self.remove_special,
            max_length=self.max_length,
        )

        if not normalized:
            return False, "Tag contains only invalid characters"

        if self.max_length and len(normalized) > self.max_length:
            return False, f"Tag exceeds maximum length of {self.max_length}"

        return True, normalized
