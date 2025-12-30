"""LLM prompt templates for file organization tasks."""

from typing import Any, Dict, List, Optional


def create_filename_prompt(
    content: str,
    file_type: str,
    original_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Create prompt for filename suggestion.

    Args:
        content: File content or extracted text
        file_type: File type category
        original_name: Original filename
        metadata: Optional metadata

    Returns:
        Formatted prompt string
    """
    # Get file extension
    import re

    ext_match = re.search(r"\.[^.]+$", original_name)
    extension = ext_match.group(0) if ext_match else ""

    prompt = f"""Analyze this {file_type} and suggest a descriptive filename.

Original filename: {original_name}
Extension: {extension}

Content:
{content[:500]}

"""

    if metadata:
        prompt += f"Metadata: {metadata}\n\n"

    prompt += """Generate a concise, descriptive filename following these rules:
- Use lowercase with hyphens or underscores
- Include key identifying information (dates, names, type)
- Keep it under 50 characters
- Do NOT include the file extension in the name

Respond with JSON:
{
    "suggested_name": "descriptive-filename-here",
    "confidence": 0.0-1.0
}

If the content is unclear or confidence is low, set suggested_name to null."""

    return prompt


def create_category_prompt(
    content: str,
    tags: Optional[List[str]] = None,
    available_categories: Optional[List[str]] = None,
    enable_subcategories: bool = False,
) -> str:
    """Create prompt for category suggestion.

    Args:
        content: File content or extracted text
        tags: Optional content tags
        available_categories: Optional list of valid categories
        enable_subcategories: Enable hierarchical categories

    Returns:
        Formatted prompt string
    """
    prompt = f"""Analyze this content and suggest the most appropriate category.

Content:
{content[:500]}

"""

    if tags:
        prompt += f"Tags: {', '.join(tags)}\n\n"

    if available_categories:
        prompt += f"Available categories:\n"
        for cat in available_categories:
            prompt += f"- {cat}\n"
        prompt += "\n"

    if enable_subcategories:
        prompt += "You may suggest hierarchical categories using forward slashes (e.g., 'finance/invoices/2024').\n\n"

    prompt += """Respond with JSON:
{
    "category": "category-name",
    "confidence": 0.0-1.0
}

"""

    if available_categories:
        prompt += "The category MUST be from the available categories list.\n"

    prompt += "If the content is ambiguous or doesn't fit any category, set category to null."

    return prompt


def create_caption_prompt(
    detected_objects: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None
) -> str:
    """Create prompt for image captioning.

    Args:
        detected_objects: Optional list of detected objects
        context: Optional context information

    Returns:
        Formatted prompt string
    """
    prompt = "Describe this image in detail. "

    if detected_objects:
        prompt += f"The image contains: {', '.join(detected_objects)}. "

    if context:
        prompt += f"Additional context: {context}. "

    prompt += "Provide a clear, concise caption that describes what is shown in the image."

    return prompt
