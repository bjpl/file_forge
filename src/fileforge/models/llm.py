"""LLM client for Ollama integration.

Provides intelligent filename suggestions, content categorization,
and image captioning using local LLM models.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore


class LLMClient:
    """Client for Ollama LLM API."""

    def __init__(
        self,
        model: str = "qwen2.5:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        """Initialize LLM client.

        Args:
            model: Model name to use (default: qwen2.5:latest)
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt for context
            json_mode: Enable JSON output mode
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            Exception: If generation fails
        """
        if ollama is None:
            raise ImportError("ollama package not installed")

        try:
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "options": options,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if json_mode:
                kwargs["format"] = "json"

            response = ollama.generate(**kwargs)
            return response["response"]

        except Exception as e:
            raise Exception(f"LLM generation failed: {e}") from e


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters.

    Args:
        filename: Raw filename string

    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, "", filename)

    # Replace multiple spaces with single space
    sanitized = re.sub(r"\s+", " ", sanitized)

    # Trim whitespace
    sanitized = sanitized.strip()

    return sanitized


def suggest_filename(
    content: str,
    original_name: str,
    file_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    client: Optional[LLMClient] = None,
) -> Optional[str]:
    """Suggest filename based on content analysis.

    Args:
        content: File content or extracted text
        original_name: Original filename
        file_type: File type category
        metadata: Optional metadata (EXIF, dates, etc.)
        client: Optional LLM client instance

    Returns:
        Suggested filename or None if confidence too low
    """
    if client is None:
        client = LLMClient()

    # Get original extension
    ext = Path(original_name).suffix

    # Import prompt template
    from fileforge.models.prompts import create_filename_prompt

    prompt = create_filename_prompt(
        content=content, file_type=file_type, original_name=original_name, metadata=metadata
    )

    try:
        response = client.generate(prompt, json_mode=True)
        data = parse_json_response(response)

        # Check confidence threshold
        confidence = data.get("confidence", 0.0)
        if confidence < 0.5:
            return None

        suggested_name = data.get("suggested_name")
        if not suggested_name or suggested_name == "null":
            return None

        # Sanitize filename
        sanitized = sanitize_filename(suggested_name)

        # Ensure extension is preserved
        if not sanitized.endswith(ext):
            # Remove any existing extension
            sanitized = re.sub(r"\.[^.]+$", "", sanitized)
            sanitized = f"{sanitized}{ext}"

        return sanitized

    except Exception:
        return None


def suggest_category(
    content: str,
    tags: Optional[List[str]] = None,
    available_categories: Optional[List[str]] = None,
    enable_subcategories: bool = False,
    client: Optional[LLMClient] = None,
) -> Optional[str]:
    """Suggest category/folder based on content analysis.

    Args:
        content: File content or extracted text
        tags: Optional content tags
        available_categories: Optional list of valid categories
        enable_subcategories: Enable hierarchical categories
        client: Optional LLM client instance

    Returns:
        Suggested category or None if confidence too low
    """
    if client is None:
        client = LLMClient()

    # Import prompt template
    from fileforge.models.prompts import create_category_prompt

    prompt = create_category_prompt(
        content=content,
        tags=tags,
        available_categories=available_categories,
        enable_subcategories=enable_subcategories,
    )

    try:
        response = client.generate(prompt, json_mode=True)
        data = parse_json_response(response)

        # Check confidence threshold
        confidence = data.get("confidence", 0.0)
        if confidence < 0.5:
            return None

        category = data.get("category")
        if not category or category == "null":
            return None

        return category

    except Exception:
        return None


def caption_image(
    image_path: Path,
    detected_objects: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    model: str = "llava:7b",
) -> str:
    """Generate caption for image using vision model.

    Args:
        image_path: Path to image file
        detected_objects: Optional list of detected objects
        context: Optional context information
        model: Vision model to use

    Returns:
        Generated caption

    Raises:
        FileNotFoundError: If image doesn't exist
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if ollama is None:
        raise ImportError("ollama package not installed")

    # Import prompt template
    from fileforge.models.prompts import create_caption_prompt

    prompt = create_caption_prompt(detected_objects=detected_objects, context=context)

    try:
        # Read image as base64
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        response = ollama.generate(model=model, prompt=prompt, images=[image_data])

        return response["response"]

    except Exception as e:
        raise Exception(f"Image captioning failed: {e}") from e


def batch_caption_images(
    image_paths: List[Path],
    detected_objects: Optional[Dict[Path, List[str]]] = None,
    model: str = "llava:7b",
) -> List[str]:
    """Generate captions for multiple images.

    Args:
        image_paths: List of image paths
        detected_objects: Optional dict mapping paths to detected objects
        model: Vision model to use

    Returns:
        List of captions in same order as input
    """
    captions = []

    for path in image_paths:
        objects = None
        if detected_objects and path in detected_objects:
            objects = detected_objects[path]

        try:
            caption = caption_image(path, detected_objects=objects, model=model)
            captions.append(caption)
        except Exception:
            captions.append("")

    return captions


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response.

    Handles both raw JSON and JSON in markdown code blocks.

    Args:
        response: LLM response text

    Returns:
        Parsed JSON data

    Raises:
        json.JSONDecodeError: If JSON is malformed
    """
    # Try to extract JSON from markdown code blocks
    json_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(json_pattern, response, re.DOTALL)

    if matches:
        # Use first JSON block found
        json_str = matches[0].strip()
    else:
        # Assume entire response is JSON
        json_str = response.strip()

    return json.loads(json_str)


def validate_response(response: str, schema_class: type) -> Any:
    """Validate LLM response against Pydantic schema.

    Args:
        response: LLM response text
        schema_class: Pydantic model class

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If validation fails
    """
    data = parse_json_response(response)
    return schema_class(**data)


class LLMModel:
    """
    Wrapper class providing a unified interface to LLM functions.

    This class wraps the module-level functions for easier testing and mocking.
    """

    def __init__(self, model: str = "qwen2.5:latest"):
        """Initialize LLM model wrapper.

        Args:
            model: Model name to use
        """
        self.client = LLMClient(model=model)

    def suggest_filename(self, content: str, file_path: Optional[Path] = None) -> Optional[str]:
        """Suggest filename based on content.

        Args:
            content: File content
            file_path: Original file path

        Returns:
            Suggested filename or None
        """
        original_name = file_path.name if file_path else "file.txt"
        file_type = file_path.suffix if file_path else ".txt"
        return suggest_filename(content, original_name, file_type, client=self.client)

    def suggest_category(self, content: str) -> Optional[str]:
        """Suggest category based on content.

        Args:
            content: File content

        Returns:
            Suggested category or None
        """
        return suggest_category(content, client=self.client)

    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content.

        Args:
            content: Text content

        Returns:
            Dictionary mapping entity types to lists of entities
        """
        # Placeholder implementation - can be enhanced later
        return {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": []
        }

    def summarize(self, content: str, max_length: int = 200) -> str:
        """Generate summary of content.

        Args:
            content: Text content to summarize
            max_length: Maximum summary length

        Returns:
            Generated summary
        """
        # Placeholder implementation - can be enhanced later
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    def detect_duplicates(self, texts: List[str], threshold: float = 0.8) -> List[List[int]]:
        """Detect semantic duplicates in list of texts.

        Args:
            texts: List of text contents
            threshold: Similarity threshold

        Returns:
            List of duplicate groups (indices)
        """
        # Placeholder implementation - can be enhanced later
        return []
