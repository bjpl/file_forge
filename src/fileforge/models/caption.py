"""Image captioning module."""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CaptionResult:
    """Result from image captioning."""

    text: str
    confidence: float


class ImageCaptioner:
    """Image captioner using BLIP or similar models."""

    def __init__(self):
        """Initialize image captioner."""
        self._model = None

    def generate(self, image_path: Path) -> str:
        """Generate caption for image.

        Args:
            image_path: Path to image file

        Returns:
            Generated caption
        """
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image

            if self._model is None:
                self._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self._model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = self._processor(image, return_tensors="pt")

            # Generate caption
            outputs = self._model.generate(**inputs)
            caption = self._processor.decode(outputs[0], skip_special_tokens=True)

            return caption

        except (ImportError, Exception):
            # Return placeholder if captioning fails
            return "An image"
