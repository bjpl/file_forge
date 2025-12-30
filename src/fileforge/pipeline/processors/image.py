"""Image processor for FileForge pipeline."""
from pathlib import Path
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field

from fileforge.models.detector import ObjectDetector, DetectedObject
from fileforge.models.ocr import OCREngine, OCRResult
from fileforge.models.faces import FaceDetector, DetectedFace
from fileforge.models.nsfw import NSFWClassifier, NSFWResult
from fileforge.models.caption import ImageCaptioner


@dataclass
class ImageAnalysis:
    """Result from image analysis."""

    detected_objects: List[DetectedObject] = field(default_factory=list)
    extracted_text: List[OCRResult] = field(default_factory=list)
    faces: List[DetectedFace] = field(default_factory=list)
    nsfw_flags: NSFWResult = None
    caption: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageProcessor:
    """Process images through the FileForge pipeline."""

    # Class attribute for supported extensions
    supported_extensions = [
        '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'
    ]

    def __init__(self):
        """Initialize image processor."""
        # Initialize models lazily
        self._object_detector = None
        self._ocr_engine = None
        self._face_detector = None
        self._nsfw_classifier = None
        self._captioner = None

    def process(self, image_path: Path) -> ImageAnalysis:
        """Process an image file.

        Args:
            image_path: Path to image file

        Returns:
            ImageAnalysis with all results

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If unsupported file format
        """
        # Validate file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Validate file extension
        if image_path.suffix.lower() not in ImageProcessor.supported_extensions:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        # Extract metadata
        metadata = self._extract_metadata(image_path)

        # Run CPU triage first
        triage = self._cpu_triage(image_path)

        # Initialize analysis result
        analysis = ImageAnalysis(metadata=metadata)

        # Run object detection
        if self._object_detector is None:
            self._object_detector = ObjectDetector()
        analysis.detected_objects = self._object_detector.detect(image_path)

        # Run OCR
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine()
        analysis.extracted_text = self._ocr_engine.extract_text(image_path)

        # Run face detection
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        analysis.faces = self._face_detector.detect(image_path)

        # Run NSFW classification
        if self._nsfw_classifier is None:
            self._nsfw_classifier = NSFWClassifier()
        analysis.nsfw_flags = self._nsfw_classifier.classify(image_path)

        # Generate caption
        if self._captioner is None:
            self._captioner = ImageCaptioner()
        analysis.caption = self._captioner.generate(image_path)

        return analysis

    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract image metadata.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with metadata

        Raises:
            Exception: If image is corrupted or cannot be opened
        """
        try:
            from PIL import Image

            img = Image.open(image_path)
            # Verify image is valid by loading it
            img.verify()

            # Re-open after verify (verify closes the file)
            img = Image.open(image_path)

            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
            }

        except Exception as e:
            # Re-raise for corrupted images
            raise Exception(f"Failed to extract image metadata: {str(e)}") from e

    def _cpu_triage(self, image_path: Path) -> Dict[str, Any]:
        """Fast CPU-based pre-screening.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with triage results
        """
        # Basic triage - check file size, dimensions, etc.
        triage = {
            'priority': 'normal',
            'has_faces': False,
            'nsfw_prescreened': False,
        }

        try:
            # Quick checks that don't require heavy models
            metadata = self._extract_metadata(image_path)

            # Adjust priority based on image size
            if metadata.get('width', 0) > 1920 or metadata.get('height', 0) > 1080:
                triage['priority'] = 'high'

        except Exception:
            pass

        return triage

    def _gpu_analyze(self, image_path: Path) -> Dict[str, Any]:
        """Run GPU-intensive analysis.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with analysis results
        """
        # This would run deep learning models
        # For now, delegated to process() method
        return {}

    def process_batch(
        self,
        image_paths: List[Path],
        batch_size: int = 8
    ) -> List[ImageAnalysis]:
        """Process multiple images in batches.

        Args:
            image_paths: List of image paths
            batch_size: Number of images per batch

        Returns:
            List of ImageAnalysis results
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]

            for image_path in batch:
                try:
                    result = self.process(image_path)
                    results.append(result)
                except Exception as e:
                    # Create empty result for failed images
                    results.append(ImageAnalysis(
                        metadata={'error': str(e)}
                    ))

        return results
