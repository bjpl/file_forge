"""OCR text extraction module."""
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Check if PaddleOCR is available
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None


@dataclass
class OCRResult:
    """Result from OCR extraction."""

    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)


class OCREngine:
    """OCR engine supporting PaddleOCR and Tesseract."""

    def __init__(self, engine: str = 'paddleocr'):
        """Initialize OCR engine.

        Args:
            engine: Engine to use ('paddleocr' or 'tesseract')
        """
        # Keep requested engine name
        self.engine_name = engine
        self._engine = None
        self._fallback_active = False

        # Set fallback flag if PaddleOCR unavailable
        if engine == 'paddleocr' and not PADDLEOCR_AVAILABLE:
            self._fallback_active = True

    def extract_text(self, image_path: Path) -> List[OCRResult]:
        """Extract text from image.

        Args:
            image_path: Path to image file

        Returns:
            List of OCR results
        """
        # Use actual engine (with fallback if needed)
        if self.engine_name == 'paddleocr' and not self._fallback_active:
            return self._extract_paddle(image_path)
        else:
            return self._extract_tesseract(image_path)

    def _extract_paddle(self, image_path: Path) -> List[OCRResult]:
        """Extract text using PaddleOCR.

        Args:
            image_path: Path to image file

        Returns:
            List of OCR results
        """
        if not PADDLEOCR_AVAILABLE:
            return []

        try:
            if self._engine is None:
                self._engine = PaddleOCR(use_angle_cls=True, lang='en')

            result = self._engine.ocr(str(image_path))

            ocr_results = []
            if result and result[0]:
                for line in result[0]:
                    bbox_points = line[0]
                    text, confidence = line[1]

                    # Convert points to bbox
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=bbox
                    ))

            return ocr_results

        except Exception:
            return []

    def _extract_tesseract(self, image_path: Path) -> List[OCRResult]:
        """Extract text using Tesseract.

        Args:
            image_path: Path to image file

        Returns:
            List of OCR results
        """
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            ocr_results = []
            n_boxes = len(data['text'])

            for i in range(n_boxes):
                if int(data['conf'][i]) > 0:
                    text = data['text'][i].strip()
                    if text:
                        bbox = (
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        )
                        confidence = float(data['conf'][i]) / 100.0

                        ocr_results.append(OCRResult(
                            text=text,
                            confidence=confidence,
                            bbox=bbox
                        ))

            return ocr_results

        except Exception:
            return []
