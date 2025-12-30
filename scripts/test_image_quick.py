#!/usr/bin/env python3
"""Quick test of image processing implementation."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Test imports
print("Testing imports...")
from fileforge.pipeline.processors.image import ImageProcessor, ImageAnalysis
from fileforge.models.detector import ObjectDetector, DetectedObject
from fileforge.models.ocr import OCREngine, OCRResult
from fileforge.models.faces import FaceDetector, DetectedFace, FaceClusterer
from fileforge.models.nsfw import NSFWClassifier, NSFWResult
from fileforge.models.caption import ImageCaptioner, CaptionResult
from fileforge.models.hashing import ImageHasher
from fileforge.models.colors import ColorAnalyzer
from fileforge.models.quality import QualityAssessor
from fileforge.models.transforms import ImageTransformer
from fileforge.pipeline.core import Pipeline
from fileforge.core.metadata import FileMetadata
print("✓ All imports successful")

# Test basic instantiation
print("\nTesting instantiation...")
processor = ImageProcessor()
print(f"✓ ImageProcessor created with {len(processor.supported_extensions)} supported extensions")
print(f"  Supported: {sorted(processor.supported_extensions)}")

detector = ObjectDetector(confidence_threshold=0.5)
print(f"✓ ObjectDetector created with threshold={detector.confidence_threshold}")

ocr_paddle = OCREngine(engine='paddleocr')
print(f"✓ OCREngine created with engine={ocr_paddle.engine_name}")

ocr_tess = OCREngine(engine='tesseract')
print(f"✓ OCREngine (tesseract) created with engine={ocr_tess.engine_name}")

face_detector = FaceDetector()
print("✓ FaceDetector created")

face_clusterer = FaceClusterer(eps=0.5, min_samples=2)
print(f"✓ FaceClusterer created with eps={face_clusterer.eps}, min_samples={face_clusterer.min_samples}")

nsfw = NSFWClassifier()
print("✓ NSFWClassifier created")

captioner = ImageCaptioner()
print("✓ ImageCaptioner created")

hasher = ImageHasher()
print("✓ ImageHasher created")

colors = ColorAnalyzer()
print("✓ ColorAnalyzer created")

quality = QualityAssessor()
print("✓ QualityAssessor created")

transformer = ImageTransformer()
print("✓ ImageTransformer created")

pipeline = Pipeline()
print("✓ Pipeline created")

# Test dataclasses
print("\nTesting dataclasses...")
obj = DetectedObject(label='person', confidence=0.95, bbox=(10, 20, 100, 200))
print(f"✓ DetectedObject: label={obj.label}, confidence={obj.confidence}")

ocr_res = OCRResult(text="Hello", confidence=0.98, bbox=(5, 5, 50, 20))
print(f"✓ OCRResult: text={ocr_res.text}, confidence={ocr_res.confidence}")

import numpy as np
face = DetectedFace(
    embedding=np.random.rand(512).astype(np.float32),
    confidence=0.99,
    bbox=(50, 50, 150, 200)
)
print(f"✓ DetectedFace: embedding_shape={face.embedding.shape}, confidence={face.confidence}")

nsfw_res = NSFWResult(is_safe=True, categories=[], confidence=0.99)
print(f"✓ NSFWResult: is_safe={nsfw_res.is_safe}, categories={nsfw_res.categories}")

caption_res = CaptionResult(text="A person in a park", confidence=0.92)
print(f"✓ CaptionResult: text={caption_res.text[:20]}..., confidence={caption_res.confidence}")

analysis = ImageAnalysis()
print(f"✓ ImageAnalysis: has {len(analysis.detected_objects)} objects")

# Test Pipeline integration
print("\nTesting Pipeline integration...")
pipeline.register_processor(processor)
print(f"✓ Processor registered with pipeline")
print(f"  Pipeline supports: {sorted(pipeline.supported_extensions)}")

# Test metadata
print("\nTesting metadata...")
metadata = FileMetadata(file_type='image')
print(f"✓ FileMetadata created: file_type={metadata.file_type}")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
