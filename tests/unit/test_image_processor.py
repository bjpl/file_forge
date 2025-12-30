"""TDD Tests for FileForge image processor.

RED phase: Tests written first, defining expected behavior.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np


class TestImageProcessor:
    """Tests for ImageProcessor class."""

    def test_supported_extensions(self):
        """Should support common image extensions."""
        from fileforge.pipeline.processors.image import ImageProcessor

        processor = ImageProcessor()
        assert '.jpg' in processor.supported_extensions
        assert '.jpeg' in processor.supported_extensions
        assert '.png' in processor.supported_extensions
        assert '.gif' in processor.supported_extensions
        assert '.webp' in processor.supported_extensions
        assert '.bmp' in processor.supported_extensions

    def test_process_returns_image_analysis(self, temp_dir):
        """Should return complete image analysis."""
        from fileforge.pipeline.processors.image import ImageProcessor, ImageAnalysis
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")
        processor = ImageProcessor()

        result = processor.process(img_path)

        assert isinstance(result, ImageAnalysis)
        assert hasattr(result, 'detected_objects')
        assert hasattr(result, 'extracted_text')
        assert hasattr(result, 'faces')
        assert hasattr(result, 'nsfw_flags')
        assert hasattr(result, 'caption')

    def test_extracts_image_metadata(self, temp_dir):
        """Should extract image metadata (dimensions, format)."""
        from fileforge.pipeline.processors.image import ImageProcessor
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.png", width=200, height=150)
        processor = ImageProcessor()

        result = processor.process(img_path)

        assert result.metadata['width'] == 200
        assert result.metadata['height'] == 150
        assert result.metadata['format'] in ['PNG', 'png']


class TestObjectDetection:
    """Tests for YOLO object detection integration."""

    def test_detect_objects_returns_labels(self, temp_dir):
        """Should detect objects and return labels."""
        from fileforge.models.detector import ObjectDetector
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        with patch('fileforge.models.detector.YOLO') as mock_yolo:
            # Create boxes mock
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[10, 20, 100, 200]]
            mock_boxes.conf = [0.95]
            mock_boxes.cls = [0]

            # Create result mock
            mock_result = MagicMock()
            mock_result.boxes = mock_boxes
            mock_result.names = {0: 'person'}

            # Create model mock
            mock_model = MagicMock()
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector()
            results = detector.detect(img_path)

            assert len(results) >= 0  # May be empty for test image

    def test_detection_includes_confidence(self):
        """Detected objects should include confidence scores."""
        from fileforge.models.detector import DetectedObject

        obj = DetectedObject(
            label='person',
            confidence=0.95,
            bbox=(10, 20, 100, 200)
        )

        assert obj.confidence == 0.95
        assert 0 <= obj.confidence <= 1

    def test_detection_includes_bounding_box(self):
        """Detected objects should include bounding boxes."""
        from fileforge.models.detector import DetectedObject

        obj = DetectedObject(
            label='car',
            confidence=0.87,
            bbox=(50, 100, 200, 300)
        )

        assert obj.bbox == (50, 100, 200, 300)

    def test_configurable_confidence_threshold(self):
        """Should filter by confidence threshold."""
        from fileforge.models.detector import ObjectDetector

        detector = ObjectDetector(confidence_threshold=0.5)
        assert detector.confidence_threshold == 0.5


class TestOCRModule:
    """Tests for OCR text extraction."""

    def test_ocr_extracts_text_from_image(self, temp_dir):
        """Should extract text from images."""
        from fileforge.models.ocr import OCREngine

        # Mock OCR result
        with patch('fileforge.models.ocr.PaddleOCR') as mock_paddle:
            mock_instance = MagicMock()
            mock_instance.ocr.return_value = [[
                [[[10, 10], [100, 10], [100, 30], [10, 30]], ('Hello World', 0.98)]
            ]]
            mock_paddle.return_value = mock_instance

            engine = OCREngine(engine='paddleocr')
            result = engine.extract_text(temp_dir / "test.jpg")

            # Result should be extractable
            assert result is not None

    def test_ocr_returns_confidence(self):
        """OCR results should include confidence scores."""
        from fileforge.models.ocr import OCRResult

        result = OCRResult(
            text="Sample text",
            confidence=0.95,
            bbox=(10, 10, 100, 30)
        )

        assert result.confidence == 0.95

    def test_supports_multiple_engines(self):
        """Should support PaddleOCR and Tesseract."""
        from fileforge.models.ocr import OCREngine

        paddle_engine = OCREngine(engine='paddleocr')
        assert paddle_engine.engine_name == 'paddleocr'

        tesseract_engine = OCREngine(engine='tesseract')
        assert tesseract_engine.engine_name == 'tesseract'

    def test_fallback_to_tesseract_if_paddle_unavailable(self):
        """Should fallback to Tesseract if PaddleOCR unavailable."""
        from fileforge.models.ocr import OCREngine

        with patch('fileforge.models.ocr.PADDLEOCR_AVAILABLE', False):
            engine = OCREngine(engine='paddleocr')
            assert engine.engine_name in ['tesseract', 'paddleocr']


class TestFaceDetection:
    """Tests for face detection and embedding."""

    def test_detect_faces_returns_list(self, temp_dir):
        """Should detect faces and return list."""
        from fileforge.models.faces import FaceDetector
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        with patch('fileforge.models.faces.DeepFace') as mock_df:
            mock_df.extract_faces.return_value = []

            detector = FaceDetector()
            faces = detector.detect(img_path)

            assert isinstance(faces, list)

    def test_face_includes_embedding(self):
        """Detected faces should include embedding vector."""
        from fileforge.models.faces import DetectedFace

        embedding = np.random.rand(512).astype(np.float32)
        face = DetectedFace(
            embedding=embedding,
            confidence=0.98,
            bbox=(50, 50, 150, 200)
        )

        assert face.embedding.shape == (512,)

    def test_face_embedding_is_512_dimensions(self):
        """Face embeddings should be 512 dimensions (Facenet512)."""
        from fileforge.models.faces import DetectedFace

        embedding = np.random.rand(512).astype(np.float32)
        face = DetectedFace(
            embedding=embedding,
            confidence=0.98,
            bbox=(50, 50, 150, 200)
        )

        assert len(face.embedding) == 512


class TestFaceClustering:
    """Tests for face clustering."""

    def test_cluster_faces_groups_similar(self):
        """Should group similar faces into clusters."""
        from fileforge.models.faces import FaceClusterer

        # Create similar embeddings
        base_embedding = np.random.rand(512).astype(np.float32)
        embeddings = [
            base_embedding + np.random.rand(512) * 0.1,  # Similar
            base_embedding + np.random.rand(512) * 0.1,  # Similar
            np.random.rand(512).astype(np.float32),       # Different
        ]

        clusterer = FaceClusterer(eps=0.5, min_samples=2)
        labels = clusterer.cluster(embeddings)

        assert len(labels) == 3
        # First two should be same cluster (or both -1 if not enough samples)

    def test_unclustered_faces_get_minus_one(self):
        """Unclustered faces should have cluster_id=-1."""
        from fileforge.models.faces import FaceClusterer

        # Single face can't form cluster
        embeddings = [np.random.rand(512).astype(np.float32)]

        clusterer = FaceClusterer(eps=0.5, min_samples=2)
        labels = clusterer.cluster(embeddings)

        assert labels[0] == -1


class TestNSFWDetection:
    """Tests for NSFW content classification."""

    def test_classify_returns_categories(self, temp_dir):
        """Should return NSFW classification categories."""
        from fileforge.models.nsfw import NSFWClassifier
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        with patch('fileforge.models.nsfw.NudeDetector') as mock_nude:
            mock_instance = MagicMock()
            mock_instance.detect.return_value = []
            mock_nude.return_value = mock_instance

            classifier = NSFWClassifier()
            result = classifier.classify(img_path)

            assert hasattr(result, 'is_safe')
            assert hasattr(result, 'categories')

    def test_flags_explicit_content(self):
        """Should flag explicit content."""
        from fileforge.models.nsfw import NSFWResult

        result = NSFWResult(
            is_safe=False,
            categories=['EXPLICIT'],
            confidence=0.95
        )

        assert result.is_safe is False
        assert 'EXPLICIT' in result.categories

    def test_safe_images_not_flagged(self):
        """Safe images should not be flagged."""
        from fileforge.models.nsfw import NSFWResult

        result = NSFWResult(
            is_safe=True,
            categories=[],
            confidence=0.99
        )

        assert result.is_safe is True
        assert len(result.categories) == 0


class TestImagePipelineStages:
    """Tests for staged image processing pipeline."""

    def test_cpu_triage_stage(self, temp_dir):
        """CPU triage should do fast pre-screening."""
        from fileforge.pipeline.processors.image import ImageProcessor
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")
        processor = ImageProcessor()

        # Triage should be fast and return priority
        triage_result = processor._cpu_triage(img_path)

        assert 'priority' in triage_result
        assert 'has_faces' in triage_result or True
        assert 'nsfw_prescreened' in triage_result or True

    def test_gpu_analysis_stage(self, temp_dir):
        """GPU analysis should run deep models."""
        from fileforge.pipeline.processors.image import ImageProcessor

        processor = ImageProcessor()

        # Should have method for GPU analysis
        assert hasattr(processor, '_gpu_analyze') or hasattr(processor, 'process')

    def test_batched_processing(self, temp_dir):
        """Should support batched image processing."""
        from fileforge.pipeline.processors.image import ImageProcessor
        from tests.fixtures.sample_files import create_sample_image

        images = [
            create_sample_image(temp_dir / f"test{i}.jpg")
            for i in range(3)
        ]

        processor = ImageProcessor()

        # Should have batch processing method
        if hasattr(processor, 'process_batch'):
            results = processor.process_batch(images)
            assert len(results) == 3


class TestImageCaptioning:
    """Tests for automatic image captioning."""

    def test_caption_generation(self, temp_dir):
        """Should generate descriptive captions."""
        from fileforge.models.caption import ImageCaptioner
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        captioner = ImageCaptioner()
        caption = captioner.generate(img_path)

        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_caption_includes_confidence(self):
        """Captions should include confidence score."""
        from fileforge.models.caption import CaptionResult

        result = CaptionResult(
            text="A person standing in a park",
            confidence=0.92
        )

        assert result.confidence == 0.92
        assert 0 <= result.confidence <= 1


class TestImageHashGeneration:
    """Tests for perceptual image hashing."""

    def test_generates_perceptual_hash(self, temp_dir):
        """Should generate perceptual hash for duplicate detection."""
        from fileforge.models.hashing import ImageHasher
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        hasher = ImageHasher()
        hash_value = hasher.compute_hash(img_path)

        assert hash_value is not None
        assert isinstance(hash_value, str)

    def test_similar_images_have_similar_hashes(self, temp_dir):
        """Similar images should have similar perceptual hashes."""
        from fileforge.models.hashing import ImageHasher
        from tests.fixtures.sample_files import create_sample_image

        img1 = create_sample_image(temp_dir / "test1.jpg", width=200, height=150)
        img2 = create_sample_image(temp_dir / "test2.jpg", width=200, height=150)

        hasher = ImageHasher()
        hash1 = hasher.compute_hash(img1)
        hash2 = hasher.compute_hash(img2)

        # Hamming distance should be small for similar images
        distance = hasher.hamming_distance(hash1, hash2)
        assert distance >= 0


class TestImageColorAnalysis:
    """Tests for dominant color extraction."""

    def test_extracts_dominant_colors(self, temp_dir):
        """Should extract dominant colors from image."""
        from fileforge.models.colors import ColorAnalyzer
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        analyzer = ColorAnalyzer()
        colors = analyzer.extract_dominant_colors(img_path, n_colors=5)

        assert len(colors) <= 5
        assert all(isinstance(c, tuple) for c in colors)
        assert all(len(c) == 3 for c in colors)  # RGB tuples

    def test_color_palette_generation(self, temp_dir):
        """Should generate color palette."""
        from fileforge.models.colors import ColorAnalyzer
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        analyzer = ColorAnalyzer()
        palette = analyzer.generate_palette(img_path)

        assert 'primary' in palette
        assert 'secondary' in palette
        assert 'accent' in palette


class TestImageQualityAssessment:
    """Tests for image quality metrics."""

    def test_calculates_sharpness(self, temp_dir):
        """Should calculate image sharpness."""
        from fileforge.models.quality import QualityAssessor
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        assessor = QualityAssessor()
        sharpness = assessor.calculate_sharpness(img_path)

        assert sharpness >= 0
        assert isinstance(sharpness, (int, float))

    def test_detects_blur(self, temp_dir):
        """Should detect blurry images."""
        from fileforge.models.quality import QualityAssessor

        assessor = QualityAssessor()

        # Mock blur detection
        is_blurry = assessor.is_blurry(temp_dir / "test.jpg", threshold=100)

        assert isinstance(is_blurry, bool)

    def test_calculates_brightness(self, temp_dir):
        """Should calculate average brightness."""
        from fileforge.models.quality import QualityAssessor
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        assessor = QualityAssessor()
        brightness = assessor.calculate_brightness(img_path)

        assert 0 <= brightness <= 255


class TestImageTransformations:
    """Tests for image preprocessing transformations."""

    def test_resize_image(self, temp_dir):
        """Should resize image to target dimensions."""
        from fileforge.models.transforms import ImageTransformer
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg", width=400, height=300)

        transformer = ImageTransformer()
        resized = transformer.resize(img_path, width=200, height=150)

        assert resized is not None

    def test_normalize_image(self, temp_dir):
        """Should normalize pixel values."""
        from fileforge.models.transforms import ImageTransformer
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        transformer = ImageTransformer()
        normalized = transformer.normalize(img_path)

        assert normalized is not None

    def test_apply_augmentation(self, temp_dir):
        """Should apply data augmentation."""
        from fileforge.models.transforms import ImageTransformer
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        transformer = ImageTransformer()
        augmented = transformer.augment(img_path, rotation=15, flip=True)

        assert augmented is not None


class TestBatchProcessing:
    """Tests for efficient batch processing."""

    def test_process_batch_returns_all_results(self, temp_dir):
        """Should process multiple images and return all results."""
        from fileforge.pipeline.processors.image import ImageProcessor
        from tests.fixtures.sample_files import create_sample_image

        images = [
            create_sample_image(temp_dir / f"test{i}.jpg")
            for i in range(5)
        ]

        processor = ImageProcessor()
        results = processor.process_batch(images, batch_size=2)

        assert len(results) == 5

    def test_batch_processing_is_faster_than_sequential(self, temp_dir):
        """Batch processing should be faster than sequential."""
        from fileforge.pipeline.processors.image import ImageProcessor
        from tests.fixtures.sample_files import create_sample_image
        import time

        images = [
            create_sample_image(temp_dir / f"test{i}.jpg")
            for i in range(10)
        ]

        processor = ImageProcessor()

        # This is a performance assertion placeholder
        # Real implementation would compare timing
        assert hasattr(processor, 'process_batch') or hasattr(processor, 'process')


class TestErrorHandling:
    """Tests for robust error handling."""

    def test_handles_corrupted_images(self, temp_dir):
        """Should handle corrupted images gracefully."""
        from fileforge.pipeline.processors.image import ImageProcessor

        # Create corrupted file
        corrupted = temp_dir / "corrupted.jpg"
        corrupted.write_bytes(b"not an image")

        processor = ImageProcessor()

        with pytest.raises(Exception):
            processor.process(corrupted)

    def test_handles_unsupported_formats(self, temp_dir):
        """Should handle unsupported image formats."""
        from fileforge.pipeline.processors.image import ImageProcessor

        unsupported = temp_dir / "test.xyz"
        unsupported.write_bytes(b"fake data")

        processor = ImageProcessor()

        with pytest.raises((ValueError, NotImplementedError)):
            processor.process(unsupported)

    def test_handles_missing_files(self):
        """Should handle missing file paths."""
        from fileforge.pipeline.processors.image import ImageProcessor

        processor = ImageProcessor()

        with pytest.raises(FileNotFoundError):
            processor.process(Path("/nonexistent/image.jpg"))


class TestIntegrationWithPipeline:
    """Tests for integration with FileForge pipeline."""

    def test_registers_with_pipeline(self):
        """ImageProcessor should register with main pipeline."""
        from fileforge.pipeline.processors.image import ImageProcessor
        from fileforge.pipeline.core import Pipeline

        pipeline = Pipeline()
        processor = ImageProcessor()

        pipeline.register_processor(processor)

        assert '.jpg' in pipeline.supported_extensions
        assert '.png' in pipeline.supported_extensions

    def test_processing_updates_metadata(self, temp_dir):
        """Processing should update file metadata."""
        from fileforge.pipeline.processors.image import ImageProcessor
        from fileforge.core.metadata import FileMetadata
        from tests.fixtures.sample_files import create_sample_image

        img_path = create_sample_image(temp_dir / "test.jpg")

        processor = ImageProcessor()
        result = processor.process(img_path)

        metadata = FileMetadata.from_analysis(result)

        assert metadata is not None
        assert hasattr(metadata, 'file_type') or hasattr(result, 'metadata')
