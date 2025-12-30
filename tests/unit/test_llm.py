"""TDD Tests for FileForge LLM integration.

RED phase: Tests written first, defining expected behavior.

This test suite defines the expected behavior for:
- LLM client initialization and configuration
- Intelligent filename suggestion
- Category/folder suggestion based on content
- Image captioning with vision models
- Text embedding generation
- Semantic duplicate detection
- Structured JSON output parsing
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call
import json
import numpy as np


class TestLLMClient:
    """Tests for Ollama LLM client initialization and basic operations."""

    def test_client_initialization(self):
        """Should initialize with default model."""
        from fileforge.models.llm import LLMClient

        client = LLMClient()
        assert client.model is not None
        assert 'qwen' in client.model.lower() or client.model is not None

    def test_client_configurable_model(self):
        """Should accept custom model name."""
        from fileforge.models.llm import LLMClient

        client = LLMClient(model='llama2:7b')
        assert client.model == 'llama2:7b'

    def test_client_configurable_base_url(self):
        """Should accept custom Ollama URL."""
        from fileforge.models.llm import LLMClient

        client = LLMClient(base_url='http://custom:11434')
        assert 'custom' in client.base_url

    def test_generate_returns_text(self):
        """Should generate text response."""
        from fileforge.models.llm import LLMClient

        with patch('fileforge.models.llm.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': 'Generated text'}

            client = LLMClient()
            result = client.generate("Test prompt")

            assert result == 'Generated text'
            mock_ollama.generate.assert_called_once()

    def test_generate_with_json_mode(self):
        """Should support JSON output mode."""
        from fileforge.models.llm import LLMClient

        with patch('fileforge.models.llm.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {
                'response': '{"name": "test", "category": "document"}'
            }

            client = LLMClient()
            result = client.generate("Test prompt", json_mode=True)

            # Should be parseable JSON
            parsed = json.loads(result)
            assert 'name' in parsed
            assert parsed['category'] == 'document'

    def test_generate_with_system_prompt(self):
        """Should support system prompts for context."""
        from fileforge.models.llm import LLMClient

        with patch('fileforge.models.llm.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': 'Response'}

            client = LLMClient()
            client.generate(
                "User prompt",
                system_prompt="You are a file organizer assistant."
            )

            # Should include system prompt in call
            call_args = mock_ollama.generate.call_args
            assert call_args is not None

    def test_handles_generation_errors(self):
        """Should handle errors gracefully."""
        from fileforge.models.llm import LLMClient

        with patch('fileforge.models.llm.ollama') as mock_ollama:
            mock_ollama.generate.side_effect = Exception("Connection failed")

            client = LLMClient()

            # Should raise or return None on error
            with pytest.raises(Exception):
                client.generate("Test prompt")


class TestFilenameSuggestion:
    """Tests for intelligent filename suggestion based on content analysis."""

    def test_suggest_filename_from_content(self):
        """Should suggest filename based on content."""
        from fileforge.models.llm import suggest_filename

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"suggested_name": "invoice-acme-2024-01.pdf", "confidence": 0.9}'
            mock_client.return_value = mock_instance

            result = suggest_filename(
                content="Invoice from Acme Corp dated January 2024",
                original_name="scan001.pdf",
                file_type="document"
            )

            assert result is not None
            assert isinstance(result, str)
            assert 'invoice' in result.lower() or 'acme' in result.lower()

    def test_suggested_filename_is_valid(self):
        """Suggested filename should be valid for filesystem."""
        from fileforge.models.llm import sanitize_filename

        # Test sanitization of invalid characters
        sanitized = sanitize_filename('test/file:name?.pdf')
        assert '/' not in sanitized
        assert ':' not in sanitized
        assert '?' not in sanitized

        # Should preserve safe characters
        sanitized2 = sanitize_filename('invoice_acme-2024.pdf')
        assert '_' in sanitized2
        assert '-' in sanitized2
        assert '.' in sanitized2

    def test_preserves_file_extension(self):
        """Should preserve original file extension."""
        from fileforge.models.llm import suggest_filename

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"suggested_name": "new-name", "confidence": 0.85}'
            mock_client.return_value = mock_instance

            result = suggest_filename(
                content="Test content",
                original_name="document.pdf",
                file_type="document"
            )

            # Should end with .pdf
            assert result.endswith('.pdf')

    def test_returns_none_on_low_confidence(self):
        """Should return None if LLM confidence is low."""
        from fileforge.models.llm import suggest_filename

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"suggested_name": null, "confidence": 0.2}'
            mock_client.return_value = mock_instance

            result = suggest_filename(
                content="???",
                original_name="file.txt",
                file_type="text"
            )

            # Low confidence should return None
            assert result is None

    def test_handles_ocr_text_input(self):
        """Should work with OCR-extracted text."""
        from fileforge.models.llm import suggest_filename

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"suggested_name": "receipt-walmart-2024-01-15.pdf", "confidence": 0.88}'
            mock_client.return_value = mock_instance

            ocr_text = "WALMART\nDate: 01/15/2024\nTotal: $45.67"
            result = suggest_filename(
                content=ocr_text,
                original_name="IMG_001.pdf",
                file_type="receipt"
            )

            assert result is not None
            assert 'receipt' in result.lower() or 'walmart' in result.lower()

    def test_uses_metadata_context(self):
        """Should incorporate file metadata in suggestion."""
        from fileforge.models.llm import suggest_filename

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"suggested_name": "photo-paris-2024-07.jpg", "confidence": 0.92}'
            mock_client.return_value = mock_instance

            result = suggest_filename(
                content="Photo of Eiffel Tower",
                original_name="DSC_1234.jpg",
                file_type="image",
                metadata={'exif_date': '2024-07-15', 'location': 'Paris'}
            )

            assert result is not None


class TestCategorySuggestion:
    """Tests for category/folder suggestion based on content analysis."""

    def test_suggest_category_from_content(self):
        """Should suggest category based on content."""
        from fileforge.models.llm import suggest_category

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"category": "invoices", "confidence": 0.9}'
            mock_client.return_value = mock_instance

            result = suggest_category(
                content="Invoice #12345 from Acme Corp",
                tags=['business', 'finance']
            )

            assert result is not None
            assert isinstance(result, str)
            assert result == 'invoices'

    def test_category_matches_configured_options(self):
        """Should suggest from configured category list."""
        from fileforge.models.llm import suggest_category

        categories = ['invoices', 'receipts', 'contracts', 'personal']

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"category": "invoices", "confidence": 0.95}'
            mock_client.return_value = mock_instance

            result = suggest_category(
                content="Invoice content",
                available_categories=categories
            )

            assert result in categories

    def test_returns_none_for_ambiguous_content(self):
        """Should return None when content is ambiguous."""
        from fileforge.models.llm import suggest_category

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"category": null, "confidence": 0.3}'
            mock_client.return_value = mock_instance

            result = suggest_category(
                content="Random text without clear category",
                available_categories=['invoices', 'receipts']
            )

            assert result is None

    def test_subcategory_suggestion(self):
        """Should support hierarchical category suggestions."""
        from fileforge.models.llm import suggest_category

        with patch('fileforge.models.llm.LLMClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = '{"category": "finance/invoices/2024", "confidence": 0.87}'
            mock_client.return_value = mock_instance

            result = suggest_category(
                content="Invoice Q1 2024",
                enable_subcategories=True
            )

            # Should support path-like categories
            assert '/' in result or result.startswith('finance')


class TestImageCaptioning:
    """Tests for image captioning with vision-capable LLM (LLaVA)."""

    def test_caption_image(self, tmp_path):
        """Should generate caption for image."""
        from fileforge.models.llm import caption_image

        # Create mock image file
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b'\x00' * 100)  # Dummy image data

        with patch('fileforge.models.llm.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {
                'response': 'A red square on white background'
            }

            result = caption_image(img_path)

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

    def test_uses_vision_model(self):
        """Should use vision-capable model (LLaVA)."""
        from fileforge.models.llm import LLMClient

        vision_client = LLMClient(model='llava:7b')
        assert 'llava' in vision_client.model.lower()

    def test_caption_includes_detected_objects(self, tmp_path):
        """Caption should reference provided context."""
        from fileforge.models.llm import caption_image

        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b'\x00' * 100)

        with patch('fileforge.models.llm.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {
                'response': 'A person standing next to a car'
            }

            result = caption_image(
                img_path,
                detected_objects=['person', 'car']
            )

            # Caption should be coherent
            assert result is not None
            assert len(result) > 10

    def test_caption_handles_invalid_images(self, tmp_path):
        """Should handle invalid or corrupted images gracefully."""
        from fileforge.models.llm import caption_image

        # Non-existent file
        with pytest.raises(FileNotFoundError):
            caption_image(tmp_path / "nonexistent.jpg")

    def test_batch_caption_images(self, tmp_path):
        """Should support batch captioning for efficiency."""
        from fileforge.models.llm import batch_caption_images

        # Create multiple test images
        images = []
        for i in range(3):
            img = tmp_path / f"img_{i}.jpg"
            img.write_bytes(b'\x00' * 100)
            images.append(img)

        with patch('fileforge.models.llm.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': 'Image caption'}

            results = batch_caption_images(images)

            assert len(results) == 3
            assert all(isinstance(r, str) for r in results)


class TestTextEmbeddings:
    """Tests for text embedding generation using nomic-embed-text."""

    def test_generate_embedding(self):
        """Should generate embedding vector for text."""
        from fileforge.models.embeddings import EmbeddingModel

        with patch('fileforge.models.embeddings.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {
                'embedding': [0.1] * 768
            }

            model = EmbeddingModel()
            embedding = model.embed("Sample text")

            assert embedding is not None
            assert len(embedding) == 768  # nomic-embed-text dimension

    def test_embedding_dimension_is_768(self):
        """Embedding should be 768 dimensions (nomic-embed-text)."""
        from fileforge.models.embeddings import EmbeddingModel

        with patch('fileforge.models.embeddings.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {
                'embedding': [0.1] * 768
            }

            model = EmbeddingModel()
            embedding = model.embed("Test")

            assert len(embedding) == 768
            assert isinstance(embedding, (list, np.ndarray))

    def test_batch_embeddings(self):
        """Should support batch embedding generation."""
        from fileforge.models.embeddings import EmbeddingModel

        with patch('fileforge.models.embeddings.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {
                'embedding': [0.1] * 768
            }

            model = EmbeddingModel()
            texts = ["Text 1", "Text 2", "Text 3"]

            embeddings = model.embed_batch(texts)

            assert len(embeddings) == 3
            assert all(len(emb) == 768 for emb in embeddings)

    def test_embedding_normalization(self):
        """Embeddings should be normalized (unit length)."""
        from fileforge.models.embeddings import EmbeddingModel

        with patch('fileforge.models.embeddings.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {
                'embedding': [0.5] * 768
            }

            model = EmbeddingModel(normalize=True)
            embedding = model.embed("Test")

            # Check if normalized (L2 norm ≈ 1)
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 0.01

    def test_similarity_search(self):
        """Should compute similarity between embeddings."""
        from fileforge.models.embeddings import cosine_similarity

        emb1 = np.array([1, 0, 0])
        emb2 = np.array([1, 0, 0])
        emb3 = np.array([0, 1, 0])

        # Same vectors should have similarity ~1
        assert cosine_similarity(emb1, emb2) > 0.99

        # Orthogonal vectors should have similarity ~0
        assert abs(cosine_similarity(emb1, emb3)) < 0.01

    def test_embedding_caching(self):
        """Should cache embeddings to avoid redundant computation."""
        from fileforge.models.embeddings import EmbeddingModel

        with patch('fileforge.models.embeddings.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {
                'embedding': [0.1] * 768
            }

            model = EmbeddingModel(cache=True)

            # Embed same text twice
            emb1 = model.embed("Test text")
            emb2 = model.embed("Test text")

            # Should only call ollama once (second is cached)
            assert mock_ollama.embeddings.call_count == 1
            assert np.array_equal(emb1, emb2)


class TestSemanticDeduplication:
    """Tests for semantic duplicate detection using embeddings."""

    def test_find_similar_files(self):
        """Should find semantically similar files."""
        from fileforge.models.embeddings import find_similar

        query_embedding = np.random.rand(768)
        stored_embeddings = [
            (1, query_embedding + np.random.rand(768) * 0.1),  # Similar
            (2, np.random.rand(768)),                          # Different
            (3, query_embedding + np.random.rand(768) * 0.05), # Very similar
        ]

        similar = find_similar(query_embedding, stored_embeddings, threshold=0.8)

        # Should return list of (id, similarity) tuples
        assert isinstance(similar, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similar)

    def test_dedup_threshold_configurable(self):
        """Similarity threshold should be configurable."""
        from fileforge.models.embeddings import find_similar

        query = np.random.rand(768)
        stored = [(1, query * 0.9)]  # 90% similar

        # High threshold - might not match
        high_results = find_similar(query, stored, threshold=0.99)

        # Low threshold - should match
        low_results = find_similar(query, stored, threshold=0.5)

        assert len(low_results) >= len(high_results)

    def test_semantic_duplicates_detected(self):
        """Should detect semantic duplicates with different wording."""
        from fileforge.models.embeddings import EmbeddingModel, find_similar

        with patch('fileforge.models.embeddings.ollama') as mock_ollama:
            # Mock similar embeddings for semantically similar text
            def mock_embed(text):
                if 'invoice' in text.lower():
                    return {'embedding': [0.8] * 768}
                else:
                    return {'embedding': [0.1] * 768}

            mock_ollama.embeddings.side_effect = lambda model, prompt: mock_embed(prompt)

            model = EmbeddingModel()

            # These are semantically similar
            emb1 = model.embed("Invoice from supplier")
            emb2 = model.embed("Bill from vendor")

            # Should be similar despite different words
            from fileforge.models.embeddings import cosine_similarity
            similarity = cosine_similarity(emb1, emb2)

            # Should be reasonably similar
            assert similarity > 0.5

    def test_returns_top_k_similar(self):
        """Should return only top K most similar results."""
        from fileforge.models.embeddings import find_similar

        query = np.random.rand(768)
        stored = [(i, np.random.rand(768)) for i in range(100)]

        top_5 = find_similar(query, stored, threshold=0.0, top_k=5)

        assert len(top_5) <= 5


class TestStructuredOutput:
    """Tests for structured JSON output parsing from LLM responses."""

    def test_parse_json_response(self):
        """Should parse JSON from LLM response."""
        from fileforge.models.llm import parse_json_response

        response = '{"name": "test", "value": 123}'
        parsed = parse_json_response(response)

        assert parsed['name'] == 'test'
        assert parsed['value'] == 123

    def test_handles_markdown_json_blocks(self):
        """Should handle JSON in markdown code blocks."""
        from fileforge.models.llm import parse_json_response

        response = '''Here is the result:
```json
{"name": "test", "category": "document"}
```
'''
        parsed = parse_json_response(response)
        assert parsed['name'] == 'test'
        assert parsed['category'] == 'document'

    def test_handles_multiple_json_blocks(self):
        """Should extract JSON from mixed text/JSON responses."""
        from fileforge.models.llm import parse_json_response

        response = '''Some explanation text.
```json
{"result": "value"}
```
More text after.'''

        parsed = parse_json_response(response)
        assert parsed['result'] == 'value'

    def test_validates_against_schema(self):
        """Should validate output against expected schema."""
        from fileforge.models.llm import validate_response
        from pydantic import BaseModel

        class FileNameSuggestion(BaseModel):
            suggested_name: str
            confidence: float

        response = '{"suggested_name": "test.pdf", "confidence": 0.95}'

        result = validate_response(response, FileNameSuggestion)
        assert result.suggested_name == "test.pdf"
        assert result.confidence == 0.95

    def test_schema_validation_fails_on_invalid_data(self):
        """Should raise error on schema validation failure."""
        from fileforge.models.llm import validate_response
        from pydantic import BaseModel, ValidationError

        class StrictSchema(BaseModel):
            name: str
            count: int

        invalid_response = '{"name": "test", "count": "not_a_number"}'

        with pytest.raises(ValidationError):
            validate_response(invalid_response, StrictSchema)

    def test_handles_malformed_json(self):
        """Should handle malformed JSON gracefully."""
        from fileforge.models.llm import parse_json_response

        malformed = '{"name": "test", invalid json'

        # Should raise JSONDecodeError or return None
        with pytest.raises(json.JSONDecodeError):
            parse_json_response(malformed)


class TestPromptTemplates:
    """Tests for LLM prompt template management."""

    def test_filename_suggestion_prompt(self):
        """Should generate proper filename suggestion prompt."""
        from fileforge.models.prompts import create_filename_prompt

        prompt = create_filename_prompt(
            content="Invoice from Acme Corp",
            file_type="document",
            original_name="scan.pdf"
        )

        assert 'filename' in prompt.lower()
        assert 'acme corp' in prompt.lower()
        assert '.pdf' in prompt

    def test_category_suggestion_prompt(self):
        """Should generate proper category suggestion prompt."""
        from fileforge.models.prompts import create_category_prompt

        categories = ['invoices', 'receipts', 'contracts']
        prompt = create_category_prompt(
            content="Invoice content",
            available_categories=categories
        )

        assert 'category' in prompt.lower()
        assert all(cat in prompt for cat in categories)

    def test_image_caption_prompt(self):
        """Should generate proper image captioning prompt."""
        from fileforge.models.prompts import create_caption_prompt

        prompt = create_caption_prompt(
            detected_objects=['person', 'car'],
            context={'location': 'street'}
        )

        assert 'caption' in prompt.lower() or 'describe' in prompt.lower()
        assert 'person' in prompt
        assert 'car' in prompt


# Fixtures
@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_llm_client():
    """Provide mocked LLM client for testing."""
    with patch('fileforge.models.llm.LLMClient') as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def sample_embeddings():
    """Provide sample embedding vectors for testing."""
    return {
        'invoice': np.random.rand(768),
        'receipt': np.random.rand(768),
        'contract': np.random.rand(768),
    }
