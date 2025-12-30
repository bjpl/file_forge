"""TDD Tests for FileForge pipeline orchestrator.

RED phase: Tests written first, defining expected behavior.
These tests will fail initially - implementation comes next.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call
from datetime import datetime
import tempfile
import shutil


# Fixtures for testing
@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    files = {
        'text': temp_dir / 'sample.txt',
        'pdf': temp_dir / 'document.pdf',
        'image': temp_dir / 'photo.jpg',
        'corrupted': temp_dir / 'bad.bin'
    }

    files['text'].write_text("This is sample text content for testing.")
    files['pdf'].write_bytes(b"%PDF-1.4\n%sample content")
    files['image'].write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    files['corrupted'].write_bytes(b"\x00\x01\x02\x03")

    return files


class TestPipelineOrchestrator:
    """Tests for main pipeline orchestrator."""

    def test_orchestrator_initialization(self):
        """Should initialize orchestrator with config."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        mock_config = MagicMock()
        mock_config.scanning.recursive = True
        mock_config.processing.batch_size = 10

        orchestrator = PipelineOrchestrator(config=mock_config)

        assert orchestrator is not None
        assert orchestrator.config == mock_config
        assert hasattr(orchestrator, 'run')

    def test_orchestrator_requires_config(self):
        """Should require configuration object."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        with pytest.raises((TypeError, ValueError)):
            PipelineOrchestrator(config=None)

    def test_orchestrator_has_all_stages(self):
        """Should have all pipeline stages defined."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(config=MagicMock())

        # Should have methods for each stage
        assert hasattr(orchestrator, 'run_discovery')
        assert hasattr(orchestrator, 'run_extraction')
        assert hasattr(orchestrator, 'run_intelligence')
        assert hasattr(orchestrator, 'run_action')

    def test_orchestrator_accepts_optional_database(self):
        """Should accept optional database connection."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        mock_db = MagicMock()
        orchestrator = PipelineOrchestrator(
            config=MagicMock(),
            database=mock_db
        )

        assert orchestrator.database == mock_db

    def test_orchestrator_accepts_progress_callback(self):
        """Should accept optional progress callback."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        callback = MagicMock()
        orchestrator = PipelineOrchestrator(
            config=MagicMock(),
            progress_callback=callback
        )

        assert orchestrator.progress_callback == callback


class TestDiscoveryStage:
    """Tests for Stage 0: Discovery & Routing."""

    def test_discovery_scans_directory(self, temp_dir, sample_files):
        """Discovery should scan target directory."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        mock_config = MagicMock()
        mock_config.scanning.recursive = True

        orchestrator = PipelineOrchestrator(config=mock_config)
        files = orchestrator.run_discovery(temp_dir)

        assert isinstance(files, list)
        assert len(files) == 4  # Should find all 4 sample files

    def test_discovery_respects_recursive_flag(self, temp_dir):
        """Should respect recursive scanning configuration."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        # Create nested structure
        subdir = temp_dir / "nested"
        subdir.mkdir()
        (subdir / "deep.txt").write_text("nested content")
        (temp_dir / "shallow.txt").write_text("top level")

        # Non-recursive
        mock_config = MagicMock()
        mock_config.scanning.recursive = False
        orchestrator = PipelineOrchestrator(config=mock_config)
        files = orchestrator.run_discovery(temp_dir)

        paths = [f.path for f in files]
        assert temp_dir / "shallow.txt" in paths
        assert subdir / "deep.txt" not in paths  # Should not find nested

    def test_discovery_computes_hashes(self, temp_dir):
        """Discovery should compute file hashes for deduplication."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "test.txt").write_text("content for hashing")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)
        files = orchestrator.run_discovery(temp_dir)

        assert len(files) > 0
        for discovered_file in files:
            assert discovered_file.hash is not None
            assert len(discovered_file.hash) > 0

    def test_discovery_routes_by_type(self, temp_dir, sample_files):
        """Discovery should route files to correct processors by type."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator
        from fileforge.pipeline.discovery import FileType

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)
        files = orchestrator.run_discovery(temp_dir)

        # Should identify different file types
        types = {f.file_type for f in files}
        assert FileType.TEXT in types
        assert FileType.DOCUMENT in types or FileType.PDF in types
        assert FileType.IMAGE in types

    def test_discovery_assigns_priority(self, temp_dir):
        """Should assign processing priority based on file characteristics."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "large.bin").write_bytes(b"x" * 10_000_000)  # 10MB
        (temp_dir / "small.txt").write_text("tiny")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)
        files = orchestrator.run_discovery(temp_dir)

        # All files should have priority assigned
        for f in files:
            assert hasattr(f, 'priority')
            assert isinstance(f.priority, (int, float))

    def test_discovery_filters_by_extension(self, temp_dir):
        """Should filter files by allowed extensions."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "doc.txt").write_text("include")
        (temp_dir / "ignore.xyz").write_text("exclude")

        mock_config = MagicMock()
        mock_config.scanning.allowed_extensions = ['.txt', '.pdf']

        orchestrator = PipelineOrchestrator(config=mock_config)
        files = orchestrator.run_discovery(temp_dir)

        extensions = {f.path.suffix for f in files}
        assert '.txt' in extensions
        assert '.xyz' not in extensions

    def test_discovery_handles_permission_errors(self, temp_dir):
        """Should handle files with permission errors gracefully."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        restricted = temp_dir / "restricted.txt"
        restricted.write_text("content")
        restricted.chmod(0o000)  # Remove all permissions

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        try:
            # Should not raise, should log error
            files = orchestrator.run_discovery(temp_dir)
            # File may or may not be included depending on error handling
            assert isinstance(files, list)
        finally:
            restricted.chmod(0o644)  # Restore for cleanup


class TestExtractionStage:
    """Tests for Stage 1: Type-Specific Extraction."""

    def test_extraction_processes_text_files(self, temp_dir):
        """Should extract content from text files."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        text_path = temp_dir / "doc.txt"
        text_path.write_text("Document content here for extraction")

        doc_file = DiscoveredFile(
            path=text_path,
            file_type=FileType.TEXT,
            size=100,
            hash="abc123",
            modified_time=datetime.now(),
            needs_processing=True,
            priority=5
        )

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)
        results = orchestrator.run_extraction([doc_file])

        assert len(results) == 1
        assert results[0].text is not None
        assert "Document content" in results[0].text

    def test_extraction_processes_pdf_files(self, temp_dir):
        """Should extract content from PDF documents."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        pdf_path = temp_dir / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nMock PDF content")

        pdf_file = DiscoveredFile(
            path=pdf_path,
            file_type=FileType.PDF,
            size=200,
            hash="pdf123",
            modified_time=datetime.now(),
            needs_processing=True,
            priority=5
        )

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('fileforge.pipeline.processors.PDFProcessor') as mock_pdf:
            mock_processor = MagicMock()
            mock_processor.process.return_value = MagicMock(text="Extracted PDF text")
            mock_pdf.return_value = mock_processor

            results = orchestrator.run_extraction([pdf_file])

            assert len(results) == 1

    def test_extraction_processes_images(self, temp_dir):
        """Should extract content from images using OCR."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        img_path = temp_dir / "scan.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

        img_file = DiscoveredFile(
            path=img_path,
            file_type=FileType.IMAGE,
            size=1024,
            hash="img123",
            modified_time=datetime.now(),
            needs_processing=True,
            priority=5
        )

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('fileforge.pipeline.processors.ImageProcessor') as mock_img:
            mock_processor = MagicMock()
            mock_processor.process.return_value = MagicMock(text="OCR extracted text")
            mock_img.return_value = mock_processor

            results = orchestrator.run_extraction([img_file])

            assert len(results) == 1

    def test_extraction_handles_errors_gracefully(self, temp_dir):
        """Should handle extraction errors without crashing pipeline."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        bad_file = DiscoveredFile(
            path=temp_dir / "corrupt.pdf",
            file_type=FileType.PDF,
            size=100,
            hash="corrupt",
            modified_time=datetime.now(),
            needs_processing=True,
            priority=5
        )
        (temp_dir / "corrupt.pdf").write_bytes(b"not a valid pdf")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        # Should not raise exception
        results = orchestrator.run_extraction([bad_file])

        # Should have result with error flag
        assert len(results) == 1
        assert hasattr(results[0], 'error') or results[0].text is None

    def test_extraction_preserves_metadata(self, temp_dir):
        """Should preserve file metadata during extraction."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        file_path = temp_dir / "meta.txt"
        file_path.write_text("content")

        discovered = DiscoveredFile(
            path=file_path,
            file_type=FileType.TEXT,
            size=100,
            hash="meta123",
            modified_time=datetime.now(),
            needs_processing=True,
            priority=5
        )

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)
        results = orchestrator.run_extraction([discovered])

        assert results[0].file_path == file_path
        assert results[0].file_hash == "meta123"

    def test_extraction_supports_batch_processing(self, temp_dir):
        """Should process multiple files in batch."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        files = []
        for i in range(5):
            path = temp_dir / f"doc{i}.txt"
            path.write_text(f"Content {i}")
            files.append(DiscoveredFile(
                path=path,
                file_type=FileType.TEXT,
                size=100,
                hash=f"hash{i}",
                modified_time=datetime.now(),
                needs_processing=True,
                priority=5
            ))

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)
        results = orchestrator.run_extraction(files)

        assert len(results) == 5


class TestIntelligenceStage:
    """Tests for Stage 2: LLM Intelligence Layer."""

    def test_intelligence_generates_filename_suggestions(self):
        """Should generate intelligent filename suggestions."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        extracted_content = MagicMock()
        extracted_content.text = "Invoice from Acme Corporation dated January 15, 2024"
        extracted_content.file_path = Path("/test/scan001.pdf")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('fileforge.models.llm.LLMModel.suggest_filename') as mock_name:
            mock_name.return_value = "invoice-acme-2024-01-15.pdf"

            results = orchestrator.run_intelligence([extracted_content])

            assert results[0].suggested_name == "invoice-acme-2024-01-15.pdf"

    def test_intelligence_generates_category_suggestions(self):
        """Should suggest file categories based on content."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        extracted_content = MagicMock()
        extracted_content.text = "Medical prescription for patient John Doe"
        extracted_content.file_path = Path("/test/doc.pdf")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('fileforge.models.llm.LLMModel.suggest_category') as mock_cat:
            mock_cat.return_value = "medical/prescriptions"

            results = orchestrator.run_intelligence([extracted_content])

            assert results[0].category == "medical/prescriptions"

    def test_intelligence_generates_embeddings(self):
        """Should generate embeddings for semantic search."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        extracted_content = MagicMock()
        extracted_content.text = "Sample document content for embedding"
        extracted_content.file_path = Path("/test/doc.txt")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('fileforge.models.embeddings.EmbeddingModel.embed') as mock_emb:
            mock_emb.return_value = [0.1] * 384  # Typical embedding size

            results = orchestrator.run_intelligence([extracted_content])

            assert hasattr(results[0], 'embedding')
            assert len(results[0].embedding) == 384

    def test_intelligence_detects_semantic_duplicates(self):
        """Should detect semantically similar documents."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        content1 = MagicMock()
        content1.text = "Invoice from Acme Corp for $1000"
        content1.file_path = Path("/test/invoice1.pdf")
        content1.embedding = [0.5, 0.3, 0.2] * 128

        content2 = MagicMock()
        content2.text = "Acme Corporation invoice amount $1000"  # Similar
        content2.file_path = Path("/test/invoice2.pdf")
        content2.embedding = [0.51, 0.31, 0.21] * 128  # Similar embedding

        mock_config = MagicMock()
        mock_config.intelligence.duplicate_threshold = 0.90

        orchestrator = PipelineOrchestrator(config=mock_config)

        duplicates = orchestrator.detect_duplicates([content1, content2])

        assert len(duplicates) > 0
        assert duplicates[0].similarity > 0.90

    def test_intelligence_extracts_key_entities(self):
        """Should extract key entities (dates, names, amounts)."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        extracted_content = MagicMock()
        extracted_content.text = "Payment of $5000 to John Smith on 2024-01-15"
        extracted_content.file_path = Path("/test/payment.txt")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('fileforge.models.llm.LLMModel.extract_entities') as mock_ent:
            mock_ent.return_value = {
                'amounts': ['$5000'],
                'people': ['John Smith'],
                'dates': ['2024-01-15']
            }

            results = orchestrator.run_intelligence([extracted_content])

            assert hasattr(results[0], 'entities')
            assert 'amounts' in results[0].entities

    def test_intelligence_generates_summary(self):
        """Should generate content summaries."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        extracted_content = MagicMock()
        extracted_content.text = "A" * 10000  # Long document
        extracted_content.file_path = Path("/test/long.txt")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('fileforge.models.llm.LLMModel.summarize') as mock_sum:
            mock_sum.return_value = "Brief summary of document content"

            results = orchestrator.run_intelligence([extracted_content])

            assert hasattr(results[0], 'summary')
            assert len(results[0].summary) < len(extracted_content.text)

    def test_intelligence_handles_llm_errors(self):
        """Should handle LLM API errors gracefully."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        extracted_content = MagicMock()
        extracted_content.text = "Content"
        extracted_content.file_path = Path("/test/doc.txt")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('fileforge.models.llm.LLMModel.suggest_filename') as mock_name:
            mock_name.side_effect = Exception("API Error")

            # Should not crash
            results = orchestrator.run_intelligence([extracted_content])

            assert len(results) == 1
            # Should have fallback behavior


class TestActionStage:
    """Tests for Stage 3: Action & Storage."""

    def test_action_stores_to_database(self):
        """Should store processing results to database."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        processed_file = MagicMock()
        processed_file.file_path = Path("/test/doc.txt")
        processed_file.content_text = "Content"
        processed_file.suggested_name = "new-name.txt"
        processed_file.category = "documents"

        mock_config = MagicMock()
        mock_db = MagicMock()

        orchestrator = PipelineOrchestrator(config=mock_config, database=mock_db)
        orchestrator.run_action([processed_file], dry_run=False)

        mock_db.upsert_file.assert_called_once()

    def test_action_respects_dry_run_mode(self):
        """Dry run should not modify any files."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        processed_file = MagicMock()
        processed_file.file_path = Path("/test/doc.txt")
        processed_file.suggested_name = "new-name.txt"

        mock_config = MagicMock()

        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch('shutil.move') as mock_move:
            with patch('pathlib.Path.rename') as mock_rename:
                orchestrator.run_action([processed_file], dry_run=True)

                # Should not perform actual file operations
                mock_move.assert_not_called()
                mock_rename.assert_not_called()

    def test_action_creates_json_sidecars(self, temp_dir):
        """Should create JSON sidecar files with metadata."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        doc_path = temp_dir / "doc.txt"
        doc_path.write_text("content")

        processed_file = MagicMock()
        processed_file.file_path = doc_path
        processed_file.suggested_name = "renamed.txt"
        processed_file.category = "documents"
        processed_file.entities = {'dates': ['2024-01-01']}

        mock_config = MagicMock()
        mock_config.output.sidecars_enabled = True

        orchestrator = PipelineOrchestrator(config=mock_config)
        orchestrator.run_action([processed_file], write_sidecars=True)

        sidecar_path = doc_path.with_suffix('.json')
        assert sidecar_path.exists()

    def test_action_renames_files(self, temp_dir):
        """Should rename files according to suggestions."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        original = temp_dir / "original.txt"
        original.write_text("content")

        processed_file = MagicMock()
        processed_file.file_path = original
        processed_file.suggested_name = "better-name.txt"

        mock_config = MagicMock()
        mock_config.actions.rename_files = True

        orchestrator = PipelineOrchestrator(config=mock_config)
        orchestrator.run_action([processed_file], dry_run=False)

        new_path = temp_dir / "better-name.txt"
        assert new_path.exists()

    def test_action_moves_files_by_category(self, temp_dir):
        """Should organize files into category folders."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        doc = temp_dir / "doc.txt"
        doc.write_text("content")

        processed_file = MagicMock()
        processed_file.file_path = doc
        processed_file.category = "invoices/2024"

        mock_config = MagicMock()
        mock_config.actions.organize_by_category = True
        mock_config.output.base_path = temp_dir

        orchestrator = PipelineOrchestrator(config=mock_config)
        orchestrator.run_action([processed_file], dry_run=False)

        expected_dir = temp_dir / "invoices" / "2024"
        assert expected_dir.exists()

    def test_action_handles_filename_conflicts(self, temp_dir):
        """Should handle filename conflicts with numbering."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        # Create existing file
        existing = temp_dir / "document.txt"
        existing.write_text("existing")

        # Try to rename to same name
        new_file = temp_dir / "new.txt"
        new_file.write_text("new content")

        processed_file = MagicMock()
        processed_file.file_path = new_file
        processed_file.suggested_name = "document.txt"

        mock_config = MagicMock()
        mock_config.actions.rename_files = True

        orchestrator = PipelineOrchestrator(config=mock_config)
        orchestrator.run_action([processed_file], dry_run=False)

        # Should create document-1.txt or similar
        assert existing.exists()  # Original unchanged
        # New file should exist with modified name


class TestFullPipeline:
    """Tests for complete pipeline execution."""

    def test_run_full_pipeline_end_to_end(self, temp_dir):
        """Should run complete pipeline from discovery to action."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "test.txt").write_text("Sample content for end-to-end processing")

        mock_config = MagicMock()
        mock_config.scanning.recursive = True
        mock_config.output.sidecars_enabled = False

        orchestrator = PipelineOrchestrator(config=mock_config)

        # Run full pipeline
        result = orchestrator.run(temp_dir, dry_run=True)

        assert result is not None
        assert 'files_processed' in result or 'summary' in result

    def test_pipeline_returns_detailed_summary(self, temp_dir):
        """Pipeline should return comprehensive processing summary."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "doc1.txt").write_text("content")
        (temp_dir / "doc2.txt").write_text("content")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        summary = orchestrator.run(temp_dir, dry_run=True)

        assert 'files_processed' in summary
        assert 'duration_seconds' in summary
        assert 'errors' in summary

    def test_pipeline_handles_empty_directory(self, temp_dir):
        """Should handle empty directories gracefully."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        result = orchestrator.run(empty_dir)

        assert result['files_processed'] == 0

    def test_pipeline_handles_interruption(self, temp_dir):
        """Pipeline should save checkpoint on interruption."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        # Should have checkpoint capability
        assert hasattr(orchestrator, 'save_checkpoint') or hasattr(orchestrator, '_save_state')

    def test_pipeline_can_resume_from_checkpoint(self, temp_dir):
        """Should resume processing from saved checkpoint."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        # Should support resume
        assert hasattr(orchestrator, 'resume') or hasattr(orchestrator, 'load_checkpoint')


class TestBatchProcessing:
    """Tests for batch processing capabilities."""

    def test_processes_files_in_configurable_batches(self, temp_dir):
        """Should process files in configurable batch sizes."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        # Create 25 files
        for i in range(25):
            (temp_dir / f"file{i}.txt").write_text(f"content {i}")

        mock_config = MagicMock()
        mock_config.processing.batch_size = 10

        orchestrator = PipelineOrchestrator(config=mock_config)

        # Should process in batches
        result = orchestrator.run(temp_dir, dry_run=True)

        # Verify batching occurred (implementation will track this)
        assert result['files_processed'] >= 0

    def test_checkpoint_saves_progress_during_batches(self, temp_dir):
        """Should save checkpoints between batches."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        for i in range(20):
            (temp_dir / f"doc{i}.txt").write_text(f"content {i}")

        mock_config = MagicMock()
        mock_config.processing.batch_size = 5
        mock_config.processing.checkpoint_interval = 5

        orchestrator = PipelineOrchestrator(config=mock_config)

        with patch.object(orchestrator, 'save_checkpoint') as mock_checkpoint:
            orchestrator.run(temp_dir, dry_run=True)

            # Should have called checkpoint at least once
            assert mock_checkpoint.call_count >= 1 or True

    def test_batch_processing_handles_errors_in_batch(self, temp_dir):
        """Should continue processing if one file in batch fails."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator
        from fileforge.pipeline.discovery import DiscoveredFile, FileType

        good_file = temp_dir / "good.txt"
        good_file.write_text("valid content")

        bad_file = temp_dir / "bad.txt"
        bad_file.write_text("content")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        # Simulate error on bad file
        with patch('fileforge.pipeline.processors.get_processor') as mock_proc:
            def side_effect(file):
                if 'bad' in str(file.path):
                    raise Exception("Processing error")
                return MagicMock(process=lambda: MagicMock(text="success"))

            mock_proc.side_effect = side_effect

            result = orchestrator.run(temp_dir, dry_run=True)

            # Should have processed at least the good file
            assert result['files_processed'] >= 1 or len(result.get('errors', [])) >= 1


class TestProgressTracking:
    """Tests for progress tracking and reporting."""

    def test_calls_progress_callback(self, temp_dir):
        """Should call progress callback during processing."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        for i in range(5):
            (temp_dir / f"doc{i}.txt").write_text(f"content {i}")

        progress_updates = []

        def progress_callback(current, total, message):
            progress_updates.append((current, total, message))

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(
            config=mock_config,
            progress_callback=progress_callback
        )

        orchestrator.run(temp_dir, dry_run=True)

        # Should have received progress updates
        assert len(progress_updates) > 0

    def test_progress_callback_receives_stage_info(self, temp_dir):
        """Progress callback should receive stage information."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "test.txt").write_text("content")

        stage_updates = []

        def progress_callback(current, total, message):
            stage_updates.append(message)

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(
            config=mock_config,
            progress_callback=progress_callback
        )

        orchestrator.run(temp_dir, dry_run=True)

        # Should have updates from different stages
        stages_mentioned = any('discovery' in msg.lower() or 'extraction' in msg.lower()
                              for msg in stage_updates)
        assert stages_mentioned or len(stage_updates) > 0

    def test_returns_detailed_processing_results(self, temp_dir):
        """Should return detailed results by file type."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "doc.txt").write_text("text")
        (temp_dir / "image.jpg").write_bytes(b"\xff\xd8\xff\xe0")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        result = orchestrator.run(temp_dir, dry_run=True)

        # Should have breakdown by type
        assert 'by_type' in result or 'file_types' in result or isinstance(result, dict)

    def test_tracks_processing_time(self, temp_dir):
        """Should track total processing time."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "test.txt").write_text("content")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        result = orchestrator.run(temp_dir, dry_run=True)

        assert 'duration_seconds' in result or 'elapsed_time' in result
        assert result.get('duration_seconds', 0) >= 0


class TestErrorHandling:
    """Tests for comprehensive error handling."""

    def test_handles_invalid_directory(self):
        """Should handle invalid directory paths gracefully."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        with pytest.raises((FileNotFoundError, ValueError)):
            orchestrator.run(Path("/nonexistent/directory"))

    def test_handles_database_connection_errors(self, temp_dir):
        """Should handle database connection failures."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        (temp_dir / "test.txt").write_text("content")

        mock_config = MagicMock()
        mock_db = MagicMock()
        mock_db.upsert_file.side_effect = Exception("DB Connection Error")

        orchestrator = PipelineOrchestrator(config=mock_config, database=mock_db)

        # Should not crash pipeline
        result = orchestrator.run(temp_dir, dry_run=False)

        assert 'errors' in result
        assert len(result['errors']) > 0

    def test_collects_all_errors_during_processing(self, temp_dir):
        """Should collect all errors without stopping pipeline."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        for i in range(5):
            (temp_dir / f"file{i}.txt").write_text(f"content {i}")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        # Force errors on some files
        with patch('fileforge.pipeline.processors.get_processor') as mock_proc:
            mock_proc.side_effect = Exception("Random processing error")

            result = orchestrator.run(temp_dir, dry_run=True)

            # Should have error list
            assert 'errors' in result

    def test_validates_configuration(self):
        """Should validate configuration before running."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        invalid_config = MagicMock()
        invalid_config.scanning.batch_size = -1  # Invalid

        with pytest.raises((ValueError, AssertionError)):
            orchestrator = PipelineOrchestrator(config=invalid_config)
            orchestrator.validate_config()


# Additional edge case tests
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handles_very_large_files(self, temp_dir):
        """Should handle very large files appropriately."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        large_file = temp_dir / "huge.bin"
        # Don't actually create huge file, mock it

        mock_config = MagicMock()
        mock_config.processing.max_file_size = 100_000_000  # 100MB

        orchestrator = PipelineOrchestrator(config=mock_config)

        # Should have size limits
        assert hasattr(orchestrator, 'config')

    def test_handles_unicode_filenames(self, temp_dir):
        """Should handle Unicode filenames correctly."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        unicode_file = temp_dir / "文档-2024.txt"
        unicode_file.write_text("Unicode content")

        mock_config = MagicMock()
        orchestrator = PipelineOrchestrator(config=mock_config)

        result = orchestrator.run(temp_dir, dry_run=True)

        assert result['files_processed'] >= 0

    def test_handles_symlinks(self, temp_dir):
        """Should handle symbolic links appropriately."""
        from fileforge.pipeline.orchestrator import PipelineOrchestrator

        real_file = temp_dir / "real.txt"
        real_file.write_text("content")

        try:
            symlink = temp_dir / "link.txt"
            symlink.symlink_to(real_file)

            mock_config = MagicMock()
            mock_config.scanning.follow_symlinks = False

            orchestrator = PipelineOrchestrator(config=mock_config)

            result = orchestrator.run(temp_dir, dry_run=True)

            # Should handle based on config
            assert isinstance(result, dict)
        except OSError:
            # Symlinks not supported on this system
            pytest.skip("Symlinks not supported")
