# TDD Status: FileForge Processors

**Phase:** RED (Tests First) ✅
**Date:** 2025-12-29
**Test File:** `/mnt/c/Users/brand/Development/Project_Workspace/active-development/file_forge/tests/unit/test_processors.py`

## Summary

Created comprehensive TDD test suite for FileForge document and text processors following strict Test-Driven Development methodology. All 29 tests are currently failing as expected in RED phase.

## Test Coverage

### TestExtractedContent (2 tests)
- ✅ `test_extracted_content_creation` - Validates dataclass creation
- ✅ `test_extracted_content_defaults` - Checks default field values

### TestDocumentProcessor (12 tests)
- ✅ `test_supported_extensions` - PDF, DOCX, DOC support
- ✅ `test_process_pdf_extracts_text` - PDF text extraction
- ✅ `test_process_pdf_detects_scanned_pages` - OCR detection
- ✅ `test_process_docx_extracts_paragraphs` - DOCX paragraph parsing
- ✅ `test_process_docx_extracts_headings` - DOCX heading structure
- ✅ `test_process_docx_extracts_metadata` - Author, title metadata
- ✅ `test_process_returns_confidence_score` - Extraction quality score
- ✅ `test_handles_corrupt_file_gracefully` - Error handling
- ✅ `test_process_extracts_embedded_images` - Image extraction
- ✅ `test_process_handles_tables` - Table structure parsing
- ✅ `test_batch_process_multiple_documents` - Batch processing
- ✅ `test_process_mixed_document` - Complex document handling

### TestTextProcessor (10 tests)
- ✅ `test_supported_extensions` - TXT, MD, Markdown support
- ✅ `test_process_txt_reads_content` - Plain text reading
- ✅ `test_process_markdown_extracts_structure` - Markdown headings
- ✅ `test_process_markdown_extracts_code_blocks` - Code fence detection
- ✅ `test_handles_different_encodings` - UTF-8, special chars
- ✅ `test_extracts_frontmatter_yaml` - YAML metadata parsing
- ✅ `test_identifies_document_type` - README, notes classification
- ✅ `test_confidence_based_on_content_quality` - Quality scoring
- ✅ `test_extracts_links_from_markdown` - Link extraction
- ✅ `test_handles_very_large_files` - Memory-efficient processing

### TestProcessorRegistry (3 tests)
- ✅ `test_get_processor_for_extension` - Extension routing
- ✅ `test_unknown_extension_returns_none` - Unknown file handling
- ✅ `test_register_custom_processor` - Custom processor registration

### TestProcessingError (2 tests)
- ✅ `test_processing_error_creation` - Exception creation
- ✅ `test_processing_error_with_cause` - Exception chaining

## Test Infrastructure

### Fixtures Created
- `/tests/conftest.py` - Pytest configuration, markers, shared fixtures
- `/tests/fixtures/sample_files.py` - Test file generators:
  - `create_sample_pdf()` - PDF file creation
  - `create_sample_docx()` - DOCX file creation
  - `create_sample_markdown()` - Markdown file creation
  - `create_sample_text()` - Plain text file creation

### Test Markers
- `slow` - Marks slow-running tests
- `integration` - Integration test marker
- `requires_ocr` - OCR dependency marker

## Current Status

```bash
# Run tests (all should fail in RED phase)
python3 -m pytest tests/unit/test_processors.py -v

# Expected output:
# 29 tests collected
# 29 FAILED - All failing with ModuleNotFoundError (expected in RED phase)
```

## Implementation Requirements

### DocumentProcessor
```python
class DocumentProcessor:
    supported_extensions = ['.pdf', '.docx', '.doc']

    def process(self, path: Path) -> ExtractedContent:
        """Extract content from document."""

    def _extract_pdf_text(self, path: Path) -> list:
        """Extract text from PDF pages."""

    def _ocr_page(self, page_image) -> str:
        """Perform OCR on scanned page."""

    def _extract_images(self, path: Path) -> list:
        """Extract embedded images."""

    def batch_process(self, paths: list[Path]) -> list[ExtractedContent]:
        """Process multiple documents efficiently."""
```

### TextProcessor
```python
class TextProcessor:
    supported_extensions = ['.txt', '.md', '.markdown']

    def process(self, path: Path) -> ExtractedContent:
        """Extract content from text file."""

    def _parse_markdown_structure(self, content: str) -> dict:
        """Extract Markdown structure (headings, code, links)."""

    def _extract_frontmatter(self, content: str) -> dict:
        """Parse YAML frontmatter."""

    def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding."""
```

### ExtractedContent Dataclass
```python
@dataclass
class ExtractedContent:
    text: str
    structure: dict
    metadata: dict
    pages: list
    embedded_images: list
    confidence: float
```

### Processor Registry
```python
def get_processor(extension: str) -> Optional[BaseProcessor]:
    """Get processor for file extension."""

def register_processor(extension: str, processor_class: Type):
    """Register custom processor."""
```

### ProcessingError Exception
```python
class ProcessingError(Exception):
    """Exception for processing errors."""
```

## Dependencies Needed

```toml
[tool.poetry.dependencies]
# PDF processing
pypdf2 = "^3.0.0"
pdfplumber = "^0.11.0"
pytesseract = "^0.3.10"  # OCR

# DOCX processing
python-docx = "^1.1.0"

# Markdown processing
markdown-it-py = "^3.0.0"
pyyaml = "^6.0.1"

# Text encoding
chardet = "^5.2.0"
```

## Next Steps (GREEN Phase)

1. **Create module structure:**
   - `/mnt/c/Users/brand/Development/Project_Workspace/active-development/file_forge/src/fileforge/pipeline/processors/__init__.py`
   - `/mnt/c/Users/brand/Development/Project_Workspace/active-development/file_forge/src/fileforge/pipeline/processors/document.py`
   - `/mnt/c/Users/brand/Development/Project_Workspace/active-development/file_forge/src/fileforge/pipeline/processors/text.py`

2. **Implement minimum viable processors:**
   - ExtractedContent dataclass
   - DocumentProcessor with basic PDF/DOCX support
   - TextProcessor with TXT/Markdown support
   - Processor registry

3. **Run tests incrementally:**
   - Start with simplest tests (dataclass creation)
   - Progress to file reading
   - Add structure extraction
   - Implement advanced features (OCR, tables, etc.)

4. **REFACTOR phase:**
   - Optimize performance
   - Improve error handling
   - Add logging
   - Enhance confidence scoring

## Test Execution

```bash
# Run all processor tests
pytest tests/unit/test_processors.py -v

# Run specific test class
pytest tests/unit/test_processors.py::TestDocumentProcessor -v

# Run with coverage
pytest tests/unit/test_processors.py --cov=fileforge.pipeline.processors

# Skip slow tests
pytest tests/unit/test_processors.py -m "not slow"
```

## Notes

- Tests use `pytest.raises()` to expect failures during RED phase
- All imports wrapped in exception handlers to fail gracefully
- Tests define complete expected behavior before implementation
- Fixtures provide reusable test file generators
- Tests follow AAA pattern (Arrange, Act, Assert)
- Each test validates single behavior
- Test names describe expected behavior clearly

---

**Status:** RED phase complete ✅
**Ready for:** GREEN phase implementation
**Test Quality:** Comprehensive, well-structured, follows TDD best practices
