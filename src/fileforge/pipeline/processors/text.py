"""Text processor for plain text and Markdown files.

Handles text extraction, structure parsing, and metadata extraction.
"""
import re
from pathlib import Path
from typing import Any, Dict, List

from .document import ExtractedContent, ProcessingError


class TextProcessor:
    """Processor for plain text and Markdown files.

    Extracts text, structure (headings, links, code blocks), and metadata.
    Supports frontmatter extraction and document type detection.
    """

    supported_extensions = ['.txt', '.md', '.markdown', '.rst', '.text']

    def process(self, file_path: Path) -> ExtractedContent:
        """Process a text file and extract content.

        Args:
            file_path: Path to the text file

        Returns:
            ExtractedContent with text, structure, and metadata

        Raises:
            ProcessingError: If processing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")

        try:
            # Try UTF-8 first, fall back to other encodings
            text = self._read_with_encoding(file_path)
        except Exception as e:
            raise ProcessingError(f"Failed to read file: {str(e)}") from e

        extension = file_path.suffix.lower()

        if extension in ['.md', '.markdown']:
            return self._process_markdown(text, file_path)
        else:
            return self._process_plain_text(text, file_path)

    def _read_with_encoding(self, file_path: Path) -> str:
        """Read file with automatic encoding detection.

        Args:
            file_path: Path to file

        Returns:
            File content as string
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort: read as bytes and decode with errors='replace'
        return file_path.read_bytes().decode('utf-8', errors='replace')

    def _process_plain_text(self, text: str, file_path: Path) -> ExtractedContent:
        """Process plain text file.

        Args:
            text: File content
            file_path: Path to file

        Returns:
            ExtractedContent with extracted data
        """
        metadata = {
            'filename': file_path.name,
            'size': len(text),
            'lines': len(text.split('\n'))
        }

        # Calculate confidence based on content quality
        confidence = self._calculate_text_confidence(text)

        return ExtractedContent(
            text=text,
            structure={},
            metadata=metadata,
            pages=[],
            embedded_images=[],
            confidence=confidence
        )

    def _process_markdown(self, text: str, file_path: Path) -> ExtractedContent:
        """Process Markdown file with structure extraction.

        Args:
            text: File content
            file_path: Path to file

        Returns:
            ExtractedContent with extracted data
        """
        # Extract frontmatter if present
        frontmatter, content = self._extract_frontmatter(text)

        # Initialize metadata
        metadata = frontmatter.copy()
        metadata['filename'] = file_path.name

        # Detect document type
        doc_type = self._detect_document_type(file_path.name, content)
        if doc_type:
            metadata['doc_type'] = doc_type

        # Extract structure
        structure = {}

        # Extract headings
        headings = self._extract_headings(content)
        if headings:
            structure['headings'] = headings

        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)
        if code_blocks:
            structure['code_blocks'] = code_blocks

        # Extract links
        links = self._extract_links(content)
        if links:
            structure['links'] = links

        # Calculate confidence
        confidence = self._calculate_text_confidence(content)

        return ExtractedContent(
            text=content,
            structure=structure,
            metadata=metadata,
            pages=[],
            embedded_images=[],
            confidence=confidence
        )

    def _extract_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from Markdown.

        Args:
            text: Markdown content

        Returns:
            Tuple of (frontmatter dict, content without frontmatter)
        """
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(frontmatter_pattern, text, re.DOTALL)

        if not match:
            return {}, text

        frontmatter_text = match.group(1)
        content = text[match.end():]

        # Parse YAML frontmatter (simple parser, doesn't require pyyaml)
        frontmatter = {}
        for line in frontmatter_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                frontmatter[key] = value

        return frontmatter, content

    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """Extract headings from Markdown.

        Args:
            text: Markdown content

        Returns:
            List of heading dictionaries
        """
        headings = []
        heading_pattern = r'^(#{1,6})\s+(.+)$'

        for line in text.split('\n'):
            match = re.match(heading_pattern, line)
            if match:
                level = len(match.group(1))
                text_content = match.group(2).strip()
                headings.append({
                    'level': level,
                    'text': text_content
                })

        return headings

    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Extract code blocks from Markdown.

        Args:
            text: Markdown content

        Returns:
            List of code block dictionaries
        """
        code_blocks = []
        # Match fenced code blocks with optional language
        pattern = r'```(\w*)\n(.*?)```'

        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            code_blocks.append({
                'language': language,
                'code': code
            })

        return code_blocks

    def _extract_links(self, text: str) -> List[Dict[str, str]]:
        """Extract links from Markdown.

        Args:
            text: Markdown content

        Returns:
            List of link dictionaries
        """
        links = []
        # Match Markdown links [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

        for match in re.finditer(link_pattern, text):
            link_text = match.group(1)
            url = match.group(2)
            links.append({
                'text': link_text,
                'url': url
            })

        return links

    def _detect_document_type(self, filename: str, content: str) -> str:
        """Detect document type based on filename and content.

        Args:
            filename: Name of the file
            content: File content

        Returns:
            Document type identifier
        """
        filename_lower = filename.lower()

        # Check filename patterns
        if filename_lower == 'readme.md' or filename_lower.startswith('readme'):
            return 'readme'
        elif filename_lower == 'changelog.md' or filename_lower.startswith('changelog'):
            return 'changelog'
        elif filename_lower == 'license' or filename_lower == 'license.md':
            return 'license'
        elif filename_lower.endswith('notes.md') or 'notes' in filename_lower:
            return 'notes'

        # Check content patterns
        content_lower = content.lower()
        if '## installation' in content_lower and '## usage' in content_lower:
            return 'readme'
        elif 'todo:' in content_lower or '- [ ]' in content:
            return 'todo'

        return 'document'

    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score based on text quality.

        Args:
            text: Text content

        Returns:
            Confidence score between 0 and 1
        """
        if not text.strip():
            return 0.0

        # Basic heuristics for text quality
        words = text.split()
        if not words:
            return 0.5

        # Check for readable words (contains vowels and reasonable length)
        readable_words = 0
        for word in words:
            if len(word) >= 2 and any(c in 'aeiouAEIOU' for c in word):
                readable_words += 1

        readable_ratio = readable_words / len(words) if words else 0

        # Check for sentence structure (periods, question marks)
        has_punctuation = any(p in text for p in '.?!')

        # Calculate confidence
        confidence = 0.5  # Base confidence
        confidence += readable_ratio * 0.3  # Up to +0.3 for readable words
        confidence += 0.2 if has_punctuation else 0  # +0.2 for sentence structure

        return min(confidence, 1.0)
