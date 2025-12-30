"""Built-in file processors.

Provides default processors for common file types.
"""


class TextProcessor:
    """Text file processor."""

    supported_extensions = ['.txt', '.text']

    def process(self, file_path):
        """Process text file.

        Args:
            file_path: Path to text file

        Returns:
            dict with extracted content
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return {
            'text': content,
            'format': 'text'
        }


class PDFProcessor:
    """PDF file processor."""

    supported_extensions = ['.pdf']

    def process(self, file_path):
        """Process PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            dict with extracted content
        """
        # Basic implementation - would use PyPDF2 or similar in production
        return {
            'text': '',
            'format': 'pdf'
        }


class ImageProcessor:
    """Image file processor."""

    supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    def process(self, file_path):
        """Process image file.

        Args:
            file_path: Path to image file

        Returns:
            dict with extracted content
        """
        # Basic implementation - would use PIL/Pillow or OCR in production
        return {
            'text': '',
            'format': 'image'
        }


class DocxProcessor:
    """DOCX file processor."""

    supported_extensions = ['.docx', '.doc']

    def process(self, file_path):
        """Process DOCX file.

        Args:
            file_path: Path to DOCX file

        Returns:
            dict with extracted content
        """
        # Basic implementation - would use python-docx in production
        return {
            'text': '',
            'format': 'docx'
        }
