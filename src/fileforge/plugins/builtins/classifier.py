"""Default file classifier plugin.

Provides basic file classification based on content analysis.
"""

import pluggy

hookimpl = pluggy.HookimplMarker("fileforge")


class DefaultClassifier:
    """Default file classifier.

    Provides basic classification logic for common file types and content patterns.
    """

    name = "DefaultClassifier"

    @hookimpl
    def classify_file(self, file_path, content):
        """Classify file based on content.

        Args:
            file_path: Path to the file
            content: File content or extracted text

        Returns:
            Category name or None to defer to other classifiers
        """
        # Basic classification - defer to user plugins by default
        # This can be extended with more sophisticated logic
        return None
