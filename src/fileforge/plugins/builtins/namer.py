"""Default filename suggester plugin.

Provides basic filename suggestion based on content and category.
"""

import pluggy

hookimpl = pluggy.HookimplMarker("fileforge")


class DefaultNamer:
    """Default filename suggester.

    Provides basic filename suggestions based on file content and category.
    """

    name = "DefaultNamer"

    @hookimpl
    def suggest_filename(self, file_path, content, category):
        """Suggest a filename.

        Args:
            file_path: Original file path
            content: File content or extracted text
            category: Assigned category

        Returns:
            Suggested filename or None to defer to other namers
        """
        # Basic naming - defer to user plugins by default
        # This can be extended with more sophisticated logic
        return None
