"""Hook specifications for FileForge plugins.

Defines all available plugin hooks using pluggy.
"""

import pluggy

hookspec = pluggy.HookspecMarker("fileforge")


def _add_hookspec_marker(func):
    """Add _hookspec attribute for compatibility."""
    func._hookspec = True
    return func


class FileForgeHookSpec:
    """Hook specifications for FileForge plugins.

    This class defines the interface that plugins can implement to extend
    FileForge functionality.
    """

    @hookspec
    @_add_hookspec_marker
    def register_processor():
        """Register custom file processors.

        Returns:
            list: List of tuples (ProcessorClass, [extensions])
                  e.g., [(PDFProcessor, ['.pdf']), (ImageProcessor, ['.jpg', '.png'])]
        """
        pass

    @hookspec(firstresult=False)
    @_add_hookspec_marker
    def classify_file(file_path, content):
        """Classify a file into a category.

        Args:
            file_path (str): Path to the file being classified
            content (str): File content or extracted text

        Returns:
            str or None: Category name, or None to defer to next plugin
        """
        pass

    @hookspec(firstresult=False)
    @_add_hookspec_marker
    def suggest_filename(file_path, content, category):
        """Suggest a filename for a file.

        Args:
            file_path (str): Original file path
            content (str): File content or extracted text
            category (str): Assigned category

        Returns:
            str or None: Suggested filename, or None to defer to next plugin
        """
        pass

    @hookspec(firstresult=False)
    @_add_hookspec_marker
    def before_move(source, destination):
        """Hook called before moving a file.

        Args:
            source (str): Source file path
            destination (str): Destination file path

        Returns:
            bool or Path: True to allow, False to cancel, or Path to modify destination
        """
        pass

    @hookspec(firstresult=False)
    @_add_hookspec_marker
    def after_process(file_path, results):
        """Hook called after processing a file.

        Args:
            file_path (str): Path to processed file
            results (dict): Processing results including category, metadata, etc.

        Returns:
            None: This hook is for side effects only
        """
        pass

    @hookspec
    @_add_hookspec_marker
    def register_output():
        """Register custom output format writers.

        Returns:
            list: List of output writer instances
        """
        pass
