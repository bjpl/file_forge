"""GUI Views.

View components for the FileForge application.
"""

from .dashboard import DashboardView
from .file_browser import FileBrowserView
from .processing import ProcessingView
from .results import ResultsView
from .settings import SettingsView

__all__ = [
    'DashboardView',
    'FileBrowserView',
    'ProcessingView',
    'ResultsView',
    'SettingsView',
]
