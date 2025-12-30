"""FileForge GUI package.

Modern Windows GUI using PySide6 with Fluent Design.
"""
from .state import AppState
from .main_window import MainWindow
from .app import run_gui

__all__ = ['AppState', 'MainWindow', 'run_gui']
