"""FileForge plugin system.

Provides extensibility through pluggy-based plugin architecture.
"""

from fileforge.plugins.manager import PluginManager
from fileforge.plugins.hookspecs import FileForgeHookSpec

__all__ = ['PluginManager', 'FileForgeHookSpec']
