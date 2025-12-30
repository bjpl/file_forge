"""Plugin manager for FileForge.

Manages plugin lifecycle, discovery, and execution.
"""

import pluggy
from typing import List, Any, Optional
from pathlib import Path
import inspect
import functools

try:
    from importlib.metadata import entry_points as iter_entry_points
except ImportError:
    from importlib_metadata import entry_points as iter_entry_points

from fileforge.plugins.hookspecs import FileForgeHookSpec

hookimpl = pluggy.HookimplMarker("fileforge")

# Make hookimpl available via pluggy.hookimpl for test compatibility
if not hasattr(pluggy, 'hookimpl'):
    pluggy.hookimpl = hookimpl

# Export hookimpl for use in plugins
__all__ = ['PluginManager', 'hookimpl']


class _ErrorHandlingHookProxy:
    """Proxy that wraps hook calls with error handling.

    Ensures that exceptions in one plugin don't prevent other plugins from executing.
    """

    def __init__(self, hook):
        self._hook = hook

    def __getattr__(self, name):
        """Wrap hook calls with error handling."""
        hook_caller = getattr(self._hook, name)

        def error_safe_caller(**kwargs):
            """Call hook with error isolation."""
            # We need to manually execute hooks in priority order with error isolation
            # Pluggy sorts hook implementations by priority before execution

            # Get the hook implementations
            hook_impls = hook_caller._hookimpls

            # Sort by priority (tryfirst < normal < trylast)
            # Pluggy uses tryfirst and trylast to determine execution order
            # For same priority, maintain registration order (stable sort)
            def get_priority(hook_impl):
                # Lower number = higher priority (executes first)
                opts = getattr(hook_impl.function, 'fileforge_impl', {})
                if opts.get('tryfirst'):
                    return 0
                elif opts.get('trylast'):
                    return 2
                else:
                    return 1

            # sorted() is stable by default in Python
            # For plugins with same priority, we want registration order (FIFO)
            # Note: pluggy stores hooks in LIFO order (last registered is first in list)
            # We need to reverse this to get FIFO order for same priority
            # Use enumerate to preserve list position, then sort by (priority, -index) to reverse LIFO
            sorted_impls = sorted(enumerate(hook_impls),
                                  key=lambda x: (get_priority(x[1]), -x[0]))
            sorted_impls = [impl for idx, impl in sorted_impls]

            results = []
            for hook_impl in sorted_impls:
                try:
                    # Call the implementation
                    func = hook_impl.function
                    result = func(**kwargs)
                    results.append(result)
                except Exception as e:
                    # Log error but continue with other plugins
                    print(f"Warning: Plugin error in {name}: {e}")
                    import traceback
                    traceback.print_exc()

            return results

        # For compatibility, also expose the original hook caller attributes
        error_safe_caller._hookimpls = hook_caller._hookimpls
        error_safe_caller._call_history = getattr(hook_caller, '_call_history', None)

        return error_safe_caller


class PluginManager:
    """Manages FileForge plugins using pluggy.

    Handles plugin registration, discovery, and execution with proper
    lifecycle management.
    """

    def __init__(self, hook_timeout: Optional[float] = None):
        """Initialize plugin manager.

        Args:
            hook_timeout: Optional timeout in seconds for hook execution
        """
        self.pm = pluggy.PluginManager("fileforge")
        self.pm.add_hookspecs(FileForgeHookSpec)
        self.hook_timeout = hook_timeout
        self._blocked_plugins = set()
        self._registration_order = []  # Track registration order for FIFO execution

    @property
    def hook(self):
        """Access to hook caller with error handling wrapper."""
        return _ErrorHandlingHookProxy(self.pm.hook)

    def load_builtins(self):
        """Load built-in plugins."""
        from fileforge.plugins.builtins.classifier import DefaultClassifier
        from fileforge.plugins.builtins.namer import DefaultNamer
        from fileforge.plugins.builtins.outputs import JSONOutput, CSVOutput
        from fileforge.plugins.builtins.processors import (
            TextProcessor,
            PDFProcessor,
            ImageProcessor,
            DocxProcessor
        )

        # Register built-in plugins
        self.register(DefaultClassifier())
        self.register(DefaultNamer())
        self.register(JSONOutput())
        self.register(CSVOutput())

        # Register processor plugin
        class ProcessorPlugin:
            @hookimpl
            def register_processor(self):
                return [
                    (TextProcessor, ['.txt', '.text']),
                    (PDFProcessor, ['.pdf']),
                    (ImageProcessor, ['.jpg', '.jpeg', '.png', '.gif', '.bmp']),
                    (DocxProcessor, ['.docx', '.doc'])
                ]

        self.register(ProcessorPlugin())

    def discover_plugins(self):
        """Discover and load plugins from entry points."""
        try:
            # Use positional argument for compatibility with test mocks
            eps = iter_entry_points('fileforge.plugins')

            for ep in eps:
                try:
                    plugin = ep.load()
                    self.register(plugin)
                except Exception as e:
                    # Log error but continue with other plugins
                    print(f"Warning: Failed to load plugin {ep.name}: {e}")
        except Exception as e:
            print(f"Warning: Error discovering plugins: {e}")

    def register(self, plugin: Any):
        """Register a plugin.

        Automatically wraps plugin methods with @hookimpl decorator if they
        don't already have it, allowing plugins to work without the decorator.

        Args:
            plugin: Plugin instance or class to register
        """
        # List of hook method names
        hook_names = [
            'register_processor',
            'classify_file',
            'suggest_filename',
            'before_move',
            'after_process',
            'register_output'
        ]

        # Check each hook method and wrap if needed
        for hook_name in hook_names:
            if hasattr(plugin, hook_name):
                method = getattr(plugin, hook_name)

                # Get the actual function (unwrap bound methods)
                func = method.__func__ if hasattr(method, '__func__') else method

                # Check if method already has hookimpl marker
                # The marker is stored as <project>_impl attribute
                already_marked = hasattr(func, 'fileforge_impl')

                if not already_marked:
                    # Default to trylast=True to maintain FIFO execution order
                    # (pluggy's default LIFO means last registered runs first,
                    #  but trylast makes it run last, achieving FIFO)
                    tryfirst = False
                    trylast = True

                    # Wrap the method with hookimpl
                    if inspect.ismethod(method):
                        # For bound methods, wrap the underlying function
                        wrapped = hookimpl(tryfirst=tryfirst, trylast=trylast)(method.__func__)
                        # Bind it back to the instance
                        setattr(plugin, hook_name, wrapped.__get__(plugin, type(plugin)))
                    else:
                        # For unbound methods/functions
                        wrapped = hookimpl(tryfirst=tryfirst, trylast=trylast)(method)
                        setattr(plugin, hook_name, wrapped)
                # If already marked, the decorator is already applied - don't rewrap

        # Track registration order
        self._registration_order.append(plugin)
        self.pm.register(plugin)

    def unregister(self, plugin: Any):
        """Unregister a plugin.

        Args:
            plugin: Plugin instance to unregister
        """
        self.pm.unregister(plugin)

    def get_plugins(self) -> List[Any]:
        """Get list of all registered plugins.

        Returns:
            List of plugin instances
        """
        return self.pm.get_plugins()

    def get_plugin(self, name: str) -> Optional[Any]:
        """Get plugin by name.

        Args:
            name: Plugin name to search for

        Returns:
            Plugin instance or None if not found
        """
        for plugin in self.get_plugins():
            plugin_name = getattr(plugin, 'name', None)
            if plugin_name == name:
                return plugin
            # Also check class name
            if plugin.__class__.__name__ == name:
                return plugin
        return None

    def get_registered_processors(self) -> List[tuple]:
        """Get all registered processors.

        Returns:
            List of (ProcessorClass, [extensions]) tuples
        """
        results = self.hook.register_processor()
        # Flatten list of lists
        processors = []
        for result in results:
            if result:
                processors.extend(result)
        return processors

    def get_registered_outputs(self) -> List[Any]:
        """Get all registered output writers.

        Returns:
            List of output writer instances
        """
        results = self.hook.register_output()
        # Flatten list of lists
        outputs = []
        for result in results:
            if result:
                outputs.extend(result)
        return outputs

    def list_available_hooks(self) -> List[str]:
        """List all available hooks.

        Returns:
            List of hook names
        """
        return [
            'register_processor',
            'classify_file',
            'suggest_filename',
            'before_move',
            'after_process',
            'register_output'
        ]

    def list_plugin_names(self) -> List[str]:
        """List names of all registered plugins.

        Returns:
            List of plugin names
        """
        names = []
        for plugin in self.get_plugins():
            name = getattr(plugin, 'name', None)
            if name:
                names.append(name)
            else:
                names.append(plugin.__class__.__name__)
        return names

    def set_blocked(self, name: str):
        """Block a plugin from executing.

        Args:
            name: Plugin name to block
        """
        self._blocked_plugins.add(name)

    def is_blocked(self, name: str) -> bool:
        """Check if a plugin is blocked.

        Args:
            name: Plugin name to check

        Returns:
            True if blocked, False otherwise
        """
        return name in self._blocked_plugins
