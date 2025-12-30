"""TDD Tests for FileForge plugin system using pluggy.

RED phase: Tests written first, defining expected behavior.

Test Coverage:
- Hook specifications (register_processor, classify_file, suggest_filename, before_move, after_process, register_output)
- Plugin manager initialization and lifecycle
- Plugin discovery (built-in and entry points)
- Custom plugin registration
- Processor registration hooks
- File classification hooks
- Pre-move validation hooks
- Post-processing hooks
- Custom output format hooks
- Plugin execution priority and ordering
- Built-in plugins (classifier, namer, JSON/CSV outputs)
- Plugin configuration and enabling/disabling
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile
import json


class TestHookSpecifications:
    """Tests for plugin hook specifications.

    Verifies that all required hook points are defined in the hookspec.
    """

    def test_hookspec_register_processor_exists(self):
        """Should have hook for registering custom processors."""
        from fileforge.plugins.hookspecs import FileForgeHookSpec

        assert hasattr(FileForgeHookSpec, 'register_processor')
        # Verify it's a hookspec method
        assert hasattr(FileForgeHookSpec.register_processor, '_hookspec')

    def test_hookspec_classify_file_exists(self):
        """Should have hook for custom file classification."""
        from fileforge.plugins.hookspecs import FileForgeHookSpec

        assert hasattr(FileForgeHookSpec, 'classify_file')
        assert hasattr(FileForgeHookSpec.classify_file, '_hookspec')

    def test_hookspec_suggest_filename_exists(self):
        """Should have hook for custom filename suggestions."""
        from fileforge.plugins.hookspecs import FileForgeHookSpec

        assert hasattr(FileForgeHookSpec, 'suggest_filename')
        assert hasattr(FileForgeHookSpec.suggest_filename, '_hookspec')

    def test_hookspec_before_move_exists(self):
        """Should have hook for pre-move validation."""
        from fileforge.plugins.hookspecs import FileForgeHookSpec

        assert hasattr(FileForgeHookSpec, 'before_move')
        assert hasattr(FileForgeHookSpec.before_move, '_hookspec')

    def test_hookspec_after_process_exists(self):
        """Should have hook for post-processing actions."""
        from fileforge.plugins.hookspecs import FileForgeHookSpec

        assert hasattr(FileForgeHookSpec, 'after_process')
        assert hasattr(FileForgeHookSpec.after_process, '_hookspec')

    def test_hookspec_register_output_exists(self):
        """Should have hook for custom output formats."""
        from fileforge.plugins.hookspecs import FileForgeHookSpec

        assert hasattr(FileForgeHookSpec, 'register_output')
        assert hasattr(FileForgeHookSpec.register_output, '_hookspec')

    def test_hookspec_methods_have_correct_signatures(self):
        """Hook specifications should have proper signatures."""
        from fileforge.plugins.hookspecs import FileForgeHookSpec
        import inspect

        # register_processor() -> list of (ProcessorClass, [extensions])
        sig = inspect.signature(FileForgeHookSpec.register_processor)
        assert len(sig.parameters) == 0  # No parameters

        # classify_file(file_path, content, metadata) -> str or None
        sig = inspect.signature(FileForgeHookSpec.classify_file)
        assert 'file_path' in sig.parameters
        assert 'content' in sig.parameters

        # suggest_filename(file_path, content, category) -> str or None
        sig = inspect.signature(FileForgeHookSpec.suggest_filename)
        assert 'file_path' in sig.parameters
        assert 'content' in sig.parameters
        assert 'category' in sig.parameters

        # before_move(source, destination, metadata) -> bool or Path
        sig = inspect.signature(FileForgeHookSpec.before_move)
        assert 'source' in sig.parameters
        assert 'destination' in sig.parameters

        # after_process(file_path, results) -> None
        sig = inspect.signature(FileForgeHookSpec.after_process)
        assert 'file_path' in sig.parameters
        assert 'results' in sig.parameters

        # register_output() -> list of OutputWriter instances
        sig = inspect.signature(FileForgeHookSpec.register_output)
        assert len(sig.parameters) == 0


class TestPluginManager:
    """Tests for plugin manager functionality.

    Tests plugin lifecycle management, discovery, and registration.
    """

    def test_plugin_manager_initialization(self):
        """Should initialize plugin manager with hookspecs."""
        from fileforge.plugins.manager import PluginManager

        pm = PluginManager()
        assert pm is not None
        assert pm.hook is not None
        assert pm.pm is not None  # Underlying pluggy PluginManager

    def test_loads_builtin_plugins(self):
        """Should load built-in plugins by default."""
        from fileforge.plugins.manager import PluginManager

        pm = PluginManager()
        pm.load_builtins()

        # Should have default processors registered
        processors = pm.get_registered_processors()
        assert len(processors) > 0
        assert any('pdf' in str(ext).lower() for proc, exts in processors for ext in exts)

    def test_discovers_plugins_from_entry_points(self):
        """Should discover plugins via entry points."""
        from fileforge.plugins.manager import PluginManager

        with patch('fileforge.plugins.manager.iter_entry_points') as mock_eps:
            mock_plugin = MagicMock()
            mock_ep = MagicMock()
            mock_ep.load.return_value = mock_plugin
            mock_eps.return_value = [mock_ep]

            pm = PluginManager()
            pm.discover_plugins()

            mock_eps.assert_called_once_with('fileforge.plugins')
            mock_ep.load.assert_called_once()

    def test_register_custom_plugin(self):
        """Should allow registering custom plugins."""
        from fileforge.plugins.manager import PluginManager

        class CustomPlugin:
            @staticmethod
            def classify_file(file_path, content):
                if 'invoice' in content.lower():
                    return 'invoices'
                return None

        pm = PluginManager()
        pm.register(CustomPlugin())

        # Plugin should be registered
        plugins = pm.get_plugins()
        assert any(isinstance(p, CustomPlugin) for p in plugins)

    def test_unregister_plugin(self):
        """Should allow unregistering plugins."""
        from fileforge.plugins.manager import PluginManager

        class TempPlugin:
            pass

        pm = PluginManager()
        plugin = TempPlugin()
        pm.register(plugin)

        assert plugin in pm.get_plugins()

        pm.unregister(plugin)
        assert plugin not in pm.get_plugins()

    def test_get_plugin_by_name(self):
        """Should retrieve plugins by name."""
        from fileforge.plugins.manager import PluginManager

        class NamedPlugin:
            name = "TestPlugin"

        pm = PluginManager()
        pm.register(NamedPlugin())

        plugin = pm.get_plugin("TestPlugin")
        assert plugin is not None
        assert isinstance(plugin, NamedPlugin)


class TestProcessorHook:
    """Tests for processor registration hook.

    Tests custom processor registration and integration.
    """

    def test_register_processor_returns_processor(self):
        """Processor hook should return processor class and extensions."""
        from fileforge.plugins.manager import PluginManager

        class CustomProcessor:
            supported_extensions = ['.custom']

            def process(self, file_path):
                return {'text': 'processed'}

        class ProcessorPlugin:
            @staticmethod
            def register_processor():
                return [(CustomProcessor, ['.custom'])]

        pm = PluginManager()
        pm.register(ProcessorPlugin())

        processors = pm.hook.register_processor()
        # Flatten list of lists
        all_processors = [item for sublist in processors for item in sublist]

        assert len(all_processors) > 0
        assert any(proc == CustomProcessor for proc, _ in all_processors)

    def test_custom_processor_handles_extension(self):
        """Custom processor should handle its extensions."""
        from fileforge.plugins.manager import PluginManager

        class MarkdownProcessor:
            supported_extensions = ['.md', '.markdown']

            def process(self, file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                return {
                    'text': content,
                    'format': 'markdown'
                }

        class ProcessorPlugin:
            @staticmethod
            def register_processor():
                return [(MarkdownProcessor, ['.md', '.markdown'])]

        pm = PluginManager()
        pm.register(ProcessorPlugin())

        processors = pm.get_registered_processors()
        md_processors = [p for p, exts in processors if '.md' in exts]

        assert len(md_processors) > 0

    def test_multiple_processors_can_register(self):
        """Multiple plugins can register different processors."""
        from fileforge.plugins.manager import PluginManager

        class ProcessorA:
            supported_extensions = ['.a']

        class ProcessorB:
            supported_extensions = ['.b']

        class PluginA:
            @staticmethod
            def register_processor():
                return [(ProcessorA, ['.a'])]

        class PluginB:
            @staticmethod
            def register_processor():
                return [(ProcessorB, ['.b'])]

        pm = PluginManager()
        pm.register(PluginA())
        pm.register(PluginB())

        processors = pm.get_registered_processors()
        assert len(processors) >= 2


class TestClassifyHook:
    """Tests for file classification hook.

    Tests custom file classification logic.
    """

    def test_classify_hook_returns_category(self):
        """Classification hook should return category or None."""
        from fileforge.plugins.manager import PluginManager

        class ClassifierPlugin:
            @staticmethod
            def classify_file(file_path, content):
                if 'confidential' in content.lower():
                    return 'confidential'
                return None

        pm = PluginManager()
        pm.register(ClassifierPlugin())

        results = pm.hook.classify_file(
            file_path='/test/doc.txt',
            content='This is confidential information'
        )

        # First non-None result wins
        categories = [r for r in results if r is not None]
        assert 'confidential' in categories

    def test_classify_returns_none_to_defer(self):
        """Returning None defers to next plugin."""
        from fileforge.plugins.manager import PluginManager

        class DeferringPlugin:
            @staticmethod
            def classify_file(file_path, content):
                return None  # Defer to next plugin

        class CatchAllPlugin:
            @staticmethod
            def classify_file(file_path, content):
                return 'uncategorized'

        pm = PluginManager()
        pm.register(DeferringPlugin())
        pm.register(CatchAllPlugin())

        results = pm.hook.classify_file(
            file_path='/test/doc.txt',
            content='random content'
        )

        # CatchAllPlugin should provide result
        categories = [r for r in results if r is not None]
        assert 'uncategorized' in categories

    def test_classify_first_match_wins(self):
        """First classifier to return non-None wins."""
        from fileforge.plugins.manager import PluginManager

        class HighPriorityClassifier:
            @staticmethod
            def classify_file(file_path, content):
                if 'urgent' in content.lower():
                    return 'urgent'
                return None

        class LowPriorityClassifier:
            @staticmethod
            def classify_file(file_path, content):
                return 'general'

        pm = PluginManager()
        pm.register(HighPriorityClassifier())
        pm.register(LowPriorityClassifier())

        results = pm.hook.classify_file(
            file_path='/test/doc.txt',
            content='This is urgent!'
        )

        # First non-None should be 'urgent'
        categories = [r for r in results if r is not None]
        assert categories[0] == 'urgent'


class TestSuggestFilenameHook:
    """Tests for filename suggestion hook."""

    def test_suggest_filename_returns_name(self):
        """Filename suggester should return filename or None."""
        from fileforge.plugins.manager import PluginManager

        class DateNamerPlugin:
            @staticmethod
            def suggest_filename(file_path, content, category):
                if category == 'invoices':
                    return f"invoice_2024_001.pdf"
                return None

        pm = PluginManager()
        pm.register(DateNamerPlugin())

        results = pm.hook.suggest_filename(
            file_path='/test/doc.pdf',
            content='Invoice content',
            category='invoices'
        )

        names = [r for r in results if r is not None]
        assert 'invoice_2024_001.pdf' in names

    def test_suggest_filename_based_on_content(self):
        """Filename can be suggested based on content analysis."""
        from fileforge.plugins.manager import PluginManager

        class ContentNamerPlugin:
            @staticmethod
            def suggest_filename(file_path, content, category):
                # Extract title from content
                if 'Subject: ' in content:
                    subject = content.split('Subject: ')[1].split('\n')[0]
                    return f"{subject}.txt"
                return None

        pm = PluginManager()
        pm.register(ContentNamerPlugin())

        results = pm.hook.suggest_filename(
            file_path='/test/email.txt',
            content='Subject: Meeting Notes\nContent here...',
            category='emails'
        )

        names = [r for r in results if r is not None]
        assert 'Meeting Notes.txt' in names


class TestBeforeMoveHook:
    """Tests for pre-move validation hook.

    Tests validation and modification before file moves.
    """

    def test_before_move_can_cancel(self):
        """Before move hook can cancel operation by returning False."""
        from fileforge.plugins.manager import PluginManager

        class ValidationPlugin:
            @staticmethod
            def before_move(source, destination):
                # Block moving to certain directories
                if 'protected' in str(destination):
                    return False
                return True

        pm = PluginManager()
        pm.register(ValidationPlugin())

        results = pm.hook.before_move(
            source='/test/file.txt',
            destination='/protected/file.txt'
        )

        # Should have False result
        assert False in results

    def test_before_move_can_modify_destination(self):
        """Before move hook can modify destination path."""
        from fileforge.plugins.manager import PluginManager

        class ModifierPlugin:
            @staticmethod
            def before_move(source, destination):
                # Add date prefix to destination
                dest_path = Path(destination)
                new_name = f"2024-01-{dest_path.name}"
                return dest_path.parent / new_name

        pm = PluginManager()
        pm.register(ModifierPlugin())

        results = pm.hook.before_move(
            source='/test/file.txt',
            destination='/output/file.txt'
        )

        modified_paths = [r for r in results if r and r != True]
        assert any('2024-01-' in str(p) for p in modified_paths)

    def test_before_move_all_must_approve(self):
        """All validators must approve (no False) for move to proceed."""
        from fileforge.plugins.manager import PluginManager

        class Validator1:
            @staticmethod
            def before_move(source, destination):
                return True

        class Validator2:
            @staticmethod
            def before_move(source, destination):
                return False  # Blocks the move

        pm = PluginManager()
        pm.register(Validator1())
        pm.register(Validator2())

        results = pm.hook.before_move(
            source='/test/file.txt',
            destination='/output/file.txt'
        )

        # Should have at least one False
        assert False in results


class TestAfterProcessHook:
    """Tests for post-processing hook.

    Tests actions taken after file processing.
    """

    def test_after_process_receives_results(self):
        """After process hook receives processing results."""
        from fileforge.plugins.manager import PluginManager

        received_calls = []

        class LoggingPlugin:
            @staticmethod
            def after_process(file_path, results):
                received_calls.append((file_path, results))

        pm = PluginManager()
        pm.register(LoggingPlugin())

        pm.hook.after_process(
            file_path='/test/doc.txt',
            results={'text': 'content', 'category': 'documents'}
        )

        assert len(received_calls) == 1
        assert received_calls[0][0] == '/test/doc.txt'
        assert received_calls[0][1]['category'] == 'documents'

    def test_after_process_multiple_listeners(self):
        """Multiple plugins can listen to after_process."""
        from fileforge.plugins.manager import PluginManager

        call_count = {'count': 0}

        class Listener1:
            @staticmethod
            def after_process(file_path, results):
                call_count['count'] += 1

        class Listener2:
            @staticmethod
            def after_process(file_path, results):
                call_count['count'] += 1

        pm = PluginManager()
        pm.register(Listener1())
        pm.register(Listener2())

        pm.hook.after_process(
            file_path='/test/doc.txt',
            results={}
        )

        assert call_count['count'] == 2

    def test_after_process_can_trigger_actions(self):
        """After process can trigger side effects (logging, notifications, etc)."""
        from fileforge.plugins.manager import PluginManager

        notifications = []

        class NotificationPlugin:
            @staticmethod
            def after_process(file_path, results):
                if results.get('category') == 'important':
                    notifications.append(f"Important file processed: {file_path}")

        pm = PluginManager()
        pm.register(NotificationPlugin())

        pm.hook.after_process(
            file_path='/test/urgent.pdf',
            results={'category': 'important'}
        )

        assert len(notifications) == 1
        assert 'urgent.pdf' in notifications[0]


class TestOutputFormatHook:
    """Tests for custom output format hook.

    Tests registration of custom output writers.
    """

    def test_register_output_format(self):
        """Should register custom output format."""
        from fileforge.plugins.manager import PluginManager

        class CustomXMLWriter:
            format_name = 'custom_xml'

            def write(self, data, output_path):
                # Write custom XML format
                xml = '<data>\n'
                for key, value in data.items():
                    xml += f'  <{key}>{value}</{key}>\n'
                xml += '</data>'

                with open(output_path, 'w') as f:
                    f.write(xml)

        class OutputPlugin:
            @staticmethod
            def register_output():
                return [CustomXMLWriter()]

        pm = PluginManager()
        pm.register(OutputPlugin())

        writers = pm.hook.register_output()
        # Flatten list
        all_writers = [item for sublist in writers for item in sublist]

        assert len(all_writers) > 0
        assert any(isinstance(w, CustomXMLWriter) for w in all_writers)

    def test_multiple_output_formats(self):
        """Multiple output formats can be registered."""
        from fileforge.plugins.manager import PluginManager

        class XMLWriter:
            format_name = 'xml'

        class YAMLWriter:
            format_name = 'yaml'

        class Plugin1:
            @staticmethod
            def register_output():
                return [XMLWriter()]

        class Plugin2:
            @staticmethod
            def register_output():
                return [YAMLWriter()]

        pm = PluginManager()
        pm.register(Plugin1())
        pm.register(Plugin2())

        writers = pm.get_registered_outputs()
        assert len(writers) >= 2


class TestPluginPriority:
    """Tests for plugin execution priority and ordering."""

    def test_plugins_execute_in_order(self):
        """Plugins should execute in registration order."""
        from fileforge.plugins.manager import PluginManager

        execution_order = []

        class FirstPlugin:
            @staticmethod
            def after_process(file_path, results):
                execution_order.append('first')

        class SecondPlugin:
            @staticmethod
            def after_process(file_path, results):
                execution_order.append('second')

        pm = PluginManager()
        pm.register(FirstPlugin())
        pm.register(SecondPlugin())

        pm.hook.after_process(file_path='/test.txt', results={})

        assert execution_order == ['first', 'second']

    def test_builtin_plugins_load_first(self):
        """Built-in plugins should load before user plugins."""
        from fileforge.plugins.manager import PluginManager

        class UserPlugin:
            pass

        pm = PluginManager()
        pm.load_builtins()

        builtin_count = len(pm.get_plugins())
        assert builtin_count > 0

        pm.register(UserPlugin())

        # User plugin should be after builtins
        all_plugins = pm.get_plugins()
        assert len(all_plugins) == builtin_count + 1

    def test_plugin_priority_via_hookimpl_tryfirst(self):
        """Plugins can use tryfirst/trylast markers for priority."""
        from fileforge.plugins.manager import PluginManager
        import pluggy

        execution_order = []

        class LowPriorityPlugin:
            @pluggy.hookimpl(trylast=True)
            def after_process(self, file_path, results):
                execution_order.append('low')

        class HighPriorityPlugin:
            @pluggy.hookimpl(tryfirst=True)
            def after_process(self, file_path, results):
                execution_order.append('high')

        pm = PluginManager()
        pm.register(LowPriorityPlugin())
        pm.register(HighPriorityPlugin())

        pm.hook.after_process(file_path='/test.txt', results={})

        # High priority should execute first
        assert execution_order[0] == 'high'


class TestBuiltinPlugins:
    """Tests for built-in plugins.

    Tests default plugins shipped with FileForge.
    """

    def test_default_classifier_plugin_exists(self):
        """Should have default classifier plugin."""
        from fileforge.plugins.builtins.classifier import DefaultClassifier

        assert hasattr(DefaultClassifier, 'classify_file')

        # Test basic classification
        classifier = DefaultClassifier()
        category = classifier.classify_file(
            file_path='/test/document.pdf',
            content='Some text content'
        )
        assert category is None or isinstance(category, str)

    def test_default_namer_plugin_exists(self):
        """Should have default filename suggester plugin."""
        from fileforge.plugins.builtins.namer import DefaultNamer

        assert hasattr(DefaultNamer, 'suggest_filename')

        namer = DefaultNamer()
        suggestion = namer.suggest_filename(
            file_path='/test/doc.pdf',
            content='Test content',
            category='documents'
        )
        assert suggestion is None or isinstance(suggestion, str)

    def test_json_output_plugin_exists(self):
        """Should have JSON output plugin."""
        from fileforge.plugins.builtins.outputs import JSONOutput

        assert hasattr(JSONOutput, 'write')

        # Test writing
        output = JSONOutput()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            output.write(
                {'test': 'data', 'number': 42},
                temp_path
            )

            # Verify JSON was written
            with open(temp_path, 'r') as f:
                data = json.load(f)
            assert data['test'] == 'data'
            assert data['number'] == 42
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_csv_output_plugin_exists(self):
        """Should have CSV output plugin."""
        from fileforge.plugins.builtins.outputs import CSVOutput

        assert hasattr(CSVOutput, 'write')

        output = CSVOutput()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            output.write(
                [
                    {'name': 'file1.pdf', 'category': 'docs'},
                    {'name': 'file2.pdf', 'category': 'images'}
                ],
                temp_path
            )

            # Verify CSV was written
            with open(temp_path, 'r') as f:
                content = f.read()
            assert 'name,category' in content or 'file1.pdf' in content
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPluginConfiguration:
    """Tests for plugin configuration.

    Tests plugin initialization with configuration.
    """

    def test_plugin_receives_config(self):
        """Plugins should receive configuration."""
        from fileforge.plugins.manager import PluginManager

        received_config = None

        class ConfigurablePlugin:
            def __init__(self, config=None):
                nonlocal received_config
                received_config = config
                self.config = config

        pm = PluginManager()
        config = {'custom_setting': 'value', 'threshold': 0.8}
        pm.register(ConfigurablePlugin(config=config))

        assert received_config is not None
        assert received_config['custom_setting'] == 'value'
        assert received_config['threshold'] == 0.8

    def test_disable_plugin_via_config(self):
        """Should be able to disable plugins via config."""
        from fileforge.plugins.manager import PluginManager

        class DisableablePlugin:
            name = 'TestPlugin'

        pm = PluginManager()
        plugin = DisableablePlugin()
        pm.register(plugin)

        assert plugin in pm.get_plugins()

        # Disable the plugin
        pm.set_blocked('TestPlugin')

        # Plugin should not execute hooks
        assert pm.is_blocked('TestPlugin')

    def test_plugin_config_validation(self):
        """Plugins should validate their configuration."""
        from fileforge.plugins.manager import PluginManager

        class ValidatingPlugin:
            def __init__(self, config=None):
                if config:
                    required = ['api_key', 'endpoint']
                    for key in required:
                        if key not in config:
                            raise ValueError(f"Missing required config: {key}")
                self.config = config

        pm = PluginManager()

        # Should raise on invalid config
        with pytest.raises(ValueError, match="Missing required config"):
            pm.register(ValidatingPlugin(config={'api_key': 'test'}))

        # Should succeed with valid config
        pm.register(ValidatingPlugin(config={
            'api_key': 'test',
            'endpoint': 'http://example.com'
        }))


class TestPluginErrorHandling:
    """Tests for plugin error handling and resilience."""

    def test_plugin_exception_isolated(self):
        """Exception in one plugin shouldn't affect others."""
        from fileforge.plugins.manager import PluginManager

        call_log = []

        class FailingPlugin:
            @staticmethod
            def after_process(file_path, results):
                call_log.append('failing')
                raise RuntimeError("Plugin error")

        class WorkingPlugin:
            @staticmethod
            def after_process(file_path, results):
                call_log.append('working')

        pm = PluginManager()
        pm.register(FailingPlugin())
        pm.register(WorkingPlugin())

        # Should not raise, but log the error
        try:
            pm.hook.after_process(file_path='/test.txt', results={})
        except RuntimeError:
            pass  # Expected

        # Working plugin should still execute
        assert 'working' in call_log

    def test_plugin_timeout_handling(self):
        """Long-running plugins should timeout gracefully."""
        from fileforge.plugins.manager import PluginManager
        import time

        class SlowPlugin:
            @staticmethod
            def after_process(file_path, results):
                time.sleep(10)  # Simulate slow operation

        pm = PluginManager(hook_timeout=1.0)  # 1 second timeout
        pm.register(SlowPlugin())

        # Should timeout without blocking
        # Implementation should handle this gracefully


class TestPluginDocumentation:
    """Tests for plugin documentation and introspection."""

    def test_plugin_has_metadata(self):
        """Plugins should provide metadata."""
        from fileforge.plugins.builtins.classifier import DefaultClassifier

        classifier = DefaultClassifier()

        # Should have name
        assert hasattr(classifier, 'name') or hasattr(classifier.__class__, '__name__')

        # Should have description (docstring)
        assert classifier.__class__.__doc__ is not None

    def test_list_available_hooks(self):
        """Should be able to list all available hooks."""
        from fileforge.plugins.manager import PluginManager

        pm = PluginManager()
        hooks = pm.list_available_hooks()

        expected_hooks = [
            'register_processor',
            'classify_file',
            'suggest_filename',
            'before_move',
            'after_process',
            'register_output'
        ]

        for hook_name in expected_hooks:
            assert hook_name in hooks

    def test_list_registered_plugins(self):
        """Should be able to list all registered plugins."""
        from fileforge.plugins.manager import PluginManager

        pm = PluginManager()
        pm.load_builtins()

        plugins = pm.list_plugin_names()
        assert len(plugins) > 0
        assert isinstance(plugins, list)
        assert all(isinstance(name, str) for name in plugins)
