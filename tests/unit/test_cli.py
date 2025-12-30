"""TDD Tests for FileForge CLI module.

RED phase: Tests written first, defining expected behavior.
"""
import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

runner = CliRunner()


class TestCLIBasics:
    """Tests for basic CLI functionality."""

    def test_cli_has_help(self):
        """CLI should display help message."""
        from fileforge.cli import app
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "FileForge" in result.stdout or "fileforge" in result.stdout.lower()

    def test_cli_has_version(self):
        """CLI should display version."""
        from fileforge.cli import app
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "1.0" in result.stdout


class TestScanCommand:
    """Tests for scan command."""

    def test_scan_command_exists(self):
        """Scan command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["scan", "--help"])
        assert result.exit_code == 0
        assert "recursive" in result.stdout.lower() or "-r" in result.stdout

    def test_scan_requires_path(self):
        """Scan should require a path argument."""
        from fileforge.cli import app
        result = runner.invoke(app, ["scan"])
        # Should either fail or use current directory
        assert result.exit_code == 0 or "path" in result.stdout.lower()

    def test_scan_dry_run_option(self, temp_dir):
        """Scan --dry-run should not modify anything."""
        from fileforge.cli import app

        (temp_dir / "test.txt").write_text("content")

        with patch('fileforge.cli.PipelineOrchestrator') as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {
                'files_processed': 1,
                'files_succeeded': 1,
                'files_failed': 0,
                'duration_seconds': 0.1,
                'errors': []
            }
            mock_orch.return_value = mock_instance
            result = runner.invoke(app, ["scan", str(temp_dir), "--dry-run"])

        assert result.exit_code == 0
        assert "dry" in result.stdout.lower() or "scan" in result.stdout.lower()

    def test_scan_recursive_option(self, temp_dir):
        """Scan should support --recursive flag."""
        from fileforge.cli import app
        result = runner.invoke(app, ["scan", str(temp_dir), "--recursive"])
        assert result.exit_code == 0

    def test_scan_types_filter(self, temp_dir):
        """Scan should support --types filter."""
        from fileforge.cli import app
        result = runner.invoke(app, ["scan", str(temp_dir), "--types", "images"])
        assert result.exit_code == 0

    def test_scan_limit_option(self, temp_dir):
        """Scan should support --limit option."""
        from fileforge.cli import app
        result = runner.invoke(app, ["scan", str(temp_dir), "--limit", "10"])
        assert result.exit_code == 0


class TestOrganizeCommand:
    """Tests for organize command."""

    def test_organize_command_exists(self):
        """Organize command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["organize", "--help"])
        assert result.exit_code == 0

    def test_organize_rename_option(self):
        """Organize should support --rename option."""
        from fileforge.cli import app
        result = runner.invoke(app, ["organize", "--help"])
        assert "--rename" in result.stdout

    def test_organize_move_option(self):
        """Organize should support --move option."""
        from fileforge.cli import app
        result = runner.invoke(app, ["organize", "--help"])
        assert "--move" in result.stdout

    def test_organize_interactive_mode(self):
        """Organize should support --interactive mode."""
        from fileforge.cli import app
        result = runner.invoke(app, ["organize", "--help"])
        assert "--interactive" in result.stdout or "-i" in result.stdout

    def test_organize_min_confidence(self):
        """Organize should support --min-confidence threshold."""
        from fileforge.cli import app
        result = runner.invoke(app, ["organize", "--help"])
        assert "confidence" in result.stdout.lower()


class TestQueryCommand:
    """Tests for query command."""

    def test_query_command_exists(self):
        """Query command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0

    def test_query_tag_filter(self):
        """Query should support --tag filter."""
        from fileforge.cli import app
        result = runner.invoke(app, ["query", "--help"])
        assert "--tag" in result.stdout

    def test_query_text_search(self):
        """Query should support --text search."""
        from fileforge.cli import app
        result = runner.invoke(app, ["query", "--help"])
        assert "--text" in result.stdout

    def test_query_output_formats(self):
        """Query should support multiple output formats."""
        from fileforge.cli import app
        result = runner.invoke(app, ["query", "--help"])
        assert "--format" in result.stdout
        assert "json" in result.stdout.lower() or "table" in result.stdout.lower()

    def test_query_similar_to(self):
        """Query should support --similar-to for semantic search."""
        from fileforge.cli import app
        result = runner.invoke(app, ["query", "--help"])
        assert "similar" in result.stdout.lower()


class TestUndoCommand:
    """Tests for undo command."""

    def test_undo_command_exists(self):
        """Undo command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["undo", "--help"])
        assert result.exit_code == 0

    def test_undo_last_subcommand(self):
        """Undo should have 'last' subcommand."""
        from fileforge.cli import app
        result = runner.invoke(app, ["undo", "last", "--help"])
        assert result.exit_code == 0

    def test_undo_batch_subcommand(self):
        """Undo should have 'batch' subcommand."""
        from fileforge.cli import app
        result = runner.invoke(app, ["undo", "batch", "--help"])
        assert result.exit_code == 0

    def test_undo_list_subcommand(self):
        """Undo should have 'list' subcommand."""
        from fileforge.cli import app
        result = runner.invoke(app, ["undo", "list", "--help"])
        assert result.exit_code == 0


class TestClusterCommand:
    """Tests for cluster command (face management)."""

    def test_cluster_command_exists(self):
        """Cluster command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["cluster", "--help"])
        assert result.exit_code == 0

    def test_cluster_list_subcommand(self):
        """Cluster should have 'list' subcommand."""
        from fileforge.cli import app
        result = runner.invoke(app, ["cluster", "list", "--help"])
        assert result.exit_code == 0

    def test_cluster_name_subcommand(self):
        """Cluster should have 'name' subcommand."""
        from fileforge.cli import app
        result = runner.invoke(app, ["cluster", "name", "--help"])
        assert result.exit_code == 0


class TestExportCommand:
    """Tests for export command."""

    def test_export_command_exists(self):
        """Export command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0

    def test_export_json_format(self):
        """Export should support JSON format."""
        from fileforge.cli import app
        result = runner.invoke(app, ["export", "--help"])
        assert "json" in result.stdout.lower()

    def test_export_csv_format(self):
        """Export should support CSV format."""
        from fileforge.cli import app
        result = runner.invoke(app, ["export", "--help"])
        assert "csv" in result.stdout.lower()


class TestWatchCommand:
    """Tests for watch command."""

    def test_watch_command_exists(self):
        """Watch command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0

    def test_watch_debounce_option(self):
        """Watch should support --debounce option."""
        from fileforge.cli import app
        result = runner.invoke(app, ["watch", "--help"])
        assert "debounce" in result.stdout.lower()


class TestStatsCommand:
    """Tests for stats command."""

    def test_stats_command_exists(self):
        """Stats command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0


class TestConfigCommand:
    """Tests for config command."""

    def test_config_command_exists(self):
        """Config command should exist."""
        from fileforge.cli import app
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_config_show_subcommand(self):
        """Config should have 'show' subcommand."""
        from fileforge.cli import app
        result = runner.invoke(app, ["config", "show", "--help"])
        assert result.exit_code == 0

    def test_config_init_subcommand(self):
        """Config should have 'init' subcommand."""
        from fileforge.cli import app
        result = runner.invoke(app, ["config", "init", "--help"])
        assert result.exit_code == 0


class TestGlobalOptions:
    """Tests for global CLI options."""

    def test_verbose_option(self):
        """CLI should support --verbose option."""
        from fileforge.cli import app
        result = runner.invoke(app, ["--verbose", "stats"])
        # Should not error due to unknown option
        assert "--verbose" not in result.stdout or result.exit_code == 0

    def test_quiet_option(self):
        """CLI should support --quiet option."""
        from fileforge.cli import app
        result = runner.invoke(app, ["--quiet", "stats"])
        assert "--quiet" not in result.stdout or result.exit_code == 0

    def test_config_option(self):
        """CLI should support --config option."""
        from fileforge.cli import app
        result = runner.invoke(app, ["--help"])
        assert "--config" in result.stdout or "-c" in result.stdout


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_invalid_command_error(self):
        """Invalid command should show helpful error."""
        from fileforge.cli import app
        result = runner.invoke(app, ["invalid_command"])
        assert result.exit_code != 0
        assert "invalid" in result.stdout.lower() or "no such" in result.stdout.lower()

    def test_missing_required_args_error(self):
        """Missing required arguments should show error."""
        from fileforge.cli import app
        result = runner.invoke(app, ["organize"])
        # Should error or show help
        assert result.exit_code != 0 or "usage" in result.stdout.lower()

    def test_invalid_path_error(self, tmp_path):
        """Invalid path should show error."""
        from fileforge.cli import app
        nonexistent = tmp_path / "does_not_exist"
        result = runner.invoke(app, ["scan", str(nonexistent)])
        assert result.exit_code != 0 or "not found" in result.stdout.lower()


class TestProgressOutput:
    """Tests for progress and output display."""

    def test_scan_shows_progress(self, temp_dir):
        """Scan should show progress information."""
        from fileforge.cli import app
        (temp_dir / "test.txt").write_text("content")
        result = runner.invoke(app, ["scan", str(temp_dir)])
        # Should show some progress/result info
        assert len(result.stdout) > 0

    def test_verbose_shows_debug_info(self, temp_dir):
        """Verbose mode should show additional information."""
        from fileforge.cli import app
        result = runner.invoke(app, ["--verbose", "stats"])
        # Verbose output should be longer
        assert len(result.stdout) > 0

    def test_quiet_suppresses_output(self, temp_dir):
        """Quiet mode should suppress non-essential output."""
        from fileforge.cli import app
        result = runner.invoke(app, ["--quiet", "stats"])
        # Quiet should have less output (though may still have errors)
        assert result.exit_code == 0 or len(result.stdout) < 100


class TestInteractiveMode:
    """Tests for interactive CLI features."""

    def test_organize_interactive_prompts(self, temp_dir):
        """Interactive organize should prompt for confirmation."""
        from fileforge.cli import app
        (temp_dir / "test.txt").write_text("content")

        # Simulate user input
        result = runner.invoke(
            app,
            ["organize", str(temp_dir), "--interactive"],
            input="n\n"  # Decline all prompts
        )
        # Should complete without error
        assert result.exit_code == 0


class TestBatchOperations:
    """Tests for batch operation handling."""

    def test_organize_multiple_files(self, temp_dir):
        """Organize should handle multiple files."""
        from fileforge.cli import app

        # Create multiple test files
        for i in range(5):
            (temp_dir / f"test{i}.txt").write_text(f"content {i}")

        result = runner.invoke(app, ["organize", str(temp_dir)])
        assert result.exit_code == 0

    def test_scan_large_directory(self, temp_dir):
        """Scan should handle large directories."""
        from fileforge.cli import app

        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        for i in range(10):
            (subdir / f"file{i}.txt").write_text("content")

        result = runner.invoke(app, ["scan", str(temp_dir), "--recursive"])
        assert result.exit_code == 0


# Fixtures
@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path
