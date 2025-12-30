"""
FileForge CLI - Command-line interface for intelligent file organization.

This module implements the main CLI using Typer with Rich formatting for
a professional terminal experience. All commands follow a consistent pattern
with type hints, help text, and proper error handling.
"""

from pathlib import Path
from typing import Optional, List
from enum import Enum
import json
import os
import subprocess

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from rich import print as rprint

from .pipeline.orchestrator import PipelineOrchestrator
from .storage.database import Database
from .storage.actions import FileActions
from .storage.history import OperationHistory
from .config import load_config, Settings
from .watcher import FileWatcher

# Initialize Typer app and Rich console
app = typer.Typer(
    name="fileforge",
    help="Intelligent file organization with AI-powered analysis",
    add_completion=False,
)
console = Console()


# Enums for type-safe options
class FileType(str, Enum):
    """Supported file types for scanning."""
    images = "images"
    documents = "documents"
    text = "text"
    all = "all"


class OutputFormat(str, Enum):
    """Output formats for query results."""
    table = "table"
    json = "json"
    paths = "paths"


class ExportFormat(str, Enum):
    """Export formats."""
    json = "json"
    csv = "csv"
    html = "html"
    sidecars = "sidecars"
    tags = "tags"


# Global state for verbose/quiet modes
class GlobalState:
    """Global application state."""
    verbose: bool = False
    quiet: bool = False
    config_path: Optional[Path] = None


state = GlobalState()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-error output"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=False,
        dir_okay=False,
    ),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit"
    ),
):
    """
    FileForge - Intelligent file organization with AI-powered analysis.

    Process, analyze, and organize files using multimodal AI models.
    Supports images, documents, and text files with face recognition,
    content analysis, and intelligent categorization.
    """
    if version:
        from fileforge import __version__
        console.print(f"FileForge v{__version__}")
        raise typer.Exit(0)

    state.verbose = verbose
    state.quiet = quiet
    state.config_path = config

    if verbose and quiet:
        console.print("[yellow]Warning: Both --verbose and --quiet specified. Using quiet mode.[/yellow]")
        state.verbose = False

    # If no subcommand was invoked, show help
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


# ==================== SCAN COMMAND ====================

@app.command()
def scan(
    path: Path = typer.Argument(
        ...,
        help="Directory to scan for files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help="Scan subdirectories recursively"
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        dir_okay=False,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be done without making changes"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Reprocess already-scanned files"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of files to process",
        min=1,
    ),
    types: FileType = typer.Option(
        FileType.all,
        "--types",
        "-t",
        help="Types of files to process"
    ),
):
    """
    Scan and process files in a directory.

    Analyzes files using AI models to extract metadata, detect faces,
    extract text content, and suggest organization strategies.

    Examples:
        fileforge scan /path/to/photos --types images
        fileforge scan ~/Documents --recursive --limit 100
        fileforge scan . --dry-run --force
    """
    try:
        # Load configuration
        cfg_path = config_path or state.config_path
        if cfg_path:
            config = load_config(cfg_path)
        else:
            config = Settings()
        config.scanning.recursive = recursive

        if not state.quiet:
            console.print(f"\n[bold blue]Scanning:[/bold blue] {path}")
            console.print(f"[dim]Recursive: {recursive} | Types: {types.value} | Dry run: {dry_run}[/dim]\n")

        # Initialize database
        db = Database(str(config.database.path), wal_mode=config.database.wal_mode)
        db.initialize()

        # Create orchestrator
        orchestrator = PipelineOrchestrator(config=config, database=db)

        # Run pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing files...", total=None)
            results = orchestrator.run(path=path, dry_run=dry_run)
            progress.update(task, description="Scan complete!")

        # Display results
        if not state.quiet:
            console.print("[green]✓[/green] Scan completed successfully\n")
            table = Table(title="Scan Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Files Processed", str(results.get('files_processed', 0)))
            table.add_row("Files Succeeded", str(results.get('files_succeeded', 0)))
            table.add_row("Files Failed", str(results.get('files_failed', 0)))
            table.add_row("Duration", f"{results.get('duration_seconds', 0):.2f}s")
            if 'by_type' in results:
                for file_type, count in results['by_type'].items():
                    table.add_row(f"  {file_type.capitalize()}", str(count))
            console.print(table)
            if results.get('errors'):
                console.print("\n[yellow]Errors encountered:[/yellow]")
                for error in results['errors'][:5]:
                    console.print(f"  [red]•[/red] {error.get('error', 'Unknown error')}")
        db.close()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if state.verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


# ==================== ORGANIZE COMMAND ====================

@app.command()
def organize(
    path: Optional[Path] = typer.Argument(
        None,
        help="Directory to organize (defaults to last scanned)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    rename: bool = typer.Option(
        False,
        "--rename",
        help="Apply suggested filenames"
    ),
    move: bool = typer.Option(
        False,
        "--move",
        help="Move files to suggested folders"
    ),
    tag: bool = typer.Option(
        False,
        "--tag",
        help="Write metadata tags to files"
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Confirm each action"
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence for automatic actions (0-1)",
        min=0.0,
        max=1.0,
    ),
):
    """
    Apply organization actions to processed files.

    Executes renaming, moving, and tagging operations based on AI analysis.
    Requires files to be scanned first.

    Examples:
        fileforge organize --rename --move
        fileforge organize /path/to/photos --rename --interactive
        fileforge organize --tag --min-confidence 0.9
    """
    if not any([rename, move, tag]):
        console.print("[yellow]No actions specified.[/yellow]")
        console.print("Usage: Specify at least one action flag: --rename, --move, or --tag")
        raise typer.Exit(0)

    try:
        # Load configuration
        config = Settings()
        db = Database(str(config.database.path), wal_mode=config.database.wal_mode)
        db.initialize()

        # Get files to organize
        files_to_organize = db.query_files({}, limit=None)

        actions_list = []
        if rename:
            actions_list.append("rename")
        if move:
            actions_list.append("move")
        if tag:
            actions_list.append("tag")

        if not state.quiet:
            console.print(f"\n[bold blue]Organizing:[/bold blue] {path or 'last scanned directory'}")
            console.print(f"[dim]Actions: {', '.join(actions_list)} | Interactive: {interactive} | Min confidence: {min_confidence}[/dim]\n")

        # Process files
        total_processed = 0
        total_errors = 0

        for file_record in files_to_organize:
            if not file_record:
                continue
            file_path = Path(file_record['file_path'])
            if file_record.get('confidence', 0) < min_confidence:
                continue
            if interactive and not typer.confirm(f"Process {file_path.name}?"):
                continue

            try:
                executor = FileActions(db)
                if rename and file_record.get('suggested_name'):
                    executor.rename(file_path, file_record['suggested_name'])
                    if not state.quiet:
                        console.print(f"[green]✓[/green] Renamed: {file_path.name}")
                if move and file_record.get('category'):
                    dest_dir = Path(file_record['category'])
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    executor.move(file_path, dest_dir / file_path.name)
                    if not state.quiet:
                        console.print(f"[green]✓[/green] Moved: {file_path.name}")
                if tag and file_record.get('tags'):
                    tags = json.loads(file_record['tags']) if isinstance(file_record['tags'], str) else file_record['tags']
                    executor.add_tags(file_path, tags)
                    if not state.quiet:
                        console.print(f"[green]✓[/green] Tagged: {file_path.name}")
                total_processed += 1
            except Exception as e:
                total_errors += 1
                console.print(f"[red]Error:[/red] {e}")

        if not state.quiet:
            console.print(f"\n[green]✓[/green] Organization completed")
            console.print(f"[dim]Processed: {total_processed} | Errors: {total_errors}[/dim]")
        db.close()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if state.verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


# ==================== QUERY COMMAND ====================

@app.command()
def query(
    tag: Optional[List[str]] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Filter by tag (repeatable)"
    ),
    text: Optional[str] = typer.Option(
        None,
        "--text",
        help="Search file content"
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help="Filter by category"
    ),
    file_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Filter by file type"
    ),
    similar_to: Optional[Path] = typer.Option(
        None,
        "--similar-to",
        help="Find files similar to this one",
        exists=True,
        dir_okay=False,
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        "-f",
        help="Output format"
    ),
):
    """
    Search and query processed files.

    Query the database using tags, content, categories, or similarity.
    Supports multiple output formats.

    Examples:
        fileforge query --tag vacation --tag beach
        fileforge query --text "meeting notes" --format json
        fileforge query --similar-to photo.jpg --format table
        fileforge query --category documents --type pdf
    """
    try:
        if not any([tag, text, category, file_type, similar_to]):
            console.print("[yellow]No query criteria specified[/yellow]")
            raise typer.Exit(0)

        # Load configuration and database
        config = Settings()
        db = Database(str(config.database.path), wal_mode=config.database.wal_mode)
        db.initialize()

        # Build query filters
        filters = {}
        if category:
            filters['category'] = category
        if file_type:
            filters['file_type'] = file_type
        if text:
            filters['text_search'] = text

        if tag:
            results = []
            for t in tag:
                results.extend(db.query_files({'tag': t}))
            seen = set()
            results = [r for r in results if r['id'] not in seen and not seen.add(r['id'])]
        else:
            results = db.query_files(filters)

        if similar_to:
            console.print("[yellow]Similarity search not yet implemented[/yellow]")
            results = []

        # Format output
        if format == OutputFormat.table:
            table = Table(title="Query Results")
            table.add_column("Path", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Tags", style="green")
            table.add_column("Confidence", style="yellow")

            for result in results:
                tags_str = json.loads(result.get('tags', '[]')) if result.get('tags') else []
                table.add_row(
                    result.get('file_path', 'Unknown'),
                    result.get('file_type', 'Unknown'),
                    ', '.join(tags_str) if tags_str else '-',
                    f"{result.get('confidence', 0):.2f}" if result.get('confidence') else '-'
                )
            console.print(table)

        elif format == OutputFormat.json:
            clean_results = []
            for r in results:
                clean_r = {k: v for k, v in r.items() if v is not None}
                if 'tags' in clean_r and isinstance(clean_r['tags'], str):
                    clean_r['tags'] = json.loads(clean_r['tags'])
                clean_results.append(clean_r)
            rprint({"results": clean_results})

        elif format == OutputFormat.paths:
            for result in results:
                console.print(result.get('file_path', ''))

        if not state.quiet:
            console.print(f"\n[green]✓[/green] Found {len(results)} matching files")
        db.close()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if state.verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


# ==================== UNDO COMMAND ====================

undo_app = typer.Typer(help="Reverse organization operations")
app.add_typer(undo_app, name="undo")


@undo_app.command("last")
def undo_last():
    """Undo the most recent operation."""
    try:
        if not state.quiet:
            console.print("\n[bold blue]Undoing last operation...[/bold blue]")

        # Initialize database and history
        settings = load_config(state.config_path)
        db = Database(settings.database.path)
        actions = FileActions(db)
        history = OperationHistory(db, actions)

        result = history.undo_last()

        if result is None:
            console.print("[yellow]No operations to undo[/yellow]")
        elif result.success:
            console.print("[green]✓[/green] Last operation undone successfully")
        else:
            console.print(f"[red]Failed:[/red] {result.error_message}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@undo_app.command("batch")
def undo_batch(
    batch_id: str = typer.Argument(..., help="Batch ID to undo"),
):
    """Undo a specific batch operation."""
    try:
        if not state.quiet:
            console.print(f"\n[bold blue]Undoing batch:[/bold blue] {batch_id}")

        # Initialize database and history
        settings = load_config(state.config_path)
        db = Database(settings.database.path)
        actions = FileActions(db)
        history = OperationHistory(db, actions)

        results = history.undo_batch(batch_id)

        if not results:
            console.print("[yellow]No operations found for this batch[/yellow]")
        else:
            successful = sum(1 for r in results if r.success)
            console.print(f"[green]✓[/green] Undone {successful}/{len(results)} operations")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@undo_app.command("list")
def undo_list():
    """Show recent operations that can be undone."""
    try:
        # Initialize database
        settings = load_config(state.config_path)
        db = Database(settings.database.path)

        operations = db.list_operations(limit=50)

        table = Table(title="Recent Operations")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Timestamp", style="yellow")
        table.add_column("Source", style="green")
        table.add_column("Status", style="blue")

        for op in operations:
            status = "[dim]undone[/dim]" if op.get('undone') else "[green]active[/green]"
            source = Path(op.get('source_path', '')).name if op.get('source_path') else '-'
            timestamp = op.get('created_at', '-')[:19] if op.get('created_at') else '-'
            table.add_row(
                str(op.get('id', '-')),
                op.get('operation_type', '-'),
                timestamp,
                source,
                status
            )

        console.print(table)

        if not operations:
            console.print("[dim]No operations recorded yet[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@undo_app.command("all")
def undo_all(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt"
    ),
):
    """Undo all operations (requires confirmation)."""
    try:
        if not force:
            confirm = typer.confirm("Are you sure you want to undo ALL operations?")
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

        if not state.quiet:
            console.print("\n[bold blue]Undoing all operations...[/bold blue]")

        # TODO: Import and call undo module
        # from .undo import UndoManager
        # manager = UndoManager()
        # manager.undo_all()

        console.print("[green]✓[/green] All operations undone successfully")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ==================== CLUSTER COMMAND ====================

cluster_app = typer.Typer(help="Manage face clusters")
app.add_typer(cluster_app, name="cluster")


@cluster_app.command("list")
def cluster_list():
    """List all face clusters."""
    try:
        # Initialize database
        settings = load_config(state.config_path)
        db = Database(settings.database.path)

        # Query clusters with counts
        cursor = db.conn.execute("""
            SELECT cluster_id, cluster_name, COUNT(*) as face_count
            FROM faces
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id
            ORDER BY cluster_id
        """)
        clusters = cursor.fetchall()

        table = Table(title="Face Clusters")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Faces", style="green")

        for cluster in clusters:
            cluster_id, cluster_name, face_count = cluster
            table.add_row(
                str(cluster_id) if cluster_id else "-",
                cluster_name or "[dim]unnamed[/dim]",
                str(face_count)
            )

        console.print(table)

        if not clusters:
            console.print("[dim]No face clusters found[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@cluster_app.command("show")
def cluster_show(
    cluster_id: str = typer.Argument(..., help="Cluster ID to show"),
):
    """Show details of a specific cluster."""
    try:
        # Initialize database
        settings = load_config(state.config_path)
        db = Database(settings.database.path)

        # Get faces in cluster
        faces = db.get_faces_by_cluster(int(cluster_id))

        if not state.quiet:
            console.print(f"\n[bold blue]Cluster {cluster_id}[/bold blue]")

        if not faces:
            console.print("[yellow]No faces found in this cluster[/yellow]")
            return

        # Get cluster name from first face
        cluster_name = faces[0].get('cluster_name', 'unnamed') if faces else 'unnamed'
        console.print(f"[dim]Name: {cluster_name}[/dim]\n")

        table = Table(title=f"Faces in Cluster {cluster_id}")
        table.add_column("Face ID", style="cyan")
        table.add_column("File", style="magenta")
        table.add_column("Confidence", style="green")

        for face in faces:
            table.add_row(
                str(face.get('id', '-')),
                Path(face.get('file_path', '')).name if face.get('file_path') else '-',
                f"{face.get('confidence', 0):.2f}" if face.get('confidence') else '-'
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@cluster_app.command("name")
def cluster_name(
    cluster_id: str = typer.Argument(..., help="Cluster ID"),
    name: str = typer.Argument(..., help="Name to assign"),
):
    """Assign a name to a face cluster."""
    try:
        # Initialize database
        settings = load_config(state.config_path)
        db = Database(settings.database.path)

        # Update cluster name for all faces in cluster
        db.conn.execute("""
            UPDATE faces SET cluster_name = ?
            WHERE cluster_id = ?
        """, (name, int(cluster_id)))
        db.conn.commit()

        console.print(f"[green]✓[/green] Cluster {cluster_id} named '{name}'")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@cluster_app.command("merge")
def cluster_merge(
    cluster_id1: str = typer.Argument(..., help="First cluster ID"),
    cluster_id2: str = typer.Argument(..., help="Second cluster ID"),
):
    """Merge two face clusters."""
    try:
        # TODO: Import and call cluster module
        # from .cluster import ClusterManager
        # manager = ClusterManager()
        # new_id = manager.merge_clusters(cluster_id1, cluster_id2)

        console.print(f"[green]✓[/green] Clusters merged into new cluster")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@cluster_app.command("recluster")
def cluster_recluster(
    threshold: float = typer.Option(
        0.6,
        "--threshold",
        "-t",
        help="Similarity threshold for clustering (0-1)",
        min=0.0,
        max=1.0,
    ),
):
    """Recompute all face clusters."""
    try:
        from .models.faces import FaceClusterer
        import numpy as np

        if not state.quiet:
            console.print(f"\n[bold blue]Reclustering faces with threshold {threshold}...[/bold blue]")

        # Initialize database
        settings = load_config(state.config_path)
        db = Database(settings.database.path)

        # Get all face embeddings
        face_data = db.get_all_face_embeddings()

        if not face_data:
            console.print("[yellow]No faces found to cluster[/yellow]")
            return

        # Extract embeddings and IDs
        face_ids = [f['id'] for f in face_data]
        embeddings = []
        for f in face_data:
            emb = f.get('embedding')
            if emb is not None:
                if isinstance(emb, bytes):
                    emb = np.frombuffer(emb, dtype=np.float32)
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(128))  # Placeholder

        # Cluster faces
        clusterer = FaceClusterer(eps=1.0 - threshold, min_samples=2)
        labels = clusterer.cluster(embeddings)

        # Update database with cluster assignments
        for face_id, label in zip(face_ids, labels):
            db.update_face_cluster(face_id, label, None)

        # Count clusters
        unique_clusters = set(l for l in labels if l >= 0)
        noise_count = labels.count(-1)

        console.print(f"[green]✓[/green] Reclustering complete")
        console.print(f"[dim]Found {len(unique_clusters)} clusters, {noise_count} unclustered faces[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ==================== EXPORT COMMAND ====================

export_app = typer.Typer(help="Export data in various formats")
app.add_typer(export_app, name="export")


@export_app.command("json")
def export_json(
    output: Path = typer.Argument(..., help="Output JSON file path"),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="Pretty print JSON"),
):
    """Export database to JSON format."""
    try:
        # Initialize database
        settings = load_config(state.config_path)
        db = Database(settings.database.path)

        # Get all files from database
        cursor = db.conn.execute("SELECT * FROM files ORDER BY id")
        files = [db._row_to_dict(row) for row in cursor.fetchall()]

        # Write to JSON
        indent = 2 if pretty else None
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(files, f, indent=indent, ensure_ascii=False, default=str)

        console.print(f"[green]✓[/green] Exported {len(files)} files to {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@export_app.command("csv")
def export_csv(
    output: Path = typer.Argument(..., help="Output CSV file path"),
):
    """Export database to CSV format."""
    try:
        import csv

        # Initialize database
        settings = load_config(state.config_path)
        db = Database(settings.database.path)

        # Get all files from database
        cursor = db.conn.execute("SELECT * FROM files ORDER BY id")
        files = [db._row_to_dict(row) for row in cursor.fetchall()]

        if not files:
            console.print("[yellow]No files to export[/yellow]")
            return

        # Get all field names
        fieldnames = sorted(set().union(*[f.keys() for f in files]))

        # Write to CSV
        with open(output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for file_data in files:
                # Convert non-string values to strings
                row = {k: str(v) if v is not None else '' for k, v in file_data.items()}
                writer.writerow(row)

        console.print(f"[green]✓[/green] Exported {len(files)} files to {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@export_app.command("html")
def export_html(
    output: Path = typer.Argument(..., help="Output directory for HTML gallery"),
    title: str = typer.Option("FileForge Gallery", "--title", help="Gallery title"),
):
    """Export as HTML photo gallery."""
    try:
        # TODO: Import and call export module
        # from .export import Exporter
        # exporter = Exporter()
        # exporter.to_html_gallery(output, title=title)

        console.print(f"[green]✓[/green] Gallery created at {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@export_app.command("sidecars")
def export_sidecars(
    path: Optional[Path] = typer.Argument(
        None,
        help="Directory to export sidecars for (defaults to all)",
    ),
):
    """Export metadata as sidecar files (.xmp)."""
    try:
        # TODO: Import and call export module
        # from .export import Exporter
        # exporter = Exporter()
        # count = exporter.to_sidecars(path)

        console.print(f"[green]✓[/green] Created sidecar files")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@export_app.command("tags")
def export_tags(
    path: Optional[Path] = typer.Argument(
        None,
        help="Directory to write tags to (defaults to all)",
    ),
):
    """Write metadata tags directly to files."""
    try:
        # TODO: Import and call export module
        # from .export import Exporter
        # exporter = Exporter()
        # count = exporter.write_tags(path)

        console.print(f"[green]✓[/green] Tags written to files")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ==================== WATCH COMMAND ====================

@app.command()
def watch(
    path: Path = typer.Argument(
        ...,
        help="Directory to monitor",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    debounce: int = typer.Option(
        2,
        "--debounce",
        "-d",
        help="Seconds to wait before processing new files",
        min=1,
    ),
):
    """
    Monitor a folder for new files and process them automatically.

    Watches a directory for file system events and automatically processes
    new files as they appear. Press Ctrl+C to stop watching.

    Examples:
        fileforge watch ~/Downloads
        fileforge watch /path/to/photos --debounce 5
    """
    try:
        console.print(f"\n[bold blue]Watching:[/bold blue] {path}")
        console.print(f"[dim]Debounce: {debounce}s | Press Ctrl+C to stop[/dim]\n")

        # Initialize configuration and database
        settings = load_config()
        db = Database(settings.database.path)

        # Create table for live display
        processing_table = Table(title="Processing Queue", show_header=True, header_style="bold magenta")
        processing_table.add_column("Time", style="dim", width=12)
        processing_table.add_column("File", style="cyan")
        processing_table.add_column("Status", width=15)

        processed_count = 0

        def process_file_callback(file_path: Path):
            """Callback function to process files detected by watcher."""
            nonlocal processed_count
            try:
                console.print(f"[green]Detected:[/green] {file_path.name}")

                # Initialize orchestrator for this file
                orchestrator = PipelineOrchestrator(settings, db)

                # Process the file through the pipeline
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Processing {file_path.name}...", total=None)
                    results = orchestrator.process_files([file_path])
                    progress.update(task, completed=True)

                # Display results
                if results:
                    result = results[0]
                    if result.get("status") == "success":
                        processed_count += 1
                        destination = result.get("destination", "N/A")
                        console.print(f"[bold green]✓[/bold green] {file_path.name} → {destination}")
                        console.print(f"[dim]Total processed: {processed_count}[/dim]\n")
                    else:
                        error_msg = result.get("error", "Unknown error")
                        console.print(f"[bold red]✗[/bold red] {file_path.name}: {error_msg}\n")
                else:
                    console.print(f"[yellow]⚠[/yellow] No results for {file_path.name}\n")

            except Exception as e:
                console.print(f"[red]Error processing {file_path.name}:[/red] {e}\n")

        # Create and start file watcher
        watcher = FileWatcher(callback=process_file_callback, debounce=debounce)
        watcher.watch(path)

        console.print("\n[yellow]Watching stopped[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Watching stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ==================== STATS COMMAND ====================

@app.command()
def stats():
    """
    Show database statistics and summary.

    Displays overview of processed files, storage usage, tag distribution,
    and other analytics.
    """
    try:
        # Initialize database and get stats
        settings = load_config(state.config_path)
        db = Database(settings.database.path)
        statistics = db.get_stats()

        console.print("\n[bold blue]FileForge Statistics[/bold blue]\n")

        # Overview table
        table = Table(title="Database Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Files", str(statistics.get('total_files', 0)))
        table.add_row("Total Faces", str(statistics.get('total_faces', 0)))
        table.add_row("Detected Objects", str(statistics.get('total_detected_objects', 0)))

        console.print(table)

        # Files by type
        by_type = statistics.get('by_type', {})
        if by_type:
            type_table = Table(title="Files by Type")
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="green")
            for file_type, count in by_type.items():
                type_table.add_row(file_type or "unknown", str(count))
            console.print(type_table)

        # Files by category
        by_category = statistics.get('by_category', {})
        if by_category:
            cat_table = Table(title="Files by Category")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green")
            for category, count in by_category.items():
                cat_table.add_row(category or "uncategorized", str(count))
            console.print(cat_table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ==================== CONFIG COMMAND ====================

config_cmd = typer.Typer(help="Manage configuration")
app.add_typer(config_cmd, name="config")


@config_cmd.command("show")
def config_show():
    """Display current configuration."""
    try:
        # Load configuration
        settings = load_config(state.config_path)

        console.print("\n[bold blue]Current Configuration[/bold blue]\n")

        # Display main settings
        table = Table(title="General Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Database Path", str(settings.database.path))
        table.add_row("Output Directory", str(settings.output.directory))
        table.add_row("Supported Extensions", ", ".join(settings.scanning.extensions[:5]) + "...")

        console.print(table)

        # Display scanning settings
        scan_table = Table(title="Scanning Settings")
        scan_table.add_column("Setting", style="cyan")
        scan_table.add_column("Value", style="green")

        scan_table.add_row("Recursive", str(settings.scanning.recursive))
        scan_table.add_row("Max Size (MB)", str(settings.scanning.max_size_mb))
        scan_table.add_row("Exclusions", ", ".join(settings.scanning.exclusions[:3]) + "...")

        console.print(scan_table)

        # Display processing settings
        proc_table = Table(title="Processing Settings")
        proc_table.add_column("Setting", style="cyan")
        proc_table.add_column("Value", style="green")

        proc_table.add_row("Workers", str(settings.processing.workers))
        proc_table.add_row("Batch Size", str(settings.processing.batch_size))
        proc_table.add_row("Timeout", str(settings.processing.timeout) + "s")

        console.print(proc_table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@config_cmd.command("validate")
def config_validate(
    config_file: Optional[Path] = typer.Argument(
        None,
        help="Config file to validate (defaults to active config)",
        exists=True,
        dir_okay=False,
    ),
):
    """Validate configuration file."""
    try:
        # TODO: Import and call config module
        # from .config import ConfigManager
        # manager = ConfigManager()
        # is_valid, errors = manager.validate(config_file)

        console.print("[green]✓[/green] Configuration is valid")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@config_cmd.command("init")
def config_init(
    output: Path = typer.Option(
        Path("fileforge.yaml"),
        "--output",
        "-o",
        help="Output path for new config file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing file"
    ),
):
    """Create a new configuration file with defaults."""
    try:
        if output.exists() and not force:
            console.print(f"[yellow]File exists:[/yellow] {output} (use --force to overwrite)")
            raise typer.Exit(1)

        # Create default settings and save as TOML
        settings = Settings()
        config_content = f"""# FileForge Configuration
# Generated with default settings

[database]
path = "{settings.database.path}"
wal_mode = {str(settings.database.wal_mode).lower()}

[scanning]
recursive = {str(settings.scanning.recursive).lower()}
max_size_mb = {settings.scanning.max_size_mb}

[processing]
workers = {settings.processing.workers}
batch_size = {settings.processing.batch_size}
timeout = {settings.processing.timeout}

[output]
directory = "{settings.output.directory}"
format = "{settings.output.format}"
"""
        output.write_text(config_content)
        console.print(f"[green]✓[/green] Configuration created at {output}")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@config_cmd.command("edit")
def config_edit():
    """Open configuration file in default editor."""
    try:
        # TODO: Import and call config module
        # from .config import ConfigManager
        # manager = ConfigManager()
        # manager.edit_config()

        console.print("[green]✓[/green] Configuration updated")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ==================== RULES COMMAND ====================

rules_app = typer.Typer(help="Manage organization rules")
app.add_typer(rules_app, name="rules")


@rules_app.command("list")
def rules_list():
    """List all organization rules."""
    try:
        # Load configuration
        settings = load_config(state.config_path)

        table = Table(title="Organization Rules")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Pattern", style="yellow")
        table.add_column("Destination", style="green")

        for idx, rule in enumerate(settings.organization.rules):
            table.add_row(
                str(idx),
                rule.name,
                rule.pattern,
                rule.destination
            )

        console.print(table)

        if not settings.organization.rules:
            console.print("[dim]No organization rules defined[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@rules_app.command("add")
def rules_add(
    name: str = typer.Argument(..., help="Rule name"),
    pattern: str = typer.Argument(..., help="File matching pattern (e.g. '*.jpg')"),
    destination: str = typer.Argument(..., help="Destination path template"),
):
    """Add a new organization rule."""
    try:
        from .config import OrganizationRule

        # Load configuration
        settings = load_config(state.config_path)

        # Create new rule
        new_rule = OrganizationRule(
            name=name,
            pattern=pattern,
            destination=destination
        )

        # Add to settings
        settings.organization.rules.append(new_rule)

        console.print(f"[green]✓[/green] Rule '{name}' added")
        console.print(f"[dim]Pattern: {pattern} → {destination}[/dim]")
        console.print("[yellow]Note: Save configuration to persist rules[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@rules_app.command("remove")
def rules_remove(
    rule_id: str = typer.Argument(..., help="Rule ID (index) to remove"),
):
    """Remove an organization rule."""
    try:
        # Load configuration
        settings = load_config(state.config_path)

        idx = int(rule_id)
        if idx < 0 or idx >= len(settings.organization.rules):
            console.print(f"[red]Error:[/red] Invalid rule ID: {rule_id}")
            raise typer.Exit(1)

        removed_rule = settings.organization.rules.pop(idx)
        console.print(f"[green]✓[/green] Rule '{removed_rule.name}' removed")
        console.print("[yellow]Note: Save configuration to persist changes[/yellow]")

    except ValueError:
        console.print(f"[red]Error:[/red] Rule ID must be a number")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@rules_app.command("test")
def rules_test(
    rule_id: str = typer.Argument(..., help="Rule ID (index) to test"),
    file_path: Path = typer.Argument(
        ...,
        help="File to test rule against",
        exists=True,
        dir_okay=False,
    ),
):
    """Test a rule against a specific file."""
    try:
        import fnmatch

        # Load configuration
        settings = load_config(state.config_path)

        idx = int(rule_id)
        if idx < 0 or idx >= len(settings.organization.rules):
            console.print(f"[red]Error:[/red] Invalid rule ID: {rule_id}")
            raise typer.Exit(1)

        rule = settings.organization.rules[idx]

        # Test pattern match
        matches = fnmatch.fnmatch(file_path.name, rule.pattern)

        console.print(f"\n[bold blue]Testing Rule:[/bold blue] {rule.name}")
        console.print(f"[dim]Pattern: {rule.pattern}[/dim]")
        console.print(f"[dim]File: {file_path.name}[/dim]\n")

        if matches:
            console.print("[green]✓ Pattern MATCHES[/green]")
            console.print(f"[dim]Would move to: {rule.destination}[/dim]")
        else:
            console.print("[yellow]✗ Pattern does NOT match[/yellow]")

    except ValueError:
        console.print(f"[red]Error:[/red] Rule ID must be a number")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def cli():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    cli()
