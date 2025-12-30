"""
Export and Rules command implementations for FileForge CLI.
This module contains the backend wiring for export and rules commands.
"""

from pathlib import Path
from typing import Optional
import json
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import fnmatch

def wire_export_json(console, state, load_config, Database, output, pretty):
    """Export database to JSON format."""
    config = load_config(state.config_path)
    db = Database(str(config.database.path))
    db.initialize()

    cursor = db.conn.execute("SELECT * FROM files ORDER BY id")
    files = [dict(row) for row in cursor.fetchall()]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        if pretty:
            json.dump(files, f, indent=2, default=str)
        else:
            json.dump(files, f, default=str)

    db.close()
    console.print(f"[green]✓[/green] Exported {len(files)} files to {output}")


def wire_export_csv(console, state, load_config, Database, output):
    """Export database to CSV format."""
    config = load_config(state.config_path)
    db = Database(str(config.database.path))
    db.initialize()

    cursor = db.conn.execute("SELECT * FROM files ORDER BY id")
    files = [dict(row) for row in cursor.fetchall()]

    if not files:
        console.print("[yellow]No files to export[/yellow]")
        db.close()
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file_path', 'file_type', 'category', 'tags', 'file_hash',
                     'confidence', 'processed_at']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.headerheader()

        for file_data in files:
            row = {k: v for k, v in file_data.items() if k in fieldnames}
            writer.writerow(row)

    db.close()
    console.print(f"[green]✓[/green] Exported {len(files)} files to {output}")


def wire_export_html(console, state, load_config, Database, output, title):
    """Export as HTML photo gallery."""
    config = load_config(state.config_path)
    db = Database(str(config.database.path))
    db.initialize()

    cursor = db.conn.execute("""
        SELECT * FROM files
        WHERE file_type IN ('image/jpeg', 'image/png', 'image/gif', 'image/webp')
        ORDER BY processed_at DESC
    """)
    images = [dict(row) for row in cursor.fetchall()]

    output.mkdir(parents=True, exist_ok=True)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        h1 {{ text-align: center; color: #333; }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }}
        .gallery-item {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .gallery-item img {{ width: 100%; height: 200px; object-fit: cover; }}
        .gallery-item .info {{ padding: 15px; }}
        .gallery-item .info h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #333; }}
        .gallery-item .info p {{ margin: 5px 0; font-size: 12px; color: #666; }}
        .tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }}
        .tag {{ background: #e3f2fd; color: #1976d2; padding: 3px 8px; border-radius: 12px; font-size: 11px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="gallery">
"""

    for img in images:
        path = Path(img['file_path'])
        tags = json.loads(img.get('tags', '[]')) if img.get('tags') else []
        tags_html = ''.join([f'<span class="tag">{tag}</span>' for tag in tags])

        html_content += f"""
        <div class="gallery-item">
            <img src="file://{path}" alt="{path.name}">
            <div class="info">
                <h3>{path.name}</h3>
                <p><strong>Category:</strong> {img.get('category', 'N/A')}</p>
                <p><strong>Type:</strong> {img.get('file_type', 'N/A')}</p>
                <div class="tags">{tags_html}</div>
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    index_path = output / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    db.close()
    console.print(f"[green]✓[/green] Gallery created at {index_path} ({len(images)} images)")


def wire_export_sidecars(console, state, load_config, Database, path):
    """Export metadata as sidecar files (.xmp)."""
    config = load_config(state.config_path)
    db = Database(str(config.database.path))
    db.initialize()

    if path:
        cursor = db.conn.execute(
            "SELECT * FROM files WHERE file_path LIKE ? ORDER BY id",
            (f"{path}%",)
        )
    else:
        cursor = db.conn.execute("SELECT * FROM files ORDER BY id")

    files = [dict(row) for row in cursor.fetchall()]
    count = 0

    for file_data in files:
        file_path = Path(file_data['file_path'])
        if not file_path.exists():
            continue

        xmp_path = file_path.with_suffix(file_path.suffix + '.xmp')

        xmp = ET.Element('x:xmpmeta', {'xmlns:x': 'adobe:ns:meta/'})
        rdf = ET.SubElement(xmp, 'rdf:RDF', {'xmlns:rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'})
        desc = ET.SubElement(rdf, 'rdf:Description', {
            'rdf:about': str(file_path),
            'xmlns:dc': 'http://purl.org/dc/elements/1.1/',
            'xmlns:xmp': 'http://ns.adobe.com/xap/1.0/'
        })

        if file_data.get('tags'):
            tags = json.loads(file_data['tags']) if isinstance(file_data['tags'], str) else file_data['tags']
            if tags:
                subject = ET.SubElement(desc, 'dc:subject')
                bag = ET.SubElement(subject, 'rdf:Bag')
                for tag in tags:
                    ET.SubElement(bag, 'rdf:li').text = str(tag)

        if file_data.get('category'):
            ET.SubElement(desc, 'dc:type').text = file_data['category']

        if file_data.get('summary'):
            ET.SubElement(desc, 'dc:description').text = file_data['summary']

        xml_str = minidom.parseString(ET.tostring(xmp)).toprettyxml(indent='  ')
        with open(xmp_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

        count += 1

    db.close()
    console.print(f"[green]✓[/green] Created {count} sidecar files")


def wire_export_tags(console, state, load_config, Database, path, typer):
    """Write metadata tags directly to files."""
    try:
        import piexif
    except ImportError:
        console.print("[red]Error:[/red] piexif library required for writing EXIF tags")
        console.print("Install with: pip install piexif")
        raise typer.Exit(1)

    from PIL import Image

    config = load_config(state.config_path)
    db = Database(str(config.database.path))
    db.initialize()

    if path:
        cursor = db.conn.execute(
            "SELECT * FROM files WHERE file_path LIKE ? AND file_type LIKE 'image/%' ORDER BY id",
            (f"{path}%",)
        )
    else:
        cursor = db.conn.execute(
            "SELECT * FROM files WHERE file_type LIKE 'image/%' ORDER BY id"
        )

    files = [dict(row) for row in cursor.fetchall()]
    count = 0

    for file_data in files:
        file_path = Path(file_data['file_path'])
        if not file_path.exists():
            continue

        try:
            img = Image.open(file_path)

            try:
                exif_dict = piexif.load(img.info.get('exif', b''))
            except:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

            if file_data.get('tags'):
                tags = json.loads(file_data['tags']) if isinstance(file_data['tags'], str) else file_data['tags']
                if tags:
                    keywords = '; '.join(str(tag) for tag in tags)
                    exif_dict['0th'][piexif.ImageIFD.XPKeywords] = keywords.encode('utf-16le')

            if file_data.get('summary'):
                exif_dict['0th'][piexif.ImageIFD.ImageDescription] = file_data['summary'].encode('utf-8')

            exif_bytes = piexif.dump(exif_dict)
            img.save(file_path, exif=exif_bytes)
            count += 1
        except Exception:
            continue

    db.close()
    console.print(f"[green]✓[/green] Tags written to {count} files")


def wire_rules_list(console, state, load_config):
    """List all organization rules."""
    config = load_config(state.config_path)
    rules = config.organization.rules if hasattr(config, 'organization') else []

    from rich.table import Table
    table = Table(title="Organization Rules")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Pattern", style="yellow")
    table.add_column("Destination", style="green")
    table.add_column("Conditions", style="blue")

    for idx, rule in enumerate(rules):
        conditions_str = json.dumps(rule.conditions) if rule.conditions else "None"
        table.add_row(
            str(idx),
            rule.name,
            rule.pattern,
            rule.destination,
            conditions_str
        )

    console.print(table)

    if not rules:
        console.print("[yellow]No organization rules defined[/yellow]")


def wire_rules_add(console, state, load_config, Settings, OrganizationRule, OrganizationConfig,
                   name, pattern, destination, conditions):
    """Add a new organization rule."""
    config_path = state.config_path or Path.home() / ".fileforge" / "config.toml"
    config = load_config(config_path)

    conditions_dict = json.loads(conditions) if conditions else None

    new_rule = OrganizationRule(
        name=name,
        pattern=pattern,
        destination=destination,
        conditions=conditions_dict
    )

    if not hasattr(config, 'organization'):
        config_data = config.model_dump()
        config_data['organization'] = OrganizationConfig().model_dump()
        config = Settings(**config_data)

    rules_list = list(config.organization.rules)
    rules_list.append(new_rule)

    config_data = config.model_dump()
    config_data['organization']['rules'] = [r.model_dump() for r in rules_list]
    config = Settings(**config_data)

    config.save_toml(config_path)
    console.print(f"[green]✓[/green] Rule '{name}' added")


def wire_rules_remove(console, state, load_config, Settings, rule_id, typer):
    """Remove an organization rule."""
    config_path = state.config_path or Path.home() / ".fileforge" / "config.toml"
    config = load_config(config_path)

    if not hasattr(config, 'organization') or not config.organization.rules:
        console.print("[yellow]No rules to remove[/yellow]")
        return

    rules_list = list(config.organization.rules)

    if rule_id < 0 or rule_id >= len(rules_list):
        console.print(f"[red]Error:[/red] Invalid rule ID {rule_id}")
        raise typer.Exit(1)

    removed_rule = rules_list.pop(rule_id)

    config_data = config.model_dump()
    config_data['organization']['rules'] = [r.model_dump() for r in rules_list]
    config = Settings(**config_data)

    config.save_toml(config_path)
    console.print(f"[green]✓[/green] Rule '{removed_rule.name}' removed")


def wire_rules_test(console, state, load_config, Database, rule_id, file_path, typer):
    """Test a rule against a specific file."""
    config = load_config(state.config_path)

    if not hasattr(config, 'organization') or not config.organization.rules:
        console.print("[yellow]No rules defined[/yellow]")
        return

    rules_list = list(config.organization.rules)

    if rule_id < 0 or rule_id >= len(rules_list):
        console.print(f"[red]Error:[/red] Invalid rule ID {rule_id}")
        raise typer.Exit(1)

    rule = rules_list[rule_id]

    db_config = load_config(state.config_path)
    db = Database(str(db_config.database.path))
    db.initialize()

    file_data = db.get_file(str(file_path))
    db.close()

    matches_pattern = fnmatch.fnmatch(file_path.name, rule.pattern)

    matches_conditions = True
    if rule.conditions and file_data:
        for key, expected_value in rule.conditions.items():
            actual_value = file_data.get(key)
            if actual_value != expected_value:
                matches_conditions = False
                break

    console.print(f"\n[bold blue]Testing Rule:[/bold blue] {rule.name}")
    console.print(f"[dim]Pattern:[/dim] {rule.pattern}")
    console.print(f"[dim]Destination:[/dim] {rule.destination}")
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"Pattern match: [{'green' if matches_pattern else 'red'}]{matches_pattern}[/]")
    console.print(f"Conditions match: [{'green' if matches_conditions else 'red'}]{matches_conditions}[/]")

    if matches_pattern and matches_conditions:
        console.print(f"\n[green]✓[/green] File would be moved to: {rule.destination}")
    else:
        console.print(f"\n[yellow]File does not match rule[/yellow]")
