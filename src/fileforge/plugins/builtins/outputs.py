"""Built-in output format writers.

Provides JSON and CSV output formats.
"""

import json
import csv
import pluggy

hookimpl = pluggy.HookimplMarker("fileforge")


class JSONOutput:
    """JSON output format writer.

    Writes processing results to JSON format.
    """

    format_name = 'json'
    name = "JSONOutput"

    def write(self, data, output_path):
        """Write data to JSON file.

        Args:
            data: Data to write (dict or list)
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @hookimpl
    def register_output(self):
        """Register this output format."""
        return [self]


class CSVOutput:
    """CSV output format writer.

    Writes processing results to CSV format.
    """

    format_name = 'csv'
    name = "CSVOutput"

    def write(self, data, output_path):
        """Write data to CSV file.

        Args:
            data: Data to write (list of dicts)
            output_path: Output file path
        """
        if not data:
            return

        # Handle both list of dicts and single dict
        if isinstance(data, dict):
            data = [data]

        # Get all unique keys from all rows
        fieldnames = set()
        for row in data:
            fieldnames.update(row.keys())
        fieldnames = sorted(fieldnames)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    @hookimpl
    def register_output(self):
        """Register this output format."""
        return [self]
