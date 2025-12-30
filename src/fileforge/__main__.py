"""
FileForge __main__ module.

Enables running FileForge as a module using:
    python -m fileforge <command> [options]

This is equivalent to using the fileforge command directly.
"""

from .cli import cli

if __name__ == "__main__":
    cli()
