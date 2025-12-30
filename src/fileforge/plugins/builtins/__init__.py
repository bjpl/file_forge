"""Built-in FileForge plugins.

Default plugins shipped with FileForge.
"""

from fileforge.plugins.builtins.classifier import DefaultClassifier
from fileforge.plugins.builtins.namer import DefaultNamer
from fileforge.plugins.builtins.outputs import JSONOutput, CSVOutput

__all__ = ['DefaultClassifier', 'DefaultNamer', 'JSONOutput', 'CSVOutput']
