"""poplar is a package for modelling selection biases with machine learning, and performing hyperparameter estimation.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"