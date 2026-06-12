from importlib.metadata import PackageNotFoundError, version

from lume.actions import Action, ReadOnlyActionMixin, WritableActionMixin

__all__ = [
    "Action",
    "ReadOnlyActionMixin",
    "WritableActionMixin",
]

try:
    __version__ = version("lume-base")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0+unknown"
