from importlib.metadata import PackageNotFoundError, version

from lume.actions import Action, ActionModel, ReadOnlyActionMixin, WritableActionMixin

__all__ = [
    "Action",
    "ActionModel",
    "ReadOnlyActionMixin",
    "WritableActionMixin",
]

try:
    __version__ = version("lume-base")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0+unknown"
