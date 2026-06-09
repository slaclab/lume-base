from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Self

from pydantic import BaseModel, model_validator

from lume.exceptions import ReadOnlyError

SimulatorT = TypeVar("SimulatorT")


class Action(ABC, BaseModel, Generic[SimulatorT]):
    """Base for read-only actions over a generic simulator type.

    Subclasses must implement ``_get``.  The ``model_validator`` enforces
    that the associated variable is marked read-only at construction time.
    """
    name: str

    @abstractmethod
    def get(self, simulator: SimulatorT, variable) -> Any:
        """
        Outside facing get method.

        Parameters
        ----------
        simulator: SimulatorT
            The simulator object the parameter is pulled from
        variable: Variable
            A variable object that provides metadata about the variable being accessed, 
            such as its name and validation configuration.
        """
        ...

class WritableAction(Action[SimulatorT], Generic[SimulatorT]):
    """Base for actions that support both get and set."""

    @abstractmethod
    def set(self, simulator: SimulatorT, variable, value: Any) -> None:
        """
        Outside facing set method.

        Parameters
        ----------
        simulator: SimulatorT
            The simulator object
        variable: Variable
            A variable object that provides metadata about the variable being accessed, 
            such as its name and validation configuration.
        value: Any
            The value the variable associated with the action is being set to
        """
        ...
