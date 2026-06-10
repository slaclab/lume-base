from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

SimulatorT = TypeVar("SimulatorT")


def set_action(simulator: SimulatorT, variable, value: Any):
    """ execute the set action for a variable, if it exists. """
    if variable.read_only:
        raise ValueError(f"Variable {variable.name} is read-only and cannot be set.")
    if variable.action is None:
        raise ValueError(f"Variable {variable.name} does not have an associated action to set its value.")
    variable.action.set(simulator, variable, value)


def get_action(simulator: SimulatorT, variable) -> Any:
    """ execute the get action for a variable, if it exists. """
    if variable.action is None:
        raise ValueError(f"Variable {variable.name} does not have an associated action to get its value.")
    return variable.action.get(simulator, variable)


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
