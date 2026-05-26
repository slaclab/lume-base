from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Self

from pydantic import BaseModel, model_validator

from lume.variables import Variable
from lume.exceptions import ReadOnlyError

SimulatorT = TypeVar("SimulatorT")


class Action(ABC, BaseModel, Generic[SimulatorT]):
    """Base for read-only actions over a generic simulator type.

    Subclasses must implement ``_get``.  The ``model_validator`` enforces
    that the associated variable is marked read-only at construction time.
    """

    var: Variable

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def read_only(self) -> bool:
        return getattr(self.var, "read_only")

    @model_validator(mode="after")
    def _check_var(self) -> "Action[SimulatorT]":
        if not self.var.read_only:
            raise ReadOnlyError(f"{type(self).__name__} requires a read-only variable")
        return self

    @abstractmethod
    def _get(self, simulator: SimulatorT) -> Any:
        """
        The child-class implementation of the get method. Override this method and
        not `get` for defining the action's functionality.

        Parameters
        ----------
        simulator: SimulatorT
            The simulator object the parameter is pulled from
        """
        ...

    def get(self, simulator: SimulatorT) -> Any:
        """
        Outside facing get method.

        Parameters
        ----------
        simulator: SimulatorT
            The simulator object the parameter is pulled from
        """
        return self._get(simulator)


class WritableAction(Action[SimulatorT], Generic[SimulatorT]):
    """Base for actions that support both get and set.

    Overrides ``_check_var`` so writable variables are accepted.
    Subclasses must implement ``get`` and ``set``.

    The model calls ``set``, which enforces the read-only guard before
    delegating to ``_set``.
    """

    @model_validator(mode="after")
    def _check_var(self) -> Self:
        return self

    @abstractmethod
    def _set(self, simulator: SimulatorT, value: Any) -> None:
        """
        The child-class implementation of the set method. Overrid this method and
        not `set` for defining the action's set method.

        Parameters
        ----------
        simulator: SimulatorT
            The simulator object
        value: Any
            The value the variable associated with the action is being set to
        """
        ...

    def set(self, simulator: SimulatorT, value: Any) -> None:
        """
        Outside facing set method with read-only checking.

        Parameters
        ----------
        simulator: SimulatorT
            The simulator object
        value: Any
            The value the variable associated with the action is being set to
        """
        if self.var.read_only:
            raise ReadOnlyError(f"'{self.name}' is read-only")
        self._set(simulator, value)