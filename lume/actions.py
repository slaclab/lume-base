from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import field_validator

from lume.exceptions import ReadOnlyError

SimulatorT = TypeVar("SimulatorT")


class Action(ABC, Generic[SimulatorT]):
    """
    Parent class for testing if something is an Action.

    Do not subclass directly. Use ``ReadOnlyActionMixin`` or
    ``WritableActionMixin`` mixed into a ``Variable`` subclass.
    """


class ReadOnlyActionMixin(Action[SimulatorT], Generic[SimulatorT]):
    """Mixin for read-only variable-actions.

    Mix into a ``Variable`` subclass. The variable must be constructed with
    ``read_only=True``; the field validator enforces this at construction time.
    Subclasses must implement ``_get``.
    """

    @field_validator("read_only", mode="after")
    @classmethod
    def _check_read_only(cls, v: bool) -> bool:
        if not v:
            raise ReadOnlyError(f"{cls.__name__} requires read_only=True")
        return v

    @abstractmethod
    def _get(self, simulator: SimulatorT) -> Any:
        """Return the current value from ``simulator``. User-implemented version, not ``get``.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to read from.

        Returns
        -------
        Any
            The current value of this variable in the simulator.
        """
        ...

    def get(self, simulator: SimulatorT) -> Any:
        """Return the current value from ``simulator``. Do not override, put your implementation in _get instead.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to read from.

        Returns
        -------
        Any
            The current value of this variable in the simulator.
        """
        return self._get(simulator)


class WritableActionMixin(Action[SimulatorT], Generic[SimulatorT]):
    """Mixin for writable variable-actions.

    Mix into a ``Variable`` subclass. Subclasses must implement ``_get`` and
    ``_set``.

    Calling ``set`` on a variable with ``read_only=True`` raises
    ``ReadOnlyError`` at runtime.
    """

    @abstractmethod
    def _get(self, simulator: SimulatorT) -> Any:
        """Return the current value from ``simulator``. User implemented version.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to read from.

        Returns
        -------
        Any
            The current value of this variable in the simulator.
        """
        ...

    @abstractmethod
    def _set(self, simulator: SimulatorT, value: Any) -> None:
        """Write ``value`` to ``simulator``. User implemented version.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to write to.
        value : Any
            The value to assign to this variable in the simulator.
        """
        ...

    def get(self, simulator: SimulatorT) -> Any:
        """Return the current value from ``simulator``. Please override _get instead of this.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to read from.

        Returns
        -------
        Any
            The current value of this variable in the simulator.
        """
        return self._get(simulator)

    def set(self, simulator: SimulatorT, value: Any) -> None:
        """Write ``value`` to ``simulator``. Please override _set instead of this.

        Parameters
        ----------
        simulator : SimulatorT
            The simulator to write to.
        value : Any
            The value to assign to this variable in the simulator.

        Raises
        ------
        ReadOnlyError
            If this variable has ``read_only=True``.
        """
        if self.read_only:
            raise ReadOnlyError(f"'{self.name}' is read-only")
        self._set(simulator, value)
