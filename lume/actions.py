from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import field_validator

from lume.exceptions import ReadOnlyError
from lume.model import LUMEModel
from lume.variables import Variable

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


class ActionModel(LUMEModel, Generic[SimulatorT]):
    """LUMEModel backed by a collection of action variables.

    Each entry in ``action_variables`` must be an ``Action`` instance (i.e. a
    ``Variable`` subclass that also mixes in ``ReadOnlyActionMixin`` or
    ``WritableActionMixin``). The model delegates ``get`` and ``set`` calls to
    the corresponding action variable.

    Parameters
    ----------
    simulator : SimulatorT
        The simulator object passed through to each action variable's
        ``get`` / ``set`` implementation.
    action_variables : list[Variable]
        The action variables managed by this model.

    Raises
    ------
    ValueError
        If any entry in ``action_variables`` is not an ``Action`` instance.
    """

    def __init__(
        self,
        simulator: SimulatorT,
        action_variables: list[Variable],
    ) -> None:
        self.simulator = simulator
        self._action_variable_by_name: dict[str, Variable] = {}
        for av in action_variables:
            self.register_action_variable(av)

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return dict(self._action_variable_by_name)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {
            name: self._action_variable_by_name[name].get(self.simulator)
            for name in names
        }

    def _set(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            self._action_variable_by_name[name]._set(self.simulator, value)

    def reset(self) -> None:
        self.set(
            {
                av.name: av.default_value
                for av in self._action_variable_by_name.values()
                if isinstance(av, WritableActionMixin) and hasattr(av, "default_value")
            }
        )

    def register_action_variable(self, action_variable: Variable) -> None:
        """Add an action variable to the model, replacing any with the same name.

        Parameters
        ----------
        action_variable : Variable
            The action variable to register. Must be an ``Action`` instance.

        Raises
        ------
        ValueError
            If ``action_variable`` is not an ``Action`` instance.
        """
        if not isinstance(action_variable, Action):
            raise ValueError(
                f"Expected an Action instance, got {type(action_variable).__name__!r}"
            )
        self._action_variable_by_name[action_variable.name] = action_variable

    def unregister_action_variable(self, name: str) -> None:
        """Remove an action variable from the model by name.

        Parameters
        ----------
        name : str
            Name of the action variable to remove.

        Raises
        ------
        KeyError
            If no action variable with the given name is registered.
        """
        if name not in self._action_variable_by_name:
            raise KeyError(f"No action variable named '{name}' is registered")
        del self._action_variable_by_name[name]
