from abc import abstractmethod, ABC
from typing import Any
from lume.variables import Variable


class LUMEModel(ABC):
    """
    Abstract base class for creating virtual accelerator models and digital twins.
    Simulation specific implementations should inherit from this class and implement
    the abstract methods. Virtual accelerator models that require multiple simulation
    types can also inherit from this class and utilize composition to manage multiple
    simulators.

    Attributes:
        supported_variables: A dictionary of Variable instances that the model supports.

    Methods:
        get(names: list[str]) -> dict[str, Any]:
            Get measurements/state from the simulator.

        set(values: dict[str, Any]) -> None:
            Set control parameters of the simulator.

        reset() -> None:
            Reset the simulator to its initial state.

        supported_variables() -> dict[str, Variable]:
            Return a dict of variables supported by the model. Keys of this dict
            should be valid keys for get() and set() methods.
    """

    def __init__(self, supported_variables: dict[str, Variable]) -> None:
        """
        Initialize the LUMEModel with supported variables.

        Args:
            supported_variables: A dictionary of Variable instances that the model supports.
        """
        self._supported_variables = supported_variables

    def get(self, names: list[str]) -> dict[str, Any]:
        """
        Get measurements/state from the simulator. Should do the following:
        - validate input names against supported_variables
        - return cached measurements/state for the requested names

        Args:
            names: List of variable names to get from the simulator.

        Returns:
            Dictionary of variable names and their corresponding values.

        """
        # Validate input names
        for name in names:
            if name not in self._supported_variables:
                raise ValueError(f"Variable '{name}' is not supported by the model.")

        return self._get(names)

    @abstractmethod
    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Internal method to get measurements/state from the simulator.
        Should be implemented by subclasses.

        Args:
            names: List of variable names to get from the simulator.
        Returns:
            Dictionary of variable names and their corresponding values.
        """
        pass

    def set(self, values: dict[str, Any]) -> None:
        """
        Set control parameters of the simulator. Should do the following:
        - validate input values against supported_variables
        - set the control parameters of the simulator
        - run the simulator
        - update cached measurements/state

        Args:
            values: Dictionary of variable names and their corresponding values to set in the simulator.

        Returns:
            None
        """
        # Validate input values
        for name in values.keys():
            if name not in self._supported_variables:
                raise ValueError(f"Variable '{name}' is not supported by the model.")
            else:
                variable = self._supported_variables[name]
                if not variable.is_settable:
                    raise ValueError(f"Variable '{name}' is not settable.")
                variable.validate_value(values[name])

        # Set the control parameters of the simulator
        self._set(values)

    @abstractmethod
    def _set(self, values: dict[str, Any]) -> None:
        """
        Internal method to set control parameters of the simulator.
        Should be implemented by subclasses.

        Args:
            values: Dictionary of variable names and their corresponding values to set in the simulator.

        Returns:
            None
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the simulator to its initial state.
        """
        pass

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """
        Return a dict of variables supported by the model. It is expected that
        keys of this dict are valid keys for get() and set() methods.
        """
        return self._supported_variables
