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
        simulator: The underlying simulator instance used for the model.
        cached_values: A dictionary to cache measurements/state from the simulator.
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
    def __init__(self, simulator, supported_variables: dict[str, Variable]) -> None:
        self.simulator = simulator
        self.cached_values = {}
        self._supported_variables = supported_variables
        self.reset()

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
        results = {}
        for name in names:
            results[name] = self.cached_values[name]

        return results

    @abstractmethod
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
