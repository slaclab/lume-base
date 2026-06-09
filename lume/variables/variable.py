"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.

"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from lume.actions import Action


class ConfigEnum(str, Enum):
    """Enum for configuration options during validation."""

    NULL = "none"
    WARN = "warn"
    ERROR = "error"


class Variable(BaseModel, ABC):
    """Abstract variable base class.

    Attributes
    ----------
    name : str
        Name of the variable.
    read_only : bool
        Flag indicating whether the variable can be set.
    default_validation_config : ConfigEnum
        Default validation configuration to use when validating values.
        Valid options are "none" (no validation), "warn" (warn on invalid values),
        or "error" (raise error on invalid values). Defaults to "none".
    action: Action[SimulatorT]
        The action associated with the variable, which defines how to get and set the variable's value in a simulator.
        
    """

    # store/serialize as string
    model_config = ConfigDict(use_enum_values=True)

    name: str
    read_only: bool = False
    default_validation_config: ConfigEnum = "none"
    action: Action | None = None

    def _validation_config_as_enum(self, config: ConfigEnum = None) -> ConfigEnum:
        """Convert validation config to enum type.

        Parameters
        ----------
        config : ConfigEnum, optional
            The configuration for validation. If None, uses default_validation_config.

        Returns
        -------
        ConfigEnum
            The config as a ConfigEnum instance.
        """
        if config is None:
            config = self.default_validation_config
        if isinstance(config, str):
            config = ConfigEnum(config)
        return config
    
    def get(self, simulator: Any) -> Any:
        """Get the value of the variable from the simulator using the associated action.

        Parameters
        ----------
        simulator: Any
            The simulator object to get the variable's value from.

        Returns
        -------
        Any
            The value of the variable obtained from the simulator.
        """
        if self.action is None:
            raise ValueError(f"No action defined for variable '{self.name}'")
        return self.action.get(simulator, self)
    
    def set(self, simulator: Any, value: Any) -> None:
        """Set the value of the variable in the simulator using the associated action.

        Parameters
        ----------
        simulator: Any
            The simulator object to set the variable's value in.
        value: Any
            The value to set for the variable.

        Raises
        ------
        ValueError
            If the variable is read-only or if no action is defined for the variable.
        """
        if self.read_only:
            raise ValueError(f"Variable '{self.name}' is read-only and cannot be set.")
        if self.action is None:
            raise ValueError(f"No action defined for variable '{self.name}'")
        self.action.set(simulator, self, value)

    @abstractmethod
    def validate_value(self, value: Any, config: ConfigEnum = None):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        config = super().model_dump(**kwargs)
        return {"variable_class": self.__class__.__name__} | config
