"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.

"""

from abc import ABC, abstractmethod
from typing import Any
from enum import Enum
import math

import numpy as np
from pydantic import BaseModel, field_validator, model_validator


class ConfigEnum(str, Enum):
    """Enum for configuration options during validation."""

    NULL = "none"
    WARN = "warn"
    ERROR = "error"


class Variable(BaseModel, ABC):
    """Abstract variable base class.

    Attributes
    -----------
    name: str
        Name of the variable.
    read_only: bool
        Flag indicating whether the variable can be set.
    """

    name: str
    read_only: bool = False

    @property
    @abstractmethod
    def default_validation_config(self) -> ConfigEnum:
        """Determines default behavior during validation."""
        return None

    @abstractmethod
    def validate_value(self, value: Any, config: ConfigEnum = None):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        config = super().model_dump(**kwargs)
        return {"variable_class": self.__class__.__name__} | config


class ScalarVariable(Variable):
    """Variable for float values.

    Attributes
    ----------
    default_value: float | None
        Default value for the variable.
    read_only: bool
        Flag indicating whether the variable can be set.
    value_range: tuple[float, float] | None
        Value range that is considered valid for the variable. If the value range is set to None,
        the variable is interpreted as a constant and values are validated against the default value.
    unit: str | None
        Unit associated with the variable.
    """

    default_value: float | None = None
    value_range: tuple[float, float] | None = None
    unit: str | None = None

    @field_validator("value_range", mode="before")
    @classmethod
    def validate_value_range(cls, value):
        if value is not None:
            value = tuple(value)
            if not value[0] <= value[1]:
                raise ValueError(
                    f"Minimum value ({value[0]}) must be lower or equal than maximum ({value[1]})."
                )
        return value

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None and self.value_range is not None:
            if not self._value_is_within_range(self.default_value):
                raise ValueError(
                    "Default value ({}) is out of valid range: ([{},{}]).".format(
                        self.default_value, *self.value_range
                    )
                )
        return self

    @property
    def default_validation_config(self) -> ConfigEnum:
        return "warn"

    def validate_value(self, value: float, config: ConfigEnum = None):
        """
        Validates the given value.

        Args:
            value (float): The value to be validated.
            config (ConfigEnum, optional): The configuration for validation. Defaults to None.
              Allowed values are "none", "warn", and "error".

        Raises:
            TypeError: If the value is not of type float.
            ValueError: If the value is out of the valid range or does not match the default value
              for constant variables.
        """
        _config = self.default_validation_config if config is None else config
        # mandatory validation
        self._validate_value_type(value)
        # optional validation
        if config != "none":
            self._validate_value_is_within_range(value, config=_config)

    @staticmethod
    def _validate_value_type(value: float):
        if not isinstance(value, float):
            raise TypeError(
                f"Expected value to be of type {float} or {np.float64}, "
                f"but received {type(value)}."
            )

    def _validate_value_is_within_range(self, value: float, config: ConfigEnum = None):
        if not self._value_is_within_range(value):
            error_message = (
                "Value ({}) of '{}' is out of valid range: ([{},{}]).".format(
                    value, self.name, *self.value_range
                )
            )
            range_warning_message = (
                error_message
                + " Executing the model outside of the range may result in"
                " unpredictable and invalid predictions."
            )
            if config == "warn":
                print("Warning: " + range_warning_message)
            else:
                raise ValueError(error_message)

    def _value_is_within_range(self, value) -> bool:
        self.value_range = self.value_range or (-np.inf, np.inf)

        is_within_range = self.value_range[0] <= value <= self.value_range[1]
        return is_within_range
