from typing import Generic, TypeVar
import warnings

import numpy as np
from pydantic import field_validator, model_validator

from lume.variables.variable import Variable, ConfigEnum


ScalarType = float | int | np.floating | np.integer  # for isinstance type
ScalarT = TypeVar("ScalarT", bound=ScalarType)


class ScalarVariable(Variable, Generic[ScalarT]):
    """Variable for float values.

    Attributes
    ----------
    default_value : float | None
        Default value for the variable.
    value_range : tuple[float, float] | None
        Validate variable is in range [value_range[0], value_range[1]] (inclusive). Ignore if set to `None`.
    unit : str | None
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
        if self.default_value is not None:
            self.validate_value(self.default_value, ConfigEnum.ERROR)
        return self

    def validate_value(self, value: ScalarT, config: ConfigEnum | None = None):
        """Validates the given value.

        Parameters
        ----------
        value : float | int | np.floating | np.integer
            The value to be validated.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not of type float.
        ValueError
            If the value is out of the valid range or does not match the default value
            for constant variables.

        """
        # mandatory validation
        self._validate_value_type(value)

        # optional validation
        config = self._validation_config_as_enum(config)

        if config != ConfigEnum.NULL:
            self._validate_value_is_within_range(value, config=config)

    @staticmethod
    def _validate_value_type(value: ScalarT):
        if not isinstance(value, ScalarType) or isinstance(value, bool):
            raise TypeError(
                f"Expected value to be of type {ScalarType} or bool, but received {type(value)}."
            )

    def _validate_value_is_within_range(self, value: ScalarT, config: ConfigEnum | None = None):
        config = self._validation_config_as_enum(config)
    
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
            if config == ConfigEnum.WARN:
                warnings.warn(range_warning_message)
            else:
                raise ValueError(error_message)

    def _value_is_within_range(self, value) -> bool:
        value_range = self.value_range or (-np.inf, np.inf)

        is_within_range = value_range[0] <= value <= value_range[1]
        return is_within_range
