"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.

"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
from enum import Enum
import warnings

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, model_validator, ConfigDict


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

    """

    model_config = ConfigDict(use_enum_values=True)

    name: str
    read_only: bool = False
    default_validation_config: ConfigEnum = ConfigEnum.NULL

    @abstractmethod
    def validate_value(self, value: Any, config: ConfigEnum = None):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        config = super().model_dump(**kwargs)
        # Convert enum to its string value for serialization
        if 'default_validation_config' in config:
            config['default_validation_config'] = self.default_validation_config.value
        return {"variable_class": self.__class__.__name__} | config


class ScalarVariable(Variable):
    """Variable for float values.

    Attributes
    ----------
    default_value : float | None
        Default value for the variable.
    value_range : tuple[float, float] | None
        Value range that is considered valid for the variable. If the value range is set to None,
        the variable is interpreted as a constant and values are validated against the default value.
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
            self._validate_value_type(self.default_value)
            if self.value_range is not None:
                if not self._value_is_within_range(self.default_value):
                    raise ValueError(
                        "Default value ({}) is out of valid range: ([{},{}]).".format(
                            self.default_value, *self.value_range
                        )
                    )
        return self

    def validate_value(self, value: float, config: ConfigEnum = None):
        """Validates the given value.

        Parameters
        ----------
        value : float
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
        if isinstance(config, str):
            config = ConfigEnum(config)

        config = self.default_validation_config if config is None else config
        if config != ConfigEnum.NULL:
            self._validate_value_is_within_range(
                value, config=config
            )

    @staticmethod
    def _validate_value_type(value: float):
        if not isinstance(value, (int, float, np.floating)) or isinstance(value, bool):
            raise TypeError(
                f"Expected value to be of type {float} or {np.float64}, "
                f"but received {type(value)}."
            )

    def _validate_value_is_within_range(self, value: float, config: ConfigEnum = None):
        if isinstance(config, str):
            config = ConfigEnum(config)

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


class NDVariable(Variable):
    """Abstract base class for N-dimensional array variables.

    Attributes
    ----------
    shape : Tuple[int, ...]
        Shape of the array (per-sample, excluding batch dims).
    unit : str | None
        Unit associated with the variable.
    num_channels : int | None
        Number of image channels (1 for grayscale, 3 for RGB).
        When set, enables image-specific validation requiring 2D or 3D shapes.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    shape: Tuple[int, ...]
    unit: Optional[str] = None
    num_channels: Optional[int] = None

    @property
    def is_image(self) -> bool:
        """Returns True if this variable represents image data."""
        return self.num_channels is not None

    @abstractmethod
    def _validate_array_type(self, value: Any) -> None:
        """Validates that value is the correct array type (numpy/torch)."""
        pass

    @abstractmethod
    def _validate_dtype(self, value: Any, expected_dtype: Any) -> None:
        """Validates the dtype of the array."""
        pass

    @abstractmethod
    def _get_image_shape_for_validation(self, value: Any) -> Tuple[int, ...]:
        """Returns the image shape based on framework conventions (C,H,W vs H,W,C)."""
        pass

    def _validate_shape(
        self, value: Any, expected_shape: Tuple[int, ...] = None
    ) -> None:
        """Validates that the last N dimensions match expected_shape."""
        if expected_shape is not None:
            actual_shape = tuple(value.shape)
            expected_ndim = len(expected_shape)

            if len(actual_shape) < expected_ndim:
                raise ValueError(
                    f"Expected at least {expected_ndim} dimensions with shape {expected_shape}, "
                    f"got {len(actual_shape)} dimensions with shape {actual_shape}"
                )

            if actual_shape[-expected_ndim:] != expected_shape:
                raise ValueError(
                    f"Expected last {expected_ndim} dimension(s) to be {expected_shape}, "
                    f"got {actual_shape[-expected_ndim:]}"
                )

    def _validate_image_shape(self, value: Any) -> None:
        """Validates image-specific shape constraints."""
        if len(self.shape) not in (2, 3):
            raise ValueError(
                f"Image array expects shape to be 2D or 3D, got {self.shape}."
            )

        image_shape = self._get_image_shape_for_validation(value)

        if len(image_shape) not in (2, 3):
            raise ValueError(
                f"Image dimensions must be 2D or 3D, got {len(image_shape)}D with shape {image_shape}."
            )

    @model_validator(mode="after")
    def validate_image_shape_config(self):
        """Validates that image configuration has valid 2D or 3D shape."""
        if self.is_image and len(self.shape) not in (2, 3):
            raise ValueError(
                f"Image array expects shape to be 2D or 3D, got {len(self.shape)}D shape {self.shape}."
            )
        return self

    def validate_value(self, value: Any, config: str = None):
        config = self.default_validation_config if config is None else config
        if isinstance(config, str):
            config = ConfigEnum(config)

        # Mandatory validation
        self._validate_array_type(value)
        self._validate_shape(value, expected_shape=self.shape)

        # Image-specific validation
        if self.is_image:
            self._validate_image_shape(value)

        # Optional validation
        if config != ConfigEnum.NULL:
            pass  # TODO: implement optional validation


class NumpyNDVariable(NDVariable):
    """Variable for NumPy array data.

    Attributes
    ----------
    default_value : NDArray | None
        Default value for the variable.
    dtype : np.dtype
        Data type of the array. Defaults to np.float64.

    """

    default_value: Optional[NDArray] = None
    dtype: np.dtype = np.float64

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value):
        """Convert dtype to numpy dtype if needed."""
        if not isinstance(value, np.dtype):
            return np.dtype(value)
        return value

    def _validate_array_type(self, value: Any) -> None:
        """Validates that value is a numpy.ndarray."""
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Expected value to be a numpy.ndarray, but received {type(value)}."
            )

    def _validate_dtype(self, value: np.ndarray, expected_dtype: np.dtype) -> None:
        """Validates the dtype of the array."""
        if expected_dtype and value.dtype != expected_dtype:
            raise ValueError(f"Expected dtype {expected_dtype}, got {value.dtype}")

    def _get_image_shape_for_validation(self, value: np.ndarray) -> Tuple[int, ...]:
        """Returns image shape for NumPy (H, W, C) format."""
        expected_ndim = len(self.shape)
        image_shape = value.shape[-expected_ndim:]

        # Validate channel count for NumPy (H, W, C)
        if len(image_shape) == 2 and self.num_channels != 1:
            raise ValueError(
                f"Expected 1 channel for grayscale image, got {self.num_channels}."
            )
        elif len(image_shape) == 3 and image_shape[2] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {image_shape[2]}."
            )

        return image_shape

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self._validate_array_type(self.default_value)
            self._validate_shape(self.default_value, expected_shape=self.shape)
            self._validate_dtype(self.default_value, self.dtype)
            if self.is_image:
                self._validate_image_shape(self.default_value)
        return self

    def validate_value(self, value: np.ndarray, config: str = None):
        super().validate_value(value, config)
        self._validate_dtype(value, self.dtype)
