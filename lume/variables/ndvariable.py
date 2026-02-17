from abc import abstractmethod
from typing import Tuple, Optional, Any

from pydantic import ConfigDict, field_validator, model_validator
import numpy as np
from numpy.typing import NDArray

from lume.variables.variable import Variable, ConfigEnum


class NDVariable(Variable):
    """Abstract base class for N-dimensional array variables.

    Attributes
    ----------
    shape : Tuple[int, ...]
        Shape of the array (per-sample, excluding batch dims).
    unit : str | None
        Unit associated with the variable.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    shape: Tuple[int, ...]
    unit: Optional[str] = None

    @abstractmethod
    def _validate_array_type(self, value: Any) -> None:
        """Validates that value is the correct array type (numpy/torch)."""
        pass

    @abstractmethod
    def _validate_dtype(self, value: Any, expected_dtype: Any) -> None:
        """Validates the dtype of the array."""
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

    def validate_value(self, value: Any, config: str = None):
        config = self.default_validation_config if config is None else config
        if isinstance(config, str):
            config = ConfigEnum(config)

        # Mandatory validation
        self._validate_array_type(value)
        self._validate_shape(value, expected_shape=self.shape)

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

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self._validate_array_type(self.default_value)
            self._validate_shape(self.default_value, expected_shape=self.shape)
            self._validate_dtype(self.default_value, self.dtype)
        return self

    def validate_value(self, value: np.ndarray, config: str = None):
        super().validate_value(value, config)
        self._validate_dtype(value, self.dtype)
