"""N-dimensional array variable definitions for LUME-model variables.

This module provides abstract and concrete variable classes for handling
N-dimensional array data, with support for NumPy arrays and potential
extensibility for other array types (e.g., PyTorch tensors).

"""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator, model_validator

from lume.variables.variable import ConfigEnum, Variable


class NDVariable(Variable):
    """Abstract base class for N-dimensional array variables.

    This class provides a framework for validating array-like data with specific
    shape and dtype requirements. It supports batch dimensions (leading dimensions)
    while enforcing constraints on the trailing dimensions.

    Attributes
    ----------
    shape : Tuple[int, ...]
        Expected shape of the array (per-sample, excluding batch dimensions).
        The last N dimensions of any value must match this shape.
    dtype : Any
        Expected data type of the array. The specific type depends on the
        subclass implementation (e.g., np.dtype for NumPy arrays).
        Subclasses should override this with appropriate type annotations.
    default_value : Any | None
        Default value for the variable. Must match the expected shape and dtype
        if provided. Defaults to None.
    unit : str | None
        Physical unit associated with the variable (e.g., "m", "GeV", "rad").
        Defaults to None.

    Notes
    -----
    Subclasses must override:
    - dtype: With the appropriate type annotation for their array implementation
      (e.g., `dtype: np.dtype = np.float64` for NumPy arrays)
    - array_type: The expected array class (e.g., `np.ndarray` for NumPy arrays)
    - dtype_attribute: The attribute name to access dtype (default: "dtype")

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    shape: Tuple[int, ...]
    dtype: Any = None
    default_value: Optional[Any] = None
    unit: Optional[str] = None

    # Class attribute that subclasses must override
    array_type: type = None  # e.g., np.ndarray, torch.Tensor
    dtype_attribute: str = "dtype"  # Attribute name to access dtype on array instances

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype_field(cls, value: Any) -> Any:
        """Validate that dtype matches the expected type from the annotation.

        This method checks that the provided dtype value is an instance of
        the type specified in the subclass's dtype field annotation.

        Parameters
        ----------
        value : Any
            The dtype value to validate.

        Returns
        -------
        Any
            The validated dtype.

        Raises
        ------
        TypeError
            If the value is not an instance of the expected dtype type.

        Notes
        -----
        Subclasses should annotate their dtype field with the appropriate type
        (e.g., np.dtype for NumPy arrays). This validator will enforce that
        only values of that exact type are accepted.

        """
        # Get the expected type from the field annotation
        if hasattr(cls, "__annotations__") and "dtype" in cls.__annotations__:
            expected_type = cls.__annotations__["dtype"]

            # Handle Optional types (unwrap Optional to get the actual type)
            import typing

            if (
                hasattr(typing, "get_origin")
                and typing.get_origin(expected_type) is typing.Union
            ):
                # For Optional[T] which is Union[T, None], get the non-None type
                args = typing.get_args(expected_type)
                expected_type = next(
                    (arg for arg in args if arg is not type(None)), expected_type
                )

            # Check if value is an instance of the expected type
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"dtype must be a {expected_type.__name__} instance, "
                    f"got {type(value).__name__}. "
                    f"Received value: {repr(value)}"
                )

        return value

    def _validate_array_type(self, value: Any) -> None:
        """Validate that value is the correct array type.

        This method checks that the value is an instance of the array type
        specified in the subclass's array_type class attribute.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value is not of the expected array type.

        Notes
        -----
        Subclasses must set the array_type class attribute to the expected
        array class (e.g., np.ndarray for NumPy arrays).

        """
        if self.array_type is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set the 'array_type' class attribute"
            )

        if not isinstance(value, self.array_type):
            raise TypeError(
                f"Expected value to be a {self.array_type.__name__}, "
                f"but received {type(value).__name__}."
            )

    def _validate_dtype(self, value: Any, expected_dtype: Any) -> None:
        """Validate the dtype of the array.

        This method checks that the array's dtype matches the expected dtype.
        It accesses the dtype using the attribute name specified in the
        dtype_attribute class attribute.

        Parameters
        ----------
        value : Any
            The array whose dtype should be validated.
        expected_dtype : Any
            The expected data type.

        Raises
        ------
        ValueError
            If the array's dtype does not match the expected dtype.

        Notes
        -----
        Subclasses can override dtype_attribute if their array type uses
        a different attribute name to access dtype (default is "dtype").
        The dtype must match exactly. No implicit type conversions are performed.

        """
        if expected_dtype is None:
            return

        # Get the actual dtype from the array value
        if not hasattr(value, self.dtype_attribute):
            raise AttributeError(
                f"Array value does not have a '{self.dtype_attribute}' attribute"
            )

        actual_dtype = getattr(value, self.dtype_attribute)

        if actual_dtype != expected_dtype:
            raise ValueError(f"Expected dtype {expected_dtype}, got {actual_dtype}")

    def _validate_shape(
        self, value: Any, expected_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """Validate that the trailing dimensions match expected_shape.

        This method allows for batch dimensions (leading dimensions) while
        enforcing that the trailing N dimensions match the expected shape.

        Parameters
        ----------
        value : Any
            The array whose shape should be validated.
        expected_shape : Tuple[int, ...] | None, optional
            The expected shape for the trailing dimensions. If None,
            no shape validation is performed.

        Raises
        ------
        ValueError
            If the trailing dimensions do not match the expected shape.

        Examples
        --------
        >>> # If expected_shape is (3, 4)
        >>> # Valid: (3, 4), (10, 3, 4), (5, 10, 3, 4)
        >>> # Invalid: (3, 5), (10, 3, 5), (4, 3)

        """
        if expected_shape is not None:
            actual_shape = tuple(value.shape)
            expected_ndim = len(expected_shape)

            if actual_shape[-expected_ndim:] != expected_shape:
                raise ValueError(
                    f"Expected last {expected_ndim} dimension(s) to be {expected_shape}, "
                    f"got {actual_shape[-expected_ndim:]}"
                )

    @model_validator(mode="after")
    def validate_default_value(self) -> "NDVariable":
        """Validate the default_value if provided.

        Returns
        -------
        NDVariable
            The validated instance.

        Raises
        ------
        TypeError, ValueError
            If default_value fails validation (wrong type, dtype, or shape).

        """
        if self.default_value is not None:
            self.validate_value(self.default_value, ConfigEnum.ERROR)
        return self

    def validate_value(self, value: Any, config: Optional[ConfigEnum] = None) -> None:
        """Validate the given array value.

        Performs mandatory validation (type, dtype, shape) and optional
        validation based on the configuration.

        Parameters
        ----------
        value : Any
            The array value to validate.
        config : ConfigEnum | None, optional
            The validation configuration. Defaults to None (uses default_validation_config).
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not of the expected array type.
        ValueError
            If the value's dtype or shape does not match expectations.

        """
        # Mandatory validation
        self._validate_array_type(value)
        self._validate_dtype(value, self.dtype)
        self._validate_shape(value, expected_shape=self.shape)

        # Optional validation
        config = self._validation_config_as_enum(config)

        if config != ConfigEnum.NULL:
            pass  # implement optional validation if needed (e.g., value ranges)


class NumpyNDVariable(NDVariable):
    """Variable for NumPy N-dimensional array data.

    This concrete implementation of NDVariable provides validation for
    NumPy arrays with specific shape and dtype requirements.

    Attributes
    ----------
    default_value : NDArray | None
        Default NumPy array value for the variable. Must match the expected
        shape and dtype if provided. Defaults to None.
    dtype : np.dtype
        Expected NumPy data type of the array. Defaults to np.float64.

    Examples
    --------
    >>> import numpy as np
    >>> from lume.variables.ndvariable import NumpyNDVariable
    >>>
    >>> # Create a variable for 2D arrays with shape (3, 4)
    >>> var = NumpyNDVariable(
    ...     name="my_array",
    ...     shape=(3, 4),
    ...     dtype=np.float64,
    ...     unit="m"
    ... )
    >>>
    >>> # Validate a matching array
    >>> arr = np.random.rand(3, 4)
    >>> var.validate_value(arr, config="error")  # Passes
    >>>
    >>> # Validate a batched array (10 samples of shape (3, 4))
    >>> batched_arr = np.random.rand(10, 3, 4)
    >>> var.validate_value(batched_arr, config="error")  # Passes

    """

    default_value: Optional[NDArray] = None
    dtype: np.dtype = np.float64
    array_type: type = np.ndarray
