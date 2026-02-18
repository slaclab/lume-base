"""N-dimensional array variable definitions for LUME-model variables.

This module provides concrete variable classes for handling N-dimensional
array data. The base NDVariable class works directly with nested Python lists,
and the design allows for easy extensibility to support other array types
(e.g., PyTorch tensors) by subclassing NDVariable.

"""

import typing
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator, model_validator

from lume.variables.variable import ConfigEnum, Variable


def _is_list_sequence(value: Any) -> bool:
    """Check if a value is a list structure (nested or flat).

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        True if the value is a list, False otherwise.
    """
    return isinstance(value, list)


def _is_nested_sequence(value: Any) -> bool:
    """Check if a value is a nested list structure.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        True if the value is a list containing at least one list element,
        False otherwise.
    """
    if not isinstance(value, list):
        return False
    return len(value) > 0 and isinstance(value[0], list)


def _get_nested_shape(value: Any) -> Tuple[int, ...]:
    """Get the shape of a nested list structure.

    Parameters
    ----------
    value : Any
        The nested list to analyze.

    Returns
    -------
    Tuple[int, ...]
        The shape of the nested structure.

    Raises
    ------
    ValueError
        If the nested structure is ragged (inconsistent dimensions).

    Examples
    --------
    >>> _get_nested_shape([[1, 2, 3], [4, 5, 6]])
    (2, 3)
    >>> _get_nested_shape([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    (2, 2, 2)
    """
    if not isinstance(value, list):
        return ()

    if len(value) == 0:
        return (0,)

    first_elem = value[0]
    inner_shape = _get_nested_shape(first_elem)

    for elem in value[1:]:
        elem_shape = _get_nested_shape(elem)
        if elem_shape != inner_shape:
            raise ValueError(
                f"Inconsistent dimensions in nested structure: "
                f"expected {inner_shape}, got {elem_shape}"
            )

    return (len(value),) + inner_shape


class NDVariable(Variable):
    """Base class for N-dimensional array variables.

    This class provides validation for array-like data with specific shape
    requirements. It supports batch dimensions (leading dimensions) while
    enforcing constraints on the trailing dimensions.

    The base implementation works with nested Python lists. Subclasses can
    specialize for specific array types (e.g., NumPy arrays, PyTorch tensors)
    by overriding `array_type`, `dtype`, and `dtype_attribute`.

    Attributes
    ----------
    shape : Tuple[int, ...]
        Expected shape of the array (per-sample, excluding batch dimensions).
        The last N dimensions of any value must match this shape.
    dtype : Any
        Expected data type of the array. For the base class, this is not
        enforced for nested lists. Subclasses should override this with
        appropriate type annotations (e.g., np.dtype for NumPy arrays).
    default_value : Any | None
        Default value for the variable. Must match the expected shape if
        provided. Can be a nested list or array. Defaults to None.
    unit : str | None
        Physical unit associated with the variable (e.g., "m", "GeV", "rad").
        Defaults to None.

    Notes
    -----
    Subclasses should override:
    - array_type: The expected array class (default: list for nested lists)
    - dtype: With the appropriate type annotation for their array implementation
      (e.g., `dtype: np.dtype = np.float64` for NumPy arrays)
    - dtype_attribute: The attribute name to access dtype (default: "dtype")

    Examples
    --------
    >>> from lume.variables.ndvariable import NDVariable
    >>>
    >>> # Create a variable for 2D nested lists with shape (3, 4)
    >>> var = NDVariable(name="my_list", shape=(3, 4))
    >>>
    >>> # Validate a matching nested list
    >>> data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    >>> var.validate_value(data, config="error")  # Passes

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    shape: Tuple[int, ...]
    dtype: Any = None
    default_value: Optional[Any] = None
    unit: Optional[str] = None

    # Class attributes - subclasses can override
    array_type: ClassVar[type] = (
        list  # Default to nested lists; override for np.ndarray, torch.Tensor, etc.
    )
    dtype_attribute: ClassVar[str] = (
        "dtype"  # Attribute name to access dtype on array instances
    )

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
            if (
                hasattr(typing, "get_origin")
                and typing.get_origin(expected_type) is typing.Union
            ):
                # For Optional[T] which is Union[T, None], get the non-None type
                args = typing.get_args(expected_type)
                expected_type = next(
                    (arg for arg in args if arg is not type(None)), expected_type
                )

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
        specified in the class's array_type attribute.

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
        The base NDVariable class accepts nested lists (array_type=list).
        Subclasses like NumpyNDVariable only accept their specific array type
        (e.g., np.ndarray) and will reject nested lists.

        """
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

        For nested list structures, dtype validation is skipped as these
        structures don't have a formal dtype attribute.

        """
        if expected_dtype is None:
            return

        if _is_nested_sequence(value):
            return

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

        For nested list structures, the shape is computed recursively.

        Parameters
        ----------
        value : Any
            The array or nested list whose shape should be validated.
        expected_shape : Tuple[int, ...] | None, optional
            The expected shape for the trailing dimensions. If None,
            no shape validation is performed.

        Raises
        ------
        ValueError
            If the trailing dimensions do not match the expected shape, or if
            a nested list structure has inconsistent dimensions (ragged arrays).

        Examples
        --------
        >>> # If expected_shape is (3, 4)
        >>> # Valid: (3, 4), (10, 3, 4), (5, 10, 3, 4)
        >>> # Invalid: (3, 5), (10, 3, 5), (4, 3)

        """
        if expected_shape is not None:
            if _is_list_sequence(value):
                # For lists (nested or flat), compute shape recursively
                actual_shape = _get_nested_shape(value)
            else:
                # For arrays, use the .shape attribute
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
            The array value to validate. Can be an array (e.g., np.ndarray) or
            a nested list structure. Nested lists must have consistent
            dimensions (non-ragged).
        config : ConfigEnum | None, optional
            The validation configuration. Defaults to None (uses default_validation_config).
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is neither an array of the expected type nor a nested list.
        ValueError
            If the value's dtype or shape does not match expectations, or if
            a nested list has inconsistent dimensions (ragged arrays).

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
    array_type: ClassVar[type] = np.ndarray
