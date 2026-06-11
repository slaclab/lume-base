from typing import TypeVar

import numpy as np

from lume.variables.scalar import ScalarVariable

IntType = int | np.integer  # for isinstance type
IntT = TypeVar("IntT", bound=IntType)


class IntVariable(ScalarVariable[IntT]):
    """Variable for int values.

    Attributes
    ----------
    default_value : int | None
        Default value for the variable.
    value_range : tuple[int, int] | None
        Validate variable is in range [value_range[0], value_range[1]] (inclusive). Ignore if set to `None`.
    unit : str | None
        Unit associated with the variable.
    """

    default_value: int | None = None
    value_range: tuple[int, int] | None = None
    unit: str | None = None

    @staticmethod
    def _validate_value_type(value: IntT):
        if isinstance(value, bool) or not isinstance(value, IntType):
            raise TypeError(
                f"Expected value to be of type {IntType}, but received {type(value)}."
            )
