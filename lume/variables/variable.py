"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.

"""

from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

from pydantic import BaseModel


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
    default_validation_config: ConfigEnum = "none"

    @abstractmethod
    def validate_value(self, value: Any, config: ConfigEnum = None):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        config = super().model_dump(**kwargs)
        return {"variable_class": self.__class__.__name__} | config
