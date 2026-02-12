"""
Module that creates a variable type for openpmd-beamphysics particle groups.
"""
import warnings
from typing import Any

from lume.variables.variable import ConfigEnum, Variable
from pmd_beamphysics import ParticleGroup

class ParticleGroupVariable(Variable):
    def validate_value(self, value: Any, config: ConfigEnum = ConfigEnum.ERROR):
        """
        Validates the given value.

        Attributes
        ----------
        value: Any
            The value to be validated.
        config: ConfigEnum, optional
            The configuration for validation. Defaults to ConfigEnum.ERROR.
            Allowed values are "none", "warn", and "error".
        """
        if not isinstance(value, ParticleGroup):
            message = f"Value must be of type ParticleGroup, but got {type(value)}."
            if config == ConfigEnum.WARN:
                warnings.warn(message)
            elif config == ConfigEnum.ERROR:
                raise TypeError(message)
