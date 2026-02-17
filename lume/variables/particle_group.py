"""
Module that creates a variable type for openpmd-beamphysics particle groups.
"""

from typing import Any

from lume.variables.variable import Variable
from pmd_beamphysics import ParticleGroup


class ParticleGroupVariable(Variable):
    def validate_value(self, value: Any):
        """
        Validates the given value.

        Attributes
        ----------
        value: Any
            The value to be validated.

        Raises
        ------
        TypeError:
            If the value is not of type ParticleGroup.
        """
        if not isinstance(value, ParticleGroup):
            message = f"Value must be of type ParticleGroup, but got {type(value)}."
            raise TypeError(message)
