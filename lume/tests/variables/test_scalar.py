"""Tests for ScalarVariable class."""

import warnings

import numpy as np
import pytest

from lume.variables.scalar import ScalarVariable


class TestScalarVariable:
    """Test ScalarVariable creation and validation."""

    def test_creation(self):
        """Test creating a scalar variable with defaults and custom values."""
        var = ScalarVariable(name="test_var")
        assert var.name == "test_var"
        assert var.default_value is None
        assert var.value_range is None
        assert var.unit is None

        var2 = ScalarVariable(
            name="x",
            default_value=5.0,
            value_range=(0.0, 10.0),
            unit="m",
            read_only=True,
        )
        assert var2.default_value == 5.0
        assert var2.value_range == (0.0, 10.0)
        assert var2.unit == "m"

    def test_value_range_validation(self):
        """Test that value_range must have min <= max."""
        with pytest.raises(ValueError, match="Minimum value.*must be lower or equal"):
            ScalarVariable(name="test", value_range=(10.0, 5.0))

    def test_default_value_must_be_in_range(self):
        """Test that default_value is validated against value_range."""
        with pytest.raises(ValueError, match="out of valid range"):
            ScalarVariable(name="test", default_value=15.0, value_range=(0.0, 10.0))

    def test_validate_numeric_types(self):
        """Test validation accepts float, int, and numpy numeric values."""
        var = ScalarVariable(name="test")
        var.validate_value(5.0)
        var.validate_value(5)
        var.validate_value(np.float64(5.0))

    def test_reject_non_numeric_types(self):
        """Test validation rejects non-numeric values."""
        var = ScalarVariable(name="test")
        for invalid in ["5.0", True, [5.0], None]:
            with pytest.raises(TypeError, match="Expected value to be of type"):
                var.validate_value(invalid)

    def test_range_validation_with_config(self):
        """Test value range validation with different config modes."""
        var = ScalarVariable(name="test", value_range=(0.0, 10.0))

        # Within range passes
        var.validate_value(5.0, config="error")

        # Out of range with 'none' config passes
        var.validate_value(15.0, config="none")

        # Out of range with 'warn' config warns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            var.validate_value(15.0, config="warn")
            assert len(w) == 1
            assert "out of valid range" in str(w[0].message)

        # Out of range with 'error' config raises
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0, config="error")

    def test_model_dump(self):
        """Test that model_dump includes variable_class."""
        var = ScalarVariable(name="x", default_value=5.0, value_range=(0.0, 10.0))
        dumped = var.model_dump()
        assert dumped["variable_class"] == "ScalarVariable"
        assert dumped["name"] == "x"
        assert dumped["default_value"] == 5.0
