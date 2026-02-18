"""Tests for ScalarVariable class."""

import warnings

import numpy as np
import pytest

from lume.variables.scalar import ScalarVariable


class TestScalarVariableCreation:
    """Test ScalarVariable instantiation and validation."""

    def test_basic_creation(self):
        """Test creating a basic scalar variable."""
        var = ScalarVariable(name="test_var")
        assert var.name == "test_var"
        assert var.read_only is False
        assert var.default_value is None
        assert var.value_range is None
        assert var.unit is None

    def test_creation_with_all_attributes(self):
        """Test creating a scalar variable with all attributes."""
        var = ScalarVariable(
            name="x",
            default_value=5.0,
            value_range=(0.0, 10.0),
            unit="m",
            read_only=True,
            default_validation_config="warn",
        )
        assert var.name == "x"
        assert var.default_value == 5.0
        assert var.value_range == (0.0, 10.0)
        assert var.unit == "m"
        assert var.read_only is True
        assert var.default_validation_config == "warn"

    def test_value_range_validation_on_creation(self):
        """Test that value_range must have min <= max."""
        with pytest.raises(
            ValueError, match="Minimum value.*must be lower or equal than maximum"
        ):
            ScalarVariable(name="test", value_range=(10.0, 5.0))

    def test_value_range_equal_bounds(self):
        """Test that value_range can have equal min and max (constant)."""
        var = ScalarVariable(name="const", value_range=(5.0, 5.0))
        assert var.value_range == (5.0, 5.0)

    def test_default_value_validation(self):
        """Test that default_value is validated against value_range."""
        # Should raise error if default_value is out of range
        with pytest.raises(ValueError, match="out of valid range"):
            ScalarVariable(name="test", default_value=15.0, value_range=(0.0, 10.0))

    def test_default_value_within_range(self):
        """Test that default_value within range is accepted."""
        var = ScalarVariable(name="test", default_value=5.0, value_range=(0.0, 10.0))
        assert var.default_value == 5.0


class TestScalarVariableValidation:
    """Test value validation functionality."""

    def test_validate_float_value(self):
        """Test validation accepts float values."""
        var = ScalarVariable(name="test")
        var.validate_value(5.0)  # Should not raise
        var.validate_value(0.0)
        var.validate_value(-10.5)

    def test_validate_int_value(self):
        """Test validation accepts int values."""
        var = ScalarVariable(name="test")
        var.validate_value(5)  # Should not raise
        var.validate_value(0)
        var.validate_value(-10)

    def test_validate_numpy_float(self):
        """Test validation accepts numpy float values."""
        var = ScalarVariable(name="test")
        var.validate_value(np.float64(5.0))
        var.validate_value(np.float32(5.0))

    def test_reject_string(self):
        """Test validation rejects string values."""
        var = ScalarVariable(name="test")
        with pytest.raises(TypeError, match="Expected value to be of type"):
            var.validate_value("5.0")

    def test_reject_bool(self):
        """Test validation rejects boolean values."""
        var = ScalarVariable(name="test")
        with pytest.raises(TypeError, match="Expected value to be of type"):
            var.validate_value(True)

    def test_reject_list(self):
        """Test validation rejects list values."""
        var = ScalarVariable(name="test")
        with pytest.raises(TypeError, match="Expected value to be of type"):
            var.validate_value([5.0])

    def test_reject_none(self):
        """Test validation rejects None values."""
        var = ScalarVariable(name="test")
        with pytest.raises(TypeError, match="Expected value to be of type"):
            var.validate_value(None)


class TestScalarVariableRangeValidation:
    """Test value range validation with different configs."""

    def test_value_within_range_no_config(self):
        """Test value within range passes with default config."""
        var = ScalarVariable(name="test", value_range=(0.0, 10.0))
        var.validate_value(5.0)  # Should not raise

    def test_value_at_boundaries(self):
        """Test values at range boundaries are accepted."""
        var = ScalarVariable(name="test", value_range=(0.0, 10.0))
        var.validate_value(0.0, config="error")
        var.validate_value(10.0, config="error")

    def test_value_out_of_range_null_config(self):
        """Test out-of-range value passes with 'none' config."""
        var = ScalarVariable(name="test", value_range=(0.0, 10.0))
        var.validate_value(15.0, config="none")  # Should not raise

    def test_value_out_of_range_warn_config(self):
        """Test out-of-range value warns with 'warn' config."""
        var = ScalarVariable(name="test", value_range=(0.0, 10.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            var.validate_value(15.0, config="warn")
            assert len(w) == 1
            assert "out of valid range" in str(w[0].message)

    def test_value_out_of_range_error_config(self):
        """Test out-of-range value raises error with 'error' config."""
        var = ScalarVariable(name="test", value_range=(0.0, 10.0))
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0, config="error")

    def test_value_below_range(self):
        """Test value below range is caught."""
        var = ScalarVariable(name="test", value_range=(0.0, 10.0))
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(-1.0, config="error")

    def test_value_above_range(self):
        """Test value above range is caught."""
        var = ScalarVariable(name="test", value_range=(0.0, 10.0))
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(11.0, config="error")

    def test_default_validation_config_used(self):
        """Test that default_validation_config is used when config is None."""
        var = ScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="error"
        )
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0)  # No config specified, should use default

    def test_no_range_allows_any_value(self):
        """Test that variables without range accept any value."""
        var = ScalarVariable(name="test")
        var.validate_value(-1e10, config="error")
        var.validate_value(1e10, config="error")


class TestScalarVariableModelDump:
    """Test model serialization."""

    def test_model_dump_includes_variable_class(self):
        """Test that model_dump includes variable_class."""
        var = ScalarVariable(name="test")
        dumped = var.model_dump()
        assert dumped["variable_class"] == "ScalarVariable"

    def test_model_dump_includes_all_fields(self):
        """Test that all fields are included in model_dump."""
        var = ScalarVariable(
            name="x", default_value=5.0, value_range=(0.0, 10.0), unit="m"
        )
        dumped = var.model_dump()
        assert dumped["name"] == "x"
        assert dumped["default_value"] == 5.0
        assert dumped["value_range"] == (0.0, 10.0)
        assert dumped["unit"] == "m"
