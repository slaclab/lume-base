"""Tests for NDVariable."""

import json

import numpy as np
import pytest

from lume.variables.ndvariable import NDVariable


class TestNDVariable:
    """Tests for NDVariable using NumPy ndarrays."""

    def test_creation(self):
        """Test creating a basic ND variable."""
        var = NDVariable(name="test_array", shape=(3, 4))
        assert var.name == "test_array"
        assert var.shape == (3, 4)
        assert var.default_value is None
        assert var.dtype == np.float64

    def test_creation_with_default_value(self):
        """Test creating with a valid default value."""
        arr = np.ones((3, 4))
        var = NDVariable(name="test", shape=(3, 4), default_value=arr)
        assert np.array_equal(var.default_value, arr)

    def test_validate_numpy_array(self):
        """Test validation accepts numpy arrays with correct shape and dtype."""
        var = NDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float64))
        arr = np.zeros((3, 4), dtype=np.float64)
        var.validate_value(arr, config="error")

    def test_validate_1d_array(self):
        """Test validation of 1-D arrays."""
        var = NDVariable(name="test", shape=(5,))
        arr = np.zeros(5)
        var.validate_value(arr, config="error")

    def test_wrong_shape_raises(self):
        """Test wrong shape raises ValueError."""
        var = NDVariable(name="test", shape=(3, 4))
        arr = np.zeros((3, 5))
        with pytest.raises(ValueError, match="Expected shape"):
            var.validate_value(arr, config="error")

    def test_wrong_dtype_raises(self):
        """Test mismatched dtype raises ValueError."""
        var = NDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float64))
        arr = np.zeros((3, 4), dtype=np.int32)
        with pytest.raises(ValueError, match="Expected dtype.*got"):
            var.validate_value(arr, config="error")

    def test_reject_non_array_types(self):
        """Test that non-ndarray types (e.g. plain lists) are rejected."""
        var = NDVariable(name="test", shape=(3, 4))
        with pytest.raises(TypeError, match="Expected value to be a ndarray"):
            var.validate_value([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    def test_string_dtype_is_coerced(self):
        """Test that passing a dtype string is coerced to np.dtype (round-trip support)."""
        var = NDVariable(name="test", shape=(3,), dtype="float64")
        assert var.dtype == np.dtype("float64")

    def test_invalid_dtype_on_creation(self):
        """Test that passing an invalid dtype raises a TypeError."""
        with pytest.raises((TypeError, Exception)):
            NDVariable(name="test", shape=(3,), dtype="not_a_dtype")

    def test_model_dump(self):
        """Test that model_dump serializes dtype as a string and is JSON-serializable."""
        var = NDVariable(name="test", shape=(3, 4))
        dumped = var.model_dump()
        assert dumped["variable_class"] == "NDVariable"
        assert dumped["dtype"] == "float64"
        assert isinstance(dumped["dtype"], str)

    def test_model_dump_json_serializable(self):
        """Test that model_dump output is fully JSON-serializable."""
        arr = np.ones((3, 4), dtype=np.float64)
        var = NDVariable(name="test", shape=(3, 4), default_value=arr)
        dumped = var.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        assert parsed["dtype"] == "float64"
        assert parsed["default_value"] == arr.tolist()

    def test_model_dump_default_value_as_list(self):
        """Test that default_value is serialized as a nested list."""
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        var = NDVariable(name="test", shape=(3, 4), default_value=arr)
        dumped = var.model_dump()
        assert dumped["default_value"] == arr.tolist()
        assert isinstance(dumped["default_value"], list)

    def test_model_dump_default_value_none(self):
        """Test that None default_value stays None in model_dump."""
        var = NDVariable(name="test", shape=(3, 4))
        dumped = var.model_dump()
        assert dumped["default_value"] is None

    def test_round_trip_serialization(self):
        """Test that a variable can be reconstructed from its serialized form."""
        arr = np.ones((3, 4), dtype=np.float64)
        var = NDVariable(name="test", shape=(3, 4), default_value=arr)
        dumped = var.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        var2 = NDVariable(**parsed)
        assert var2.name == var.name
        assert var2.shape == var.shape
        assert var2.dtype == var.dtype
        assert np.array_equal(var2.default_value, var.default_value)
