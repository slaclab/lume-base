"""Tests for NDVariable and NumpyNDVariable classes."""

import numpy as np
import pytest

from lume.variables.ndvariable import NDVariable, NumpyNDVariable


class TestNDVariable:
    """Test NDVariable with nested lists."""

    def test_creation(self):
        """Test creating a basic ND variable."""
        var = NDVariable(name="test_list", shape=(3, 4))
        assert var.name == "test_list"
        assert var.shape == (3, 4)
        assert var.default_value is None

    def test_validate_nested_list(self):
        """Test validation of nested lists with correct shape."""
        var = NDVariable(name="test", shape=(2, 3))
        data = [[1, 2, 3], [4, 5, 6]]
        var.validate_value(data, config="error")

    def test_validate_1d_list(self):
        """Test validation of 1D flat lists."""
        var = NDVariable(name="test", shape=(5,))
        data = [1, 2, 3, 4, 5]
        var.validate_value(data, config="error")

    def test_batch_dimensions(self):
        """Test that batch dimensions (leading dims) are allowed."""
        var = NDVariable(name="test", shape=(2, 3))
        data = [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ]
        var.validate_value(data, config="error")

    def test_wrong_shape_raises(self):
        """Test that list with wrong shape raises ValueError."""
        var = NDVariable(name="test", shape=(3, 4))
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value(data, config="error")

    def test_ragged_list_raises(self):
        """Test that ragged list raises ValueError."""
        var = NDVariable(name="test", shape=(3, 4))
        data = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10]]
        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            var.validate_value(data, config="error")

    def test_reject_non_list_types(self):
        """Test that non-list types are rejected."""
        var = NDVariable(name="test", shape=(3,))
        with pytest.raises(TypeError, match="Expected value to be a list"):
            var.validate_value(np.array([1, 2, 3]), config="error")


class TestNumpyNDVariable:
    """Test NumpyNDVariable with numpy arrays."""

    def test_creation(self):
        """Test creating a basic numpy ND variable."""
        var = NumpyNDVariable(name="test_array", shape=(3, 4))
        assert var.name == "test_array"
        assert var.shape == (3, 4)
        assert var.dtype == np.float64

    def test_creation_with_default_value(self):
        """Test creating with a valid default value."""
        arr = np.ones((3, 4))
        var = NumpyNDVariable(name="test", shape=(3, 4), default_value=arr)
        assert np.array_equal(var.default_value, arr)

    def test_validate_numpy_array(self):
        """Test validation accepts numpy arrays with correct shape and dtype."""
        var = NumpyNDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float64))
        arr = np.zeros((3, 4), dtype=np.float64)
        var.validate_value(arr, config="error")

    def test_batch_dimensions(self):
        """Test that batch dimensions are allowed."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        arr = np.zeros((10, 3, 4))
        var.validate_value(arr, config="error")

    def test_wrong_shape_raises(self):
        """Test wrong shape raises error."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        arr = np.zeros((3, 5))
        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value(arr, config="error")

    def test_wrong_dtype_raises(self):
        """Test mismatched dtype raises error."""
        var = NumpyNDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float64))
        arr = np.zeros((3, 4), dtype=np.int32)
        with pytest.raises(ValueError, match="Expected dtype.*got"):
            var.validate_value(arr, config="error")

    def test_reject_non_array_types(self):
        """Test validation rejects non-numpy array types."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        with pytest.raises(TypeError, match="Expected value to be a ndarray"):
            var.validate_value([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    def test_invalid_dtype_on_creation(self):
        """Test that invalid dtype type is rejected."""
        with pytest.raises(TypeError, match="dtype must be a dtype instance"):
            NumpyNDVariable(name="test", shape=(3,), dtype="float64")

    def test_model_dump(self):
        """Test that model_dump includes variable_class."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        dumped = var.model_dump()
        assert dumped["variable_class"] == "NumpyNDVariable"
        assert dumped["dtype"] == np.float64
