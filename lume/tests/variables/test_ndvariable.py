"""Tests for NDVariable and NumpyNDVariable classes."""

import numpy as np
import pytest

from lume.variables.ndvariable import NDVariable, NumpyNDVariable
from lume.variables.variable import ConfigEnum


# Test fixture: concrete NDVariable subclass for testing
class GenericNDVariable(NDVariable):
    """Generic NDVariable implementation for testing purposes.

    This implementation accepts any array-like object (including lists)
    and doesn't enforce strict dtype validation, making it ideal for
    testing the base NDVariable functionality with lists of lists.
    """

    dtype: type = list
    array_type: type = list
    dtype_attribute: str = "__class__"


class TestNDVariableWithListOfLists:
    """Test NDVariable base class with list of lists (non-NumPy arrays)."""

    def test_basic_creation_generic_ndvariable(self):
        """Test creating a basic generic ND variable."""
        var = GenericNDVariable(name="test_list", shape=(3, 4))
        assert var.name == "test_list"
        assert var.shape == (3, 4)
        assert var.default_value is None

    def test_list_2d_exact_shape(self):
        """Test 2D list of lists with exact shape match."""
        var = GenericNDVariable(name="test", shape=(3, 4))
        data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        var.validate_value(data, config="error")  # Should not raise

    def test_list_1d(self):
        """Test 1D list."""
        var = GenericNDVariable(name="test", shape=(5,))
        data = [1, 2, 3, 4, 5]
        var.validate_value(data, config="error")  # Should not raise

    def test_list_3d(self):
        """Test 3D nested list."""
        var = GenericNDVariable(name="test", shape=(2, 2, 2))
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        var.validate_value(data, config="error")  # Should not raise

    def test_list_with_batch_dimensions(self):
        """Test list of lists with batch dimensions."""
        var = GenericNDVariable(name="test", shape=(2, 3))
        # Shape will be (3, 2, 3) - 3 batches of (2, 3)
        data = [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ]
        var.validate_value(data, config="error")  # Should not raise

    def test_list_wrong_shape_raises(self):
        """Test that list with wrong shape raises ValueError."""
        var = GenericNDVariable(name="test", shape=(3, 4))
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Shape is (3, 3), not (3, 4)
        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value(data, config="error")

    def test_ragged_list_raises(self):
        """Test that ragged list raises ValueError."""
        var = GenericNDVariable(name="test", shape=(3, 4))
        data = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10]]  # Inconsistent inner dimensions
        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            var.validate_value(data, config="error")

    def test_empty_list(self):
        """Test handling of empty lists."""
        var = GenericNDVariable(name="test", shape=(0,))
        data = []
        var.validate_value(data, config="error")  # Should not raise

    def test_list_default_value(self):
        """Test that default_value can be a list of lists."""
        data = [[1, 2, 3], [4, 5, 6]]
        var = GenericNDVariable(name="test", shape=(2, 3), default_value=data)
        assert var.default_value == data

    def test_deeply_nested_list(self):
        """Test deeply nested list (4D)."""
        var = GenericNDVariable(name="test", shape=(2, 2, 2, 2))
        data = [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ]
        var.validate_value(data, config="error")  # Should not raise

    def test_list_with_multiple_batch_dims(self):
        """Test list with multiple batch dimensions."""
        var = GenericNDVariable(name="test", shape=(2, 3))
        # Shape (2, 3, 2, 3) - 2x3 batches of (2, 3)
        data = [
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
                [[13, 14, 15], [16, 17, 18]],
            ],
            [
                [[19, 20, 21], [22, 23, 24]],
                [[25, 26, 27], [28, 29, 30]],
                [[31, 32, 33], [34, 35, 36]],
            ],
        ]
        var.validate_value(data, config="error")  # Should not raise

    def test_reject_non_list_types(self):
        """Test that non-list types are rejected."""
        var = GenericNDVariable(name="test", shape=(3,))

        # NumPy array should be rejected by GenericNDVariable
        with pytest.raises(
            TypeError, match="Expected value to be a list or nested list"
        ):
            var.validate_value(np.array([1, 2, 3]), config="error")

        # Tuple should be rejected
        with pytest.raises(
            TypeError, match="Expected value to be a list or nested list"
        ):
            var.validate_value((1, 2, 3), config="error")

        # Scalar should be rejected
        with pytest.raises(
            TypeError, match="Expected value to be a list or nested list"
        ):
            var.validate_value(42, config="error")

    def test_list_shape_validation_trailing_dims(self):
        """Test that only trailing dimensions must match shape."""
        var = GenericNDVariable(name="test", shape=(2, 3))

        # Exact match
        var.validate_value([[1, 2, 3], [4, 5, 6]], config="error")

        # With batch dimension
        var.validate_value(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], config="error"
        )

        # Wrong trailing dimensions
        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value([[1, 2], [3, 4]], config="error")

        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value([[1, 2, 3, 4], [5, 6, 7, 8]], config="error")

    def test_mixed_type_elements_in_list(self):
        """Test that lists with mixed element types are handled."""
        var = GenericNDVariable(name="test", shape=(2, 3))
        # This should pass validation (shape is correct)
        data = [[1, 2.5, 3], [4, 5, 6.7]]
        var.validate_value(data, config="error")  # Should not raise

    def test_ragged_list_different_depths(self):
        """Test ragged list with different nesting depths."""
        var = GenericNDVariable(name="test", shape=(2, 3))
        # One element is nested, another is not
        data = [[1, 2, 3], 4]
        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            var.validate_value(data, config="error")

    def test_list_of_empty_lists(self):
        """Test list containing empty lists."""
        var = GenericNDVariable(name="test", shape=(3, 0))
        data = [[], [], []]
        var.validate_value(data, config="error")  # Should not raise

    def test_single_element_list(self):
        """Test single element list."""
        var = GenericNDVariable(name="test", shape=(1,))
        data = [42]
        var.validate_value(data, config="error")  # Should not raise

    def test_list_validation_with_none_config(self):
        """Test that validation works with None config."""
        var = GenericNDVariable(name="test", shape=(2, 2))
        data = [[1, 2], [3, 4]]
        var.validate_value(data, config=None)  # Should use default config

    def test_list_validation_preserves_data(self):
        """Test that validation doesn't modify the input list."""
        var = GenericNDVariable(name="test", shape=(2, 2))
        data = [[1, 2], [3, 4]]
        original_data = [[1, 2], [3, 4]]
        var.validate_value(data, config="error")
        assert data == original_data  # Data should be unchanged


class TestNumpyNDVariableCreation:
    """Test NumpyNDVariable instantiation and validation."""

    def test_basic_creation(self):
        """Test creating a basic numpy ND variable."""
        var = NumpyNDVariable(name="test_array", shape=(3, 4))
        assert var.name == "test_array"
        assert var.shape == (3, 4)
        assert var.dtype == np.float64
        assert var.default_value is None
        assert var.unit is None
        assert var.read_only is False

    def test_creation_with_all_attributes(self):
        """Test creating a variable with all attributes."""
        default_arr = np.zeros((3, 4), dtype=np.float32)
        var = NumpyNDVariable(
            name="my_array",
            shape=(3, 4),
            dtype=np.dtype(np.float32),
            default_value=default_arr,
            unit="GeV",
            read_only=True,
            default_validation_config="warn",
        )
        assert var.name == "my_array"
        assert var.shape == (3, 4)
        assert var.dtype == np.dtype(np.float32)
        assert np.array_equal(var.default_value, default_arr)
        assert var.unit == "GeV"
        assert var.read_only is True

    def test_creation_with_1d_shape(self):
        """Test creating a 1D array variable."""
        var = NumpyNDVariable(name="vec", shape=(10,))
        assert var.shape == (10,)

    def test_creation_with_3d_shape(self):
        """Test creating a 3D array variable."""
        var = NumpyNDVariable(name="tensor", shape=(5, 10, 20))
        assert var.shape == (5, 10, 20)

    def test_dtype_validation_accepts_numpy_dtype(self):
        """Test that dtype accepts numpy dtype objects."""
        var = NumpyNDVariable(name="test", shape=(3,), dtype=np.dtype(np.float64))
        assert var.dtype == np.dtype(np.float64)

        var = NumpyNDVariable(name="test", shape=(3,), dtype=np.dtype(np.int32))
        assert var.dtype == np.dtype(np.int32)

    def test_dtype_validation_rejects_invalid_type(self):
        """Test that dtype rejects non-numpy dtype objects."""
        with pytest.raises(TypeError, match="dtype must be a dtype instance"):
            NumpyNDVariable(name="test", shape=(3,), dtype="float64")

        with pytest.raises(TypeError, match="dtype must be a dtype instance"):
            NumpyNDVariable(name="test", shape=(3,), dtype=float)

    def test_default_value_validation(self):
        """Test that default_value is validated on creation."""
        # Invalid shape
        with pytest.raises(ValueError, match="Expected last.*dimension"):
            NumpyNDVariable(name="test", shape=(3, 4), default_value=np.zeros((3, 5)))

        # Invalid dtype
        with pytest.raises(ValueError, match="Expected dtype"):
            NumpyNDVariable(
                name="test",
                shape=(3, 4),
                dtype=np.dtype(np.float64),
                default_value=np.zeros((3, 4), dtype=np.int32),
            )

    def test_default_value_valid(self):
        """Test that valid default_value is accepted."""
        arr = np.ones((3, 4))
        var = NumpyNDVariable(name="test", shape=(3, 4), default_value=arr)
        assert np.array_equal(var.default_value, arr)


class TestNumpyNDVariableArrayTypeValidation:
    """Test array type validation."""

    def test_validate_numpy_array(self):
        """Test validation accepts numpy arrays."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        arr = np.zeros((3, 4))
        var.validate_value(arr, config="error")  # Should not raise

    def test_accept_list(self):
        """Test validation accepts lists of lists with matching shape."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        var.validate_value(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        )  # Should not raise

    def test_reject_tuple(self):
        """Test validation rejects tuples."""
        var = NumpyNDVariable(name="test", shape=(3,))
        with pytest.raises(
            TypeError, match="Expected value to be a ndarray or nested list"
        ):
            var.validate_value((1, 2, 3))

    def test_reject_list_wrong_shape(self):
        """Test validation rejects lists with wrong shape."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_reject_ragged_list(self):
        """Test validation rejects ragged lists (inconsistent dimensions)."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            var.validate_value([[1, 2, 3, 4], [5, 6], [7, 8, 9, 10]])

    def test_reject_scalar(self):
        """Test validation rejects scalar values."""
        var = NumpyNDVariable(name="test", shape=(1,))
        with pytest.raises(
            TypeError, match="Expected value to be a ndarray or nested list"
        ):
            var.validate_value(5.0)

    def test_reject_none(self):
        """Test validation rejects None."""
        var = NumpyNDVariable(name="test", shape=(3,))
        with pytest.raises(
            TypeError, match="Expected value to be a ndarray or nested list"
        ):
            var.validate_value(None)


class TestNumpyNDVariableDtypeValidation:
    """Test dtype validation."""

    def test_matching_dtype_passes(self):
        """Test that matching dtype passes validation."""
        var = NumpyNDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float64))
        arr = np.zeros((3, 4), dtype=np.float64)
        var.validate_value(arr, config="error")  # Should not raise

    def test_mismatched_dtype_raises_error(self):
        """Test that mismatched dtype raises error."""
        var = NumpyNDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float64))
        arr = np.zeros((3, 4), dtype=np.int32)
        with pytest.raises(ValueError, match="Expected dtype.*got"):
            var.validate_value(arr, config="error")

    def test_float32_vs_float64(self):
        """Test that float32 and float64 are distinguished."""
        var = NumpyNDVariable(name="test", shape=(3,), dtype=np.dtype(np.float32))
        arr64 = np.zeros(3, dtype=np.float64)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(arr64, config="error")

    def test_int_dtypes(self):
        """Test integer dtype validation."""
        var = NumpyNDVariable(name="test", shape=(3,), dtype=np.dtype(np.int32))
        arr = np.array([1, 2, 3], dtype=np.int32)
        var.validate_value(arr, config="error")  # Should pass

        arr_wrong = np.array([1, 2, 3], dtype=np.int64)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(arr_wrong, config="error")


class TestNumpyNDVariableShapeValidation:
    """Test shape validation with various cases."""

    def test_exact_shape_match(self):
        """Test exact shape match passes validation."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        arr = np.zeros((3, 4))
        var.validate_value(arr, config="error")  # Should not raise

    def test_wrong_shape_raises_error(self):
        """Test wrong shape raises error."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        arr = np.zeros((3, 5))
        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value(arr, config="error")

    def test_batch_dimension_allowed(self):
        """Test that batch dimensions (leading dims) are allowed."""
        var = NumpyNDVariable(name="test", shape=(3, 4))

        # Single batch dimension
        arr = np.zeros((10, 3, 4))
        var.validate_value(arr, config="error")  # Should pass

        # Multiple batch dimensions
        arr = np.zeros((5, 10, 3, 4))
        var.validate_value(arr, config="error")  # Should pass

    def test_1d_shape_validation(self):
        """Test 1D shape validation."""
        var = NumpyNDVariable(name="test", shape=(10,))

        var.validate_value(np.zeros(10))  # Exact match
        var.validate_value(np.zeros((5, 10)))  # With batch

        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value(np.zeros(11))

    def test_3d_shape_validation(self):
        """Test 3D shape validation."""
        var = NumpyNDVariable(name="test", shape=(2, 3, 4))

        var.validate_value(np.zeros((2, 3, 4)))  # Exact match
        var.validate_value(np.zeros((10, 2, 3, 4)))  # With batch

        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value(np.zeros((2, 3, 5)))

    def test_trailing_dimensions_must_match(self):
        """Test that only trailing dimensions must match shape."""
        var = NumpyNDVariable(name="test", shape=(3, 4))

        # These should pass (correct trailing dims)
        var.validate_value(np.zeros((3, 4)), config="error")
        var.validate_value(np.zeros((10, 3, 4)), config="error")
        var.validate_value(np.zeros((5, 10, 3, 4)), config="error")

        # These should fail (wrong trailing dims)
        with pytest.raises(ValueError):
            var.validate_value(np.zeros((4, 3)), config="error")
        with pytest.raises(ValueError):
            var.validate_value(np.zeros((10, 4, 3)), config="error")


class TestNumpyNDVariableCombinedValidation:
    """Test combined dtype and shape validation."""

    def test_both_dtype_and_shape_correct(self):
        """Test validation passes when both dtype and shape are correct."""
        var = NumpyNDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float32))
        arr = np.zeros((3, 4), dtype=np.float32)
        var.validate_value(arr, config="error")

    def test_correct_shape_wrong_dtype(self):
        """Test validation fails with correct shape but wrong dtype."""
        var = NumpyNDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float32))
        arr = np.zeros((3, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(arr, config="error")

    def test_correct_dtype_wrong_shape(self):
        """Test validation fails with correct dtype but wrong shape."""
        var = NumpyNDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float64))
        arr = np.zeros((3, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="Expected last.*dimension"):
            var.validate_value(arr, config="error")

    def test_both_wrong(self):
        """Test validation fails when both dtype and shape are wrong."""
        var = NumpyNDVariable(name="test", shape=(3, 4), dtype=np.dtype(np.float32))
        arr = np.zeros((3, 5), dtype=np.float64)
        # Should fail on dtype check (happens first)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(arr, config="error")


class TestNumpyNDVariableConfigEnum:
    """Test validation config behavior."""

    def test_config_enum_null(self):
        """Test that 'none' config performs only mandatory validation."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        arr = np.zeros((3, 4))
        var.validate_value(arr, config=ConfigEnum.NULL)  # Should pass

    def test_config_enum_error(self):
        """Test that 'error' config performs all validation."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        arr = np.zeros((3, 4))
        var.validate_value(arr, config=ConfigEnum.ERROR)  # Should pass

    def test_default_validation_config_used(self):
        """Test that default_validation_config is used when config is None."""
        var = NumpyNDVariable(
            name="test", shape=(3, 4), default_validation_config="error"
        )
        arr = np.zeros((3, 4))
        var.validate_value(arr)  # Should use default config


class TestNumpyNDVariableModelDump:
    """Test model serialization."""

    def test_model_dump_includes_variable_class(self):
        """Test that model_dump includes variable_class."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        dumped = var.model_dump()
        assert dumped["variable_class"] == "NumpyNDVariable"

    def test_model_dump_includes_all_fields(self):
        """Test that all fields are included in model_dump."""
        var = NumpyNDVariable(
            name="my_array", shape=(3, 4), dtype=np.dtype(np.float32), unit="m"
        )
        dumped = var.model_dump()
        assert dumped["name"] == "my_array"
        assert dumped["shape"] == (3, 4)
        assert dumped["unit"] == "m"
        # Note: dtype serialization might be handled specially


class TestNumpyNDVariableEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_shape(self):
        """Test variable with empty shape tuple (scalar-like)."""
        var = NumpyNDVariable(name="test", shape=())
        arr = np.array(5.0)
        var.validate_value(arr, config="error")

    def test_large_batch_dimensions(self):
        """Test with many batch dimensions."""
        var = NumpyNDVariable(name="test", shape=(3, 4))
        arr = np.zeros((2, 3, 5, 7, 3, 4))  # 4 batch dimensions
        var.validate_value(arr, config="error")

    def test_complex_dtype(self):
        """Test with complex number dtype."""
        var = NumpyNDVariable(name="test", shape=(3,), dtype=np.dtype(np.complex64))
        arr = np.zeros(3, dtype=np.complex64)
        var.validate_value(arr, config="error")
