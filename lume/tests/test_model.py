import pytest
from typing import Any
from lume.model import LUMEModel
from lume.variables import Variable, ScalarVariable


# Mock Variable subclass for testing
class MockVariable(Variable):
    """Mock Variable implementation for testing purposes."""

    @property
    def default_validation_config(self):
        return "warn"

    def validate_value(self, value: Any, config=None):
        """Simple validation - just check if value is not None"""
        if value is None:
            raise ValueError("Value cannot be None")


# Concrete implementation for testing
class MockLUMEModel(LUMEModel):
    """Mock implementation of LUMEModel for testing abstract methods."""

    def __init__(self):
        self._variables = {
            "input_var": ScalarVariable(
                name="input_var",
                default_value=1.0,
                value_range=(0.0, 10.0),
                read_only=False,
            ),
            "output_var": ScalarVariable(
                name="output_var",
                default_value=2.0,
                value_range=(0.0, 20.0),
                read_only=True,
            ),
            "control_var": MockVariable(name="control_var", read_only=False),
        }
        self._state = {"input_var": 1.0, "output_var": 2.0, "control_var": "initial"}
        self._initial_state = self._state.copy()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        # Simulate setting values and running simulation
        for name, value in values.items():
            self._state[name] = value
        # Simulate computation that updates output_var based on input_var
        if "input_var" in values:
            self._state["output_var"] = values["input_var"] * 2.0

    def reset(self) -> None:
        self._state = self._initial_state.copy()


class TestLUMEModel:
    """Test suite for LUMEModel abstract base class."""

    @pytest.fixture
    def model(self):
        """Fixture providing a MockLUMEModel instance."""
        return MockLUMEModel()

    def test_supported_variables_property(self, model):
        """Test that supported_variables property returns correct variables."""
        variables = model.supported_variables
        assert isinstance(variables, dict)
        assert "input_var" in variables
        assert "output_var" in variables
        assert "control_var" in variables
        assert isinstance(variables["input_var"], ScalarVariable)
        assert isinstance(variables["output_var"], ScalarVariable)
        assert isinstance(variables["control_var"], MockVariable)

    def test_get_valid_variables(self, model):
        """Test getting valid variable values."""
        result = model.get(["input_var", "output_var"])
        assert isinstance(result, dict)
        assert result["input_var"] == 1.0
        assert result["output_var"] == 2.0

    def test_get_single_variable(self, model):
        """Test getting a single variable."""
        result = model.get(["input_var"])
        assert result == {"input_var": 1.0}

    def test_get_invalid_variable(self, model):
        """Test getting an unsupported variable raises ValueError."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.get(["invalid_var"])

    def test_get_mixed_valid_invalid_variables(self, model):
        """Test getting mix of valid and invalid variables."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.get(["input_var", "invalid_var"])

    def test_get_empty_list(self, model):
        """Test getting empty list of variables."""
        result = model.get([])
        assert result == {}

    def test_set_valid_variables(self, model):
        """Test setting valid variable values."""
        model.set({"input_var": 5.0, "control_var": "updated"})

        # Check that values were set
        result = model.get(["input_var", "control_var"])
        assert result["input_var"] == 5.0
        assert result["control_var"] == "updated"

        # Check that simulation ran (output_var should be updated)
        output_result = model.get(["output_var"])
        assert output_result["output_var"] == 10.0  # input_var * 2

    def test_set_single_variable(self, model):
        """Test setting a single variable."""
        model.set({"input_var": 3.5})
        result = model.get(["input_var"])
        assert result["input_var"] == 3.5

    def test_set_invalid_variable(self, model):
        """Test setting an unsupported variable raises ValueError."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.set({"invalid_var": 123})

    def test_set_read_only_variable(self, model):
        """Test setting a read-only variable raises ValueError."""
        with pytest.raises(ValueError, match="Variable 'output_var' is read-only"):
            model.set({"output_var": 100.0})

    def test_set_variable_validation_error(self, model):
        """Test setting variable with invalid value raises validation error."""
        # This should trigger validation in ScalarVariable
        with pytest.raises(TypeError, match="Expected value to be of type"):
            model.set({"input_var": "not_a_float"})

    def test_set_variable_out_of_range(self, model):
        """Test setting variable with out-of-range value triggers validation."""
        # This should trigger range validation warning for ScalarVariable
        # The test captures stdout to verify warning is printed

        # Set value outside range - should print warning but not raise error
        with pytest.warns(UserWarning, match="Value .* is out of valid range"):
            model.set({"input_var": 15.0})  # Range is (0.0, 10.0)

    def test_set_variable_none_validation(self, model):
        """Test setting variable to None with custom validation."""
        with pytest.raises(ValueError, match="Value cannot be None"):
            model.set({"control_var": None})

    def test_set_mixed_valid_invalid_variables(self, model):
        """Test setting mix of valid and invalid variables."""
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported"):
            model.set({"input_var": 2.0, "invalid_var": 123})

    def test_set_empty_dict(self, model):
        """Test setting empty dict of variables."""
        initial_state = model.get(["input_var", "output_var", "control_var"])
        model.set({})
        final_state = model.get(["input_var", "output_var", "control_var"])
        assert initial_state == final_state

    def test_workflow_set_get_reset(self, model):
        """Test complete workflow of set, get, and reset operations."""
        # Initial state
        initial = model.get(["input_var", "output_var", "control_var"])
        assert initial == {
            "input_var": 1.0,
            "output_var": 2.0,
            "control_var": "initial",
        }

        # Set new values
        model.set({"input_var": 4.0, "control_var": "workflow_test"})

        # Get updated state
        updated = model.get(["input_var", "output_var", "control_var"])
        assert updated == {
            "input_var": 4.0,
            "output_var": 8.0,
            "control_var": "workflow_test",
        }

        # Reset
        model.reset()

        # Verify reset
        reset_state = model.get(["input_var", "output_var", "control_var"])
        assert reset_state == initial

    def test_variable_validation_called_during_set(self, model):
        """Test that variable validation is properly called during set operations."""
        # Test with ScalarVariable validation
        with pytest.raises(TypeError):
            model.set({"input_var": [1, 2, 3]})  # Invalid type

        # Test with custom MockVariable validation
        with pytest.raises(ValueError, match="Value cannot be None"):
            model.set({"control_var": None})

    def test_supported_variables_with_non_variable_value(self):
        """Test behavior when supported_variables contains non-Variable values."""

        class InvalidVariableModel(LUMEModel):
            """Model with non-Variable in supported_variables for testing."""

            @property
            def supported_variables(self):
                return {
                    "valid_var": ScalarVariable(
                        name="valid_var", default_value=1.0, read_only=False
                    ),
                    "invalid_var": "not_a_variable_object",  # This is not a Variable!
                }

            def _get(self, names):
                return {"valid_var": 1.0, "invalid_var": "some_value"}

            def _set(self, values):
                pass

            def reset(self):
                pass

        model = InvalidVariableModel()

        # Test that the invalid entry exists in supported_variables
        variables = model.supported_variables
        assert "invalid_var" in variables
        assert not isinstance(variables["invalid_var"], Variable)

        # Test that set() fails when trying to validate the non-Variable
        with pytest.raises(ValueError, match="is not a valid Variable instance"):
            # This should fail because strings don't have read_only or validate_value attributes
            model.set({"invalid_var": "some_value"})
