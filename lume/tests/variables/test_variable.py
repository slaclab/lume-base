"""Tests for Variable base class."""

import pytest

from lume.variables.variable import ConfigEnum, Variable


class ConcreteVariable(Variable):
    """Concrete implementation of Variable for testing purposes."""

    def validate_value(self, value, config=None):
        """Simple validation that accepts only integers."""
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value)}")


class TestVariableCreation:
    """Test Variable instantiation."""

    def test_basic_creation(self):
        """Test creating a basic variable."""
        var = ConcreteVariable(name="test_var")
        assert var.name == "test_var"
        assert var.read_only is False
        assert var.default_validation_config == "none"

    def test_creation_with_read_only(self):
        """Test creating a read-only variable."""
        var = ConcreteVariable(name="const_var", read_only=True)
        assert var.name == "const_var"
        assert var.read_only is True

    def test_creation_with_validation_config(self):
        """Test creating a variable with validation config."""
        var = ConcreteVariable(name="test_var", default_validation_config="warn")
        assert var.default_validation_config == "warn"

        var = ConcreteVariable(name="test_var", default_validation_config="error")
        assert var.default_validation_config == "error"


class TestConfigEnum:
    """Test ConfigEnum functionality."""

    def test_config_enum_values(self):
        """Test ConfigEnum has expected values."""
        assert ConfigEnum.NULL == "none"
        assert ConfigEnum.WARN == "warn"
        assert ConfigEnum.ERROR == "error"

    def test_config_enum_from_string(self):
        """Test creating ConfigEnum from string."""
        assert ConfigEnum("none") == ConfigEnum.NULL
        assert ConfigEnum("warn") == ConfigEnum.WARN
        assert ConfigEnum("error") == ConfigEnum.ERROR

    def test_invalid_config_enum(self):
        """Test that invalid config values raise error."""
        with pytest.raises(ValueError):
            ConfigEnum("invalid")


class TestValidationConfigAsEnum:
    """Test _validation_config_as_enum method."""

    def test_none_uses_default(self):
        """Test that None config uses default_validation_config."""
        var = ConcreteVariable(name="test", default_validation_config="warn")
        config = var._validation_config_as_enum(None)
        assert config == ConfigEnum.WARN

    def test_string_config_converted(self):
        """Test that string configs are converted to enum."""
        var = ConcreteVariable(name="test")

        config = var._validation_config_as_enum("none")
        assert config == ConfigEnum.NULL

        config = var._validation_config_as_enum("warn")
        assert config == ConfigEnum.WARN

        config = var._validation_config_as_enum("error")
        assert config == ConfigEnum.ERROR

    def test_enum_config_preserved(self):
        """Test that ConfigEnum inputs are preserved."""
        var = ConcreteVariable(name="test")

        config = var._validation_config_as_enum(ConfigEnum.NULL)
        assert config == ConfigEnum.NULL

        config = var._validation_config_as_enum(ConfigEnum.WARN)
        assert config == ConfigEnum.WARN

        config = var._validation_config_as_enum(ConfigEnum.ERROR)
        assert config == ConfigEnum.ERROR


class TestModelDump:
    """Test basic model_dump method.

    Note: Comprehensive serialization tests (JSON/YAML) are in test_model.py
    to avoid duplication.
    """

    def test_model_dump_includes_variable_class(self):
        """Test that model_dump includes variable_class."""
        var = ConcreteVariable(name="test")
        dumped = var.model_dump()
        assert "variable_class" in dumped
        assert dumped["variable_class"] == "ConcreteVariable"

    def test_model_dump_includes_all_fields(self):
        """Test that all fields are included in model_dump."""
        var = ConcreteVariable(
            name="test_var", read_only=True, default_validation_config="error"
        )
        dumped = var.model_dump()
        assert dumped["name"] == "test_var"
        assert dumped["read_only"] is True
        assert dumped["default_validation_config"] == "error"
        assert dumped["variable_class"] == "ConcreteVariable"


class TestValidateValueAbstract:
    """Test that validate_value must be implemented in subclasses."""

    def test_concrete_implementation_works(self):
        """Test that concrete implementation works."""
        var = ConcreteVariable(name="test")
        var.validate_value(5)  # Should not raise

        with pytest.raises(TypeError):
            var.validate_value("not an int")

    def test_cannot_instantiate_abstract_variable(self):
        """Test that Variable cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Variable(name="test")


class TestVariableConfigEnumStorage:
    """Test that ConfigEnum is stored as string value."""

    def test_config_stored_as_string(self):
        """Test that config enum is stored as string value."""
        var = ConcreteVariable(name="test", default_validation_config=ConfigEnum.WARN)
        # Should be stored as string "warn"
        assert var.default_validation_config == "warn"
        assert isinstance(var.default_validation_config, str)

    def test_config_accepts_string_directly(self):
        """Test that config accepts string values directly."""
        var = ConcreteVariable(name="test", default_validation_config="error")
        assert var.default_validation_config == "error"
        assert isinstance(var.default_validation_config, str)
