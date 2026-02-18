"""Tests for ParticleGroupVariable class."""

import pytest

from lume.variables.particle_group import ParticleGroupVariable

# Try to import ParticleGroup, mark tests as skipped if not available
try:
    from pmd_beamphysics import ParticleGroup

    PARTICLE_GROUP_AVAILABLE = True
except ImportError:
    PARTICLE_GROUP_AVAILABLE = False
    ParticleGroup = None

pytestmark = pytest.mark.skipif(
    not PARTICLE_GROUP_AVAILABLE, reason="pmd_beamphysics not available"
)


class TestParticleGroupVariableCreation:
    """Test ParticleGroupVariable instantiation."""

    def test_basic_creation(self):
        """Test creating a basic particle group variable."""
        var = ParticleGroupVariable(name="test_pg")
        assert var.name == "test_pg"
        assert var.read_only is False

    def test_creation_with_read_only(self):
        """Test creating a read-only particle group variable."""
        var = ParticleGroupVariable(name="output_pg", read_only=True)
        assert var.name == "output_pg"
        assert var.read_only is True

    def test_creation_with_validation_config(self):
        """Test creating a variable with default validation config."""
        var = ParticleGroupVariable(name="test_pg", default_validation_config="warn")
        assert var.default_validation_config == "warn"


class TestParticleGroupVariableValidation:
    """Test value validation functionality."""

    def test_validate_particle_group(self):
        """Test validation accepts ParticleGroup instances."""
        var = ParticleGroupVariable(name="test")
        # Create a minimal ParticleGroup using data dict with required fields
        pg = ParticleGroup(
            data={
                "x": [0],
                "px": [0],
                "y": [0],
                "py": [0],
                "z": [0],
                "pz": [0],
                "t": [0],
                "status": [1],
                "weight": [1],
                "species": "electron",
            }
        )
        var.validate_value(pg)  # Should not raise

    def test_reject_dict(self):
        """Test validation rejects dict values."""
        var = ParticleGroupVariable(name="test")
        with pytest.raises(TypeError, match="Value must be of type ParticleGroup"):
            var.validate_value({})

    def test_reject_none(self):
        """Test validation rejects None."""
        var = ParticleGroupVariable(name="test")
        with pytest.raises(TypeError, match="Value must be of type ParticleGroup"):
            var.validate_value(None)

    def test_reject_string(self):
        """Test validation rejects string values."""
        var = ParticleGroupVariable(name="test")
        with pytest.raises(TypeError, match="Value must be of type ParticleGroup"):
            var.validate_value("particle_group")

    def test_reject_list(self):
        """Test validation rejects list values."""
        var = ParticleGroupVariable(name="test")
        with pytest.raises(TypeError, match="Value must be of type ParticleGroup"):
            var.validate_value([])

    def test_reject_numpy_array(self):
        """Test validation rejects numpy arrays."""
        import numpy as np

        var = ParticleGroupVariable(name="test")
        with pytest.raises(TypeError, match="Value must be of type ParticleGroup"):
            var.validate_value(np.array([1, 2, 3]))


class TestParticleGroupVariableModelDump:
    """Test model serialization."""

    def test_model_dump_includes_variable_class(self):
        """Test that model_dump includes variable_class."""
        var = ParticleGroupVariable(name="test")
        dumped = var.model_dump()
        assert dumped["variable_class"] == "ParticleGroupVariable"

    def test_model_dump_includes_all_fields(self):
        """Test that all fields are included in model_dump."""
        var = ParticleGroupVariable(
            name="beam", read_only=True, default_validation_config="error"
        )
        dumped = var.model_dump()
        assert dumped["name"] == "beam"
        assert dumped["read_only"] is True
        assert dumped["default_validation_config"] == "error"
