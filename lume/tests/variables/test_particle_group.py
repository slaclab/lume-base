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


class TestParticleGroupVariable:
    """Test ParticleGroupVariable creation and validation."""

    def test_creation(self):
        """Test creating a particle group variable."""
        var = ParticleGroupVariable(name="test_pg")
        assert var.name == "test_pg"
        assert var.read_only is False

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
        var.validate_value(pg)

    def test_reject_invalid_types(self):
        """Test validation rejects non-ParticleGroup values."""
        var = ParticleGroupVariable(name="test")
        for invalid in [{}, None, "particle_group", []]:
            with pytest.raises(TypeError, match="Value must be of type ParticleGroup"):
                var.validate_value(invalid)
