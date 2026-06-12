import pytest

from lume.actions import Action, ReadOnlyActionMixin, WritableActionMixin
from lume.exceptions import ReadOnlyError
from lume.variables.scalar import ScalarVariable


class MockSim:
    def __init__(self):
        self.values = {}


class MockReadOnlyVar(ReadOnlyActionMixin, ScalarVariable):
    def _get(self, simulator: MockSim) -> float:
        return simulator.values.get(self.name, 0.0)


class MockWritableVar(WritableActionMixin, ScalarVariable):
    def _get(self, simulator: MockSim) -> float:
        return simulator.values.get(self.name, 0.0)

    def _set(self, simulator: MockSim, value: float) -> None:
        simulator.values[self.name] = value


class TestAction:
    def test_is_action_instance(self):
        var = MockReadOnlyVar(name="x", read_only=True)
        assert isinstance(var, Action)

    def test_is_variable_instance(self):
        var = MockReadOnlyVar(name="x", read_only=True)
        assert isinstance(var, ScalarVariable)


class TestReadOnlyActionMixin:
    def test_construction_requires_read_only(self):
        with pytest.raises(ReadOnlyError):
            MockReadOnlyVar(name="x", read_only=False)

    def test_get(self):
        sim = MockSim()
        sim.values["x"] = 3.14
        var = MockReadOnlyVar(name="x", read_only=True)
        assert var.get(sim) == 3.14

    def test_no_set_method(self):
        var = MockReadOnlyVar(name="x", read_only=True)
        assert not hasattr(var, "set")


class TestWritableActionMixin:
    def test_construction_writable(self):
        var = MockWritableVar(name="x")
        assert var.read_only is False

    def test_get(self):
        sim = MockSim()
        sim.values["y"] = 42.0
        var = MockWritableVar(name="y")
        assert var.get(sim) == 42.0

    def test_set(self):
        sim = MockSim()
        var = MockWritableVar(name="y")
        var.set(sim, 7.0)
        assert sim.values["y"] == 7.0

    def test_set_raises_when_read_only(self):
        sim = MockSim()
        var = MockWritableVar(name="y", read_only=True)
        with pytest.raises(ReadOnlyError):
            var.set(sim, 7.0)
