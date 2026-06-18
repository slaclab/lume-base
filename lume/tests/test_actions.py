import pytest

from lume.actions import ActionModel, ReadOnlyActionMixin, WritableActionMixin
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


class TestReadOnlyActionMixin:
    def test_construction_requires_read_only(self):
        with pytest.raises(ReadOnlyError):
            MockReadOnlyVar(name="x", read_only=False)

    def test_get(self):
        sim = MockSim()
        sim.values["x"] = 3.14
        var = MockReadOnlyVar(name="x", read_only=True)
        assert var._get(sim) == 3.14

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
        assert var._get(sim) == 42.0

    def test_set(self):
        sim = MockSim()
        var = MockWritableVar(name="y")
        var._set(sim, 7.0)
        assert sim.values["y"] == 7.0


class TestActionModel:
    def _make_model(self):
        sim = MockSim()
        avs = [
            MockWritableVar(name="x", default_value=1.0),
            MockReadOnlyVar(name="y", read_only=True),
        ]
        return ActionModel(sim, avs)

    def test_supported_variables(self):
        model = self._make_model()
        assert set(model.supported_variables) == {"x", "y"}

    def test_get(self):
        model = self._make_model()
        model.simulator.values["y"] = 9.9
        assert model.get("y") == 9.9

    def test_set(self):
        model = self._make_model()
        model.set({"x": 5.0})
        assert model.simulator.values["x"] == 5.0

    def test_set_read_only_raises(self):
        model = self._make_model()
        with pytest.raises(ReadOnlyError):
            model.set({"y": 1.0})

    def test_reset(self):
        model = self._make_model()
        model.simulator.values["x"] = 99.0
        model.reset()
        assert model.simulator.values["x"] == 1.0

    def test_register_replaces_existing(self):
        model = self._make_model()
        new_x = MockWritableVar(name="x", default_value=42.0)
        model.register_action_variable(new_x)
        assert model._action_variable_by_name["x"] is new_x
        assert len(model.supported_variables) == 2

    def test_register_rejects_non_action(self):
        model = self._make_model()
        with pytest.raises(ValueError):
            model.register_action_variable(ScalarVariable(name="plain"))

    def test_unregister(self):
        model = self._make_model()
        model.unregister_action_variable("x")
        assert "x" not in model.supported_variables

    def test_unregister_missing_raises(self):
        model = self._make_model()
        with pytest.raises(KeyError):
            model.unregister_action_variable("nonexistent")
