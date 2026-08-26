from abc import ABC, abstractmethod
from typing import Any

from beamphysics import ParticleGroup

from lume.model import LUMEModel
from lume.variables.variable import Variable


class InitialParticlesMixIn(ABC):
    """
    Mix in to LUMEModel to indicate support for initial particles.
    """

    @property
    @abstractmethod
    def initial_particles(self) -> ParticleGroup: ...

    @initial_particles.setter
    @abstractmethod
    def initial_particles(self, val: ParticleGroup): ...


class FinalParticlesMixIn(ABC):
    """
    Mix in to LUMEModel to indicate support for final particles.
    """

    @property
    @abstractmethod
    def final_particles(self) -> ParticleGroup: ...


class StagedModel(LUMEModel):
    """
    Composes multiple LUMEModel instances in sequence, passing final particles
    from each stage as initial particles to the next.
    """

    def __init__(self, lume_model_instances: list[LUMEModel]):
        """
        Initialize the `StagedModel` with a list of LUMEModel instances.

        To ensure that the model is accurate after instantiation, this method will run a single
        start-to-end evaluation to propagate initial particles through all stages.

        Parameters
        ----------
        lume_model_instances: list[LUMEModel]
            Ordered list of LUMEModel instances to stage.
        """
        super().__init__()
        self.validate_lume_model_instances(lume_model_instances)
        self.lume_model_instances = lume_model_instances
        self.run_start_to_end_propagation()

    def run_start_to_end_propagation(self) -> None:
        """Run a single start-to-end propagation across all staged models."""

        # Collect initial values for the first writable variable
        # in each model to trigger updates across all models.
        initial_values = {}
        for model in self.lume_model_instances:
            first_writable_name = next(
                (
                    name
                    for name, variable in model.supported_variables.items()
                    if not variable.read_only
                ),
                None,
            )
            if first_writable_name is not None:
                initial_values.update(model.get([first_writable_name]))

        self.set(initial_values)

    @classmethod
    def validate_lume_model_instances(cls, models: list[LUMEModel]):
        """
        Parameters
        ----------
        models: list[LUMEModel]
            Models to validate for staging compatibility.
        """
        for i, model in enumerate(models[:-1]):
            if not isinstance(model, FinalParticlesMixIn):
                raise ValueError(
                    f"Model {i} must implement FinalParticlesMixIn to stage models."
                )

        for i, model in enumerate(models[1:], start=1):
            if not isinstance(model, InitialParticlesMixIn):
                raise ValueError(
                    f"Model {i} must implement InitialParticlesMixIn to stage models."
                )

        seen: dict[str, int] = {}
        for i, model in enumerate(models):
            for name in model.supported_variables:
                if name in seen:
                    raise ValueError(
                        f"Variable '{name}' is defined in both model {seen[name]} and model {i}."
                    )
                seen[name] = i

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {
            name: var
            for model in self.lume_model_instances
            for name, var in model.supported_variables.items()
        }

    def _get(self, names: list[str]) -> dict[str, Any]:
        values = {}
        for model in self.lume_model_instances:
            model_names = [n for n in names if n in model.supported_variables]
            if model_names:
                values.update(model.get(model_names))
        return values

    def _set(self, values: dict[str, Any]) -> None:
        """
        Set variable values across the staged models.

        Parameters
        ----------
        values: dict[str, Any]
            Variable names and values to set across the staged models.
        """
        incoming_particles = None
        for i, model in enumerate(self.lume_model_instances):
            model_values = {
                k: v for k, v in values.items() if k in model.supported_variables
            }

            if i > 0 and incoming_particles is not None:
                model.initial_particles = incoming_particles

            if model_values:
                model.set(model_values)

            if isinstance(model, FinalParticlesMixIn):
                incoming_particles = model.final_particles

    def reset(self):
        for model in self.lume_model_instances:
            model.reset()
        self.run_start_to_end_propagation()
