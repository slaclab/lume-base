from typing import Any
from lume.model import LUMEModel
from lume.variables.particle_group import ParticleGroupVariable
from lume.variables.variable import Variable


class StagedModel(LUMEModel):
    """
    Example of staging multiple LUMEModel instances within a parent model.
    """

    def __init__(self, lume_model_instances: list[LUMEModel]):
        """Initialize the StagedModel with a list of LUMEModel instances.

        Parameters
        ----------
        lume_model_instances: list[LUMEModel]
            A list of LUMEModel instances in order to be staged within this model.
        """
        super().__init__()
        self.validate_lume_model_instances(lume_model_instances)
        self.lume_model_instances = lume_model_instances

    @classmethod
    def validate_lume_model_instances(cls, models: list[LUMEModel]):
        """
        Validate the list of LUMEModel instances to ensure they are compatible for staging.

        Parameters
        ----------
        models: list[LUMEModel]
            A list of LUMEModel instances to be validated for staging.

        """

        # check to make sure that all of the models except the last one have a beam_output variable that is a beam distribution
        for i in range(len(models) - 1):
            model = models[i]
            if "beam_output" not in model.supported_variables:
                raise ValueError(
                    f"Model {i} must have a 'beam_output' variable to stage models."
                )

            if not isinstance(
                model.supported_variables["beam_output"], ParticleGroupVariable
            ):
                raise ValueError(
                    f"The 'beam_output' variable of model {i} must be of type ParticleGroupVariable to stage models."
                )

        # check to make sure that all models after the first has a beam_input variable that is a beam distribution
        for i in range(1, len(models)):
            model = models[i]
            if "beam_input" not in model.supported_variables:
                raise ValueError(
                    f"Model {i} must have a 'beam_input' variable to stage models."
                )

            if not isinstance(
                model.supported_variables["beam_input"], ParticleGroupVariable
            ):
                raise ValueError(
                    f"The 'beam_input' variable of model {i} must be of type ParticleGroupVariable to stage models."
                )

    @property
    def supported_variables(self) -> dict[str, Variable]:
        # TODO: handle conflicts in for beam_input and beam_output variable names across models
        return {
            variable_name: variable
            for model in self.lume_model_instances
            for variable_name, variable in model.supported_variables.items()
        }

    def initialize(self):
        """
        Initialize model by doing a first pass through the models to
        get and set the beam distributions at the entrance and exit of each model in the sequence.

        """
        # run the first model, get the output beam distribution
        self.lume_model_instances[0].set()
        current_beam_distribution = self.lume_model_instances[0].get(["beam_output"])[
            "beam_output"
        ]

        for model in self.lume_model_instances[1:]:
            # set the input beam distribution for the model
            model.set({"beam_input": current_beam_distribution})
            current_beam_distribution = model.get(["beam_output"])["beam_output"]

    def _set(self, values: dict[str, Any]) -> None:
        """
        Set control parameters of the simulator by staging the input values through the sequence of LUMEModel instances.

        Parameters
        ----------
        values: dict[str, Any]
            Dictionary of variable names and their corresponding values to set in the simulator.

        Returns
        -------
        None

        Notes
        -----
        For each model in the sequence:
        - Filter the input values for the variables supported by the model.
        - If there is a new input beam distribution from the previous model, set it as the model's beam_input.
        - Set the model variables and run the model.
        - Get the outgoing beam distribution from the model and set it as new incoming beam distribution

        """

        new_input_beam_distribution = None
        for model in self.lume_model_instances:
            # get variable names supported by the model
            model_variable_names = model.supported_variables.keys()

            # filter input values for the model
            model_values = {
                k: v for k, v in values.items() if k in model_variable_names
            }

            # if there is a new input beam distribution from the previous model,
            # set it as the model's beam_input (this will also run the model
            # and update the model's beam_output)
            if new_input_beam_distribution is not None:
                model_values["beam_input"] = new_input_beam_distribution

            if model_values:
                # set the model variables and run the model
                model.set(model_values)

                # remove set variables from input values
                values = {
                    k: v for k, v in values.items() if k not in model_variable_names
                }

                # get the outgoing beam distribution from the model
                # and update the next model's incoming beam distribution
                beam_distribution = model.get(["beam_output"])["beam_output"]
                next_model_index = self.lume_model_instances.index(model) + 1
                if next_model_index < len(self.lume_model_instances):
                    new_input_beam_distribution = beam_distribution
            else:
                # if there are no updates to the current model, we can skip to the next model
                # the incoming beam distribution will remain unchanged
                new_input_beam_distribution = None
                continue
