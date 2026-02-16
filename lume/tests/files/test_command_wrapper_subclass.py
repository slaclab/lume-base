import os

import numpy as np
import yaml

from lume.base import Base, tools


class MyModel(Base):
    def __init__(self, *args, variables=None, input_image=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_image = input_image
        self._variables = variables

    # implementation of abstract method
    @staticmethod
    def input_parser(path):
        config = {}

        if os.path.exists(tools.full_path(path)):
            yaml_file = tools.full_path(path)
            config = yaml.safe_load(open(yaml_file))

            if "input_image" in config:
                # check if input image full path provided
                if os.path.exists(tools.full_path(config["input_image"])):
                    input_image_path = tools.full_path(config["input_image"])

                # if not a full path, compose path relative to the yaml file directory
                else:
                    root, _ = os.path.split(tools.full_path(path))
                    input_image_path = os.path.join(root, config["input_image"])

                    if not os.path.exists(tools.full_path(input_image_path)):
                        raise Exception(
                            "Unable to resolve input impage path %s", input_image_path
                        )

                config["input_image"] = np.load(input_image_path)

        else:
            raise Exception("Unable to parse model input file path %s", path)

        return config

    def archive(self): ...

    def configure(self): ...

    def load_archive(self): ...

    def load_output(self): ...

    def plot(self): ...

    def run(self): ...

    def write_input(self): ...
