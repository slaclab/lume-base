from random import random
from lume.base import CommandWrapper, tools
import os
import yaml
import numpy as np

class MyModel(CommandWrapper):
    def __init__(self, *args, variables=None, input_image=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_image = input_image
        self._variables = variables


    #implementation of abstract method
    @staticmethod
    def input_parser(path):
        config = {}

        if os.path.exists(tools.full_path(path)):
            yaml_file = tools.full_path(path)
            config = yaml.safe_load(open(yaml_file))


            # convention for this is that any input file paths are relative the input 
            # yaml directory
            if "input_image" in config:
                
                # Get the yaml file root
                root, _ = os.path.split(tools.full_path(path))
                input_image_path = os.path.join(root, config["input_image"])

                if os.path.exists(tools.full_path(input_image_path)):
                    config["input_image"] = np.load(input_image_path)

                else:
                    raise Exception("Unable to resolve input impage path %s", input_image_path)

        else:
            raise Exception("Unable to parse model input file path %s", path)
                
        return config

    def archive(self):
        ...

    def configure(self):
        ...

    def load_archive(self):
        ...

    def load_output(self):
        ...

    def plot(self):
        ...

    def run(self):
        ...

    def write_input(self):
        ...
