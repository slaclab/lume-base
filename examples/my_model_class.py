from random import random
from lume.base import CommandWrapper, tools
import os
import yaml
import numpy as np

class MyModelClass(CommandWrapper):
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

            # convention for my model is that any input file paths are relative the input yaml directory
            if "input_image" in config:
                
                # Get the yaml file root
                root, _ = os.path.split(tools.full_path(path))
                input_image_path = os.path.join(root, config["input_image"])

                if os.path.exists(tools.full_path(input_image_path)):
                    with open(input_image_path, "rb") as f:
                        config["input_image"] = np.load(f)

                else:
                    print(f"unable to resolve input impage path {input_image_path}")

        else:
            print(f"unable to parse model input file at {path}")
                
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



if __name__ == "__main__":
    MyModel = MyModelClass.from_yaml("examples/lume_config.yml")

    print("Input image:")
    print(MyModel._input_image)
    print("Variables:")
    print(MyModel._variables)
