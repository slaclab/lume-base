import os

import numpy as np
import pytest

from lume.base import CommandWrapper
from lume.serializers.hdf5 import HDF5Serializer


@pytest.fixture(scope="session", autouse=True)
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


class LUMEObject(CommandWrapper):
    def __init__(self):
        self.output = {}
        self.input = {}

    # implement abstract configure method
    def configure(self):
        pass

    # implement abstract run method
    def run(self):
        # just assign data
        self.output["data"] = self.input["data"]

    # implement abstract archive method
    def archive(self, h5):
        input_group = h5.create_group("input")
        input_group.create_dataset("data", data=self.input["data"])

        output_group = h5.create_group("output")
        output_group.create_dataset("data", data=self.output["data"])

    # implement abstract plot method
    def plot(self, y=[], return_figure=False) -> None:
        pass

    # implement abstract write_input method
    def write_input(self, input_filename) -> None:
        np.save(input_filename, self.input["data"])

    # implement abstract input parser method
    def input_parser(self, path=""):
        return {"data": np.load(path)}

    # implement abstract load output method
    def load_output(self):
        pass

    # implement abstract load_archive method
    def load_archive(self, h5, configure=True):
        self.output["data"] = h5["output"]["data"][()]
        self.input["data"] = h5["input"]["data"][()]

    def to_hdf5(self, filename):
        serializer = HDF5Serializer()
        serializer.serialize(filename, self)

    @classmethod
    def from_hdf5(cls, filename):
        serializer = HDF5Serializer()
        return serializer.deserialize(filename)


@pytest.fixture(scope="session", autouse=True)
def input_file(rootdir):
    return f"{rootdir}/files/test_array.npy"


@pytest.fixture(scope="session", autouse=True)
def lume_object(input_file):
    lume_object = LUMEObject()
    lume_object.load_input(input_file)
    lume_object.run()
    return lume_object
