import numpy as np

from lume.serializers.hdf5 import HDF5Serializer


class SerializableObject:
    def __init__(self):
        self.data = {}

    # implement abstract input parser method
    def load_input(self, path=""):
        self.data = np.load(path)

    def archive(self, h5):
        g = h5.create_group("data")
        g = g.create_dataset("dataset", data=self.data)

    def load_archive(self, h5):
        self.data = h5["data"]["dataset"][()]


def test_serialize_and_deserialize(tmp_path, input_file):
    serializer = HDF5Serializer()

    filename = f"{tmp_path}/tmp_ser_obj.hdf5"
    object = SerializableObject()
    object.load_input(input_file)

    serializer.serialize(filename, object)

    assert len(list(tmp_path.iterdir())) > 0

    obj2 = serializer.deserialize(filename)

    assert (obj2.data == object.data).all()
