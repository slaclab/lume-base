import numpy as np

from lume.tests.files import INPUT_YAML, LUME_CONFIG_YAML
from lume.tests.files.test_command_wrapper_subclass import MyModel


def test_input_parser(lume_object, input_file):
    lume_object.input_parser(input_file)


def test_write_input(lume_object, tmp_path):
    lume_object.write_input(f"{tmp_path}/tmp.npy")


def test_hdf5(lume_object, tmp_path):
    hdf5_filename = f"{tmp_path}/tmp.hdf5"
    lume_object.to_hdf5(hdf5_filename)

    bj2 = lume_object.from_hdf5(hdf5_filename)

    assert (bj2.input["data"] == lume_object.input["data"]).all()


class TestCommandWrapperSubclass:
    LUME_CONFIG = f"""
    input_file: {INPUT_YAML}
    use_temp_dir: false
    use_mpi: false
    timeout: 100
    """

    def test_load_from_yaml_file(self):
        model = MyModel.from_yaml(LUME_CONFIG_YAML, parse_input=True)
        assert isinstance(model._input_image, (np.ndarray,))
        assert model._variables["variable_1"]["value"] == 1
        assert model._variables["variable_2"]["value"] == 2

    def test_load_from_yaml(self):
        # already loaded file
        model = MyModel.from_yaml(self.LUME_CONFIG, parse_input=True)
        assert isinstance(model._input_image, (np.ndarray,))
        assert model._variables["variable_1"]["value"] == 1
        assert model._variables["variable_2"]["value"] == 2
