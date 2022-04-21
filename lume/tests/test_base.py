import lume
import pytest


def test_input_parser(lume_object, input_file):
    lume_object.input_parser(input_file)


def test_write_input(lume_object, tmp_path):
    lume_object.write_input(f"{tmp_path}/tmp.npy")


def test_hdf5(lume_object, tmp_path):
    hdf5_filename = f"{tmp_path}/tmp.hdf5"
    lume_object.to_hdf5(hdf5_filename)

    bj2 = lume_object.from_hdf5(hdf5_filename)

    assert (bj2.input["data"] == lume_object.input["data"]).all()
