from importlib import resources

LUME_CONFIG_YAML = str(resources.files("lume.tests.files") / "test_lume_config.yml")
INPUT_YAML = str(resources.files("lume.tests.files") / "test_input_file.yml")
