from abc import ABC, abstractmethod


class ModuleImportError(Exception):
    def __init__(self, module_name, module_version):
        self.module_name = module_name
        module_base = module_name.split(".")[0]
        module_version = module_version
        self.message = (
            f"Unable to import module {module_name}. Object was serialized with {module_version}. "
            f"Is a compatible version of {module_base} installed?"
        )
        super().__init__(self.message)


class ClassInitError(Exception):
    def __init__(self, class_name, module_name, module_version):
        self.module_name = module_name
        self.class_name = class_name
        module_base = module_name.split(".")[0]
        self.message = (
            f"Unable to get {class_name} from {module_name}. Object was serialized with {module_version}. "
            f"Is a compatible version of {module_base} installed?"
        )
        super().__init__(self.message)


class SerializerBase(ABC):
    """Base class for serializers."""

    @abstractmethod
    def serialize(self, filename, object):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(cls, filename):
        raise NotImplementedError
