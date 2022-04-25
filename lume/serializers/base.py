from abc import ABC, abstractmethod


class SerializerBase(ABC):
    """Base class for serializers."""

    @abstractmethod
    def serialize(self, filename, object):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(cls, filename):
        raise NotImplementedError
