"""Explicit errors presented to the console user."""


class PracticeError(Exception):
    """Base class for expected application failures."""


class InvalidLayerSizeError(PracticeError):
    """A layer size is invalid or the network has no hidden layer."""


class MismatchedDataError(PracticeError):
    """Feature, target or parameter dimensions are incompatible."""


class DataFileError(PracticeError):
    """A dataset cannot be read or contains invalid values."""


class ModelFileError(PracticeError):
    """A model file cannot be saved or restored safely."""


class InvalidParameterError(PracticeError):
    """A console argument or training parameter is invalid."""
