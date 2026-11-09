"""Activation interfaces demonstrate inheritance independently of layers."""

from abc import ABC, abstractmethod

import numpy as np

from .exceptions import InvalidParameterError


class Activation(ABC):
    """Interface mapping preactivation arrays to outputs and derivatives."""

    @abstractmethod
    def forward(self, values):
        """Return activated values with the same shape as input values.

        Args:
            values: Array of preactivation values; the input shape is
            preserved.

        Returns:
            Activated array with the same shape as values.
        """

    @abstractmethod
    def derivative(self, values):
        """Return the elementwise derivative at preactivation values.

        Args:
            values: Array of preactivation values; the input shape is
            preserved.

        Returns:
            Elementwise activation derivatives with the input shape.
        """


class Sigmoid(Activation):
    """Numerically stable logistic activation."""

    def forward(self, values):
        """Map a finite array to values between zero and one.

        Args:
            values: Array of preactivation values; the input shape is
            preserved.

        Returns:
            Activated array with the same shape as values.
        """
        result = np.empty_like(values, dtype=float)
        positive = values >= 0
        result[positive] = 1 / (1 + np.exp(-values[positive]))
        exponent = np.exp(values[~positive])
        result[~positive] = exponent / (1 + exponent)
        return result

    def derivative(self, values):
        """Return sigmoid(values) * (1 - sigmoid(values)).

        Args:
            values: Array of preactivation values; the input shape is
            preserved.

        Returns:
            Elementwise activation derivatives with the input shape.
        """
        activated = self.forward(values)
        return activated * (1 - activated)


class ReLU(Activation):
    """Rectified linear activation, with derivative zero at the origin."""

    def forward(self, values):
        """Return max(values, 0) elementwise.

        Args:
            values: Array of preactivation values; the input shape is
            preserved.

        Returns:
            Activated array with the same shape as values.
        """
        return np.maximum(values, 0)

    def derivative(self, values):
        """Return one for positive inputs and zero otherwise.

        Args:
            values: Array of preactivation values; the input shape is
            preserved.

        Returns:
            Elementwise activation derivatives with the input shape.
        """
        return (values > 0).astype(float)


def make_activation(name):
    """Return the requested activation or raise InvalidParameterError.

    Args:
        name: Activation name: sigmoid or relu.

    Returns:
        A new Sigmoid or ReLU instance.
    """
    choices = {"sigmoid": Sigmoid, "relu": ReLU}
    if name not in choices:
        raise InvalidParameterError("Активация: sigmoid или relu.")
    return choices[name]()
