"""Dense layers, mini-batch backpropagation and JSON model persistence."""

import json
from pathlib import Path

import numpy as np

from .activations import Sigmoid, make_activation
from .exceptions import (
    InvalidLayerSizeError, InvalidParameterError, MismatchedDataError,
    ModelFileError,
)


class DenseLayer:
    """A fully connected layer owned by a NeuralNetwork instance."""

    def __init__(self, inputs, outputs, activation, generator, he=False):
        """Initialize sizes, Activation object, RNG and random parameters.

        Args:
            inputs: Positive number of input features.
            outputs: Positive number of output units.
            activation: Activation object for a layer, or its name for a
            network.
            generator: NumPy random generator used to initialize weights.
            he: Use He initialization when True; otherwise use 1 / inputs.

        Returns:
            None; initializes the new instance.
        """
        scale = np.sqrt((2 if he else 1) / inputs)
        self.weights = generator.normal(0, scale, (inputs, outputs))
        self.biases = np.zeros((1, outputs))
        self.activation = activation


class NeuralNetwork:
    """An MLP composed of dense layers; sigmoid outputs support binary labels.
    """

    def __init__(self, sizes, activation="sigmoid", seed=42):
        """Build at least input, hidden and output layers from positive sizes.

        Args:
            sizes: Positive layer sizes, including input, hidden and output.
            activation: Activation object for a layer, or its name for a
            network.
            seed: Integer seed for reproducible data or transformation choices.

        Returns:
            None; initializes the new instance.
        """
        if (len(sizes) < 3 or any(
                not isinstance(size, int) or isinstance(size, bool)
                or size <= 0 for size in sizes)):
            raise InvalidLayerSizeError(
                "Нужны минимум три положительных размера слоев.")
        self.sizes = list(sizes)
        self.activation_name = activation
        make_activation(activation)
        self.seed = seed
        self.generator = np.random.default_rng(seed)
        self.layers = []
        for index, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
            last = index == len(sizes) - 2
            function = Sigmoid() if last else make_activation(activation)
            self.layers.append(DenseLayer(
                inputs, outputs, function, self.generator,
                he=activation == "relu" and not last))
        self.history = []
        self.preprocessing = None

    def change_activation(self, name):
        """Replace hidden activations; keep sigmoid output and current
        weights.

        Args:
            name: Activation name: sigmoid or relu.

        Returns:
            None; the hidden layer activations are updated.
        """
        make_activation(name)
        self.activation_name = name
        for layer in self.layers[:-1]:
            layer.activation = make_activation(name)

    def _features(self, features):
        try:
            features = np.asarray(features, dtype=float)
        except (ValueError, TypeError, OverflowError) as error:
            raise MismatchedDataError(f"Неверные строки признаков: {error}")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if (features.ndim != 2 or len(features) == 0
                or features.shape[1] != self.sizes[0]
                or not np.isfinite(features).all()):
            raise MismatchedDataError("Неверное число или значения признаков.")
        return features

    def _targets(self, targets, count):
        try:
            targets = np.asarray(targets, dtype=float)
        except (ValueError, TypeError, OverflowError) as error:
            raise MismatchedDataError(f"Нечисловые целевые значения: {error}")
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if (targets.shape != (count, self.sizes[-1])
                or not np.isfinite(targets).all()
                or ((targets < 0) | (targets > 1)).any()):
            raise MismatchedDataError(
                "Цели должны иметь совместимую форму и значения от 0 до 1.")
        return targets

    def forward(self, features, return_cache=False):
        """Return predictions; optionally include activations for backprop.

        Args:
            features: Numeric feature matrix; columns must match the model.
            return_cache: Include intermediate layer values when True.

        Returns:
            Activated array; NeuralNetwork optionally also returns cache.
        """
        activated = [self._features(features)]
        preactivated = []
        for layer in self.layers:
            values = activated[-1] @ layer.weights + layer.biases
            preactivated.append(values)
            activated.append(layer.activation.forward(values))
        if return_cache:
            return activated[-1], (activated, preactivated)
        return activated[-1]

    def loss_and_gradients(self, features, targets):
        """Return mean squared error and analytic (weight, bias) gradients.

        Args:
            features: Numeric feature matrix; columns must match the model.
            targets: Target matrix matching the batch and output dimensions.

        Returns:
            Pair of scalar MSE and per-layer weight/bias gradients.
        """
        predicted, (activated, preactivated) = self.forward(features, True)
        expected = self._targets(targets, len(predicted))
        difference = predicted - expected
        loss = float(np.mean(difference ** 2))
        derivative = 2 * difference / difference.size
        gradients = [None] * len(self.layers)
        for index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[index]
            delta = derivative * layer.activation.derivative(
                preactivated[index])
            gradients[index] = (activated[index].T @ delta,
                                delta.sum(axis=0, keepdims=True))
            derivative = delta @ layer.weights.T
        return loss, gradients

    def train(self, features, targets, learning_rate=1.0, epochs=1500,
              batch_size=16, progress=None):
        """Train on shuffled mini-batches and append full-training MSE per
        epoch.

        Args:
            features: Numeric feature matrix; columns must match the model.
            targets: Target matrix matching the batch and output dimensions.
            learning_rate: Positive gradient descent step size.
            epochs: Positive number of complete training passes.
            batch_size: Positive maximum number of rows per mini-batch.
            progress: Optional callback receiving the epoch and training MSE.

        Returns:
            Complete list of training MSE values recorded on this model.
        """
        features = self._features(features)
        targets = self._targets(targets, len(features))
        if (not np.isfinite(learning_rate) or learning_rate <= 0
                or not isinstance(epochs, int) or epochs <= 0
                or not isinstance(batch_size, int) or batch_size <= 0):
            raise InvalidParameterError("Параметры обучения должны быть > 0.")
        for epoch in range(1, epochs + 1):
            order = self.generator.permutation(len(features))
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                _, gradients = self.loss_and_gradients(
                    features[batch], targets[batch])
                for layer, (weight_grad, bias_grad) in zip(
                        self.layers, gradients):
                    layer.weights -= learning_rate * weight_grad
                    layer.biases -= learning_rate * bias_grad
            loss = float(np.mean((self.forward(features) - targets) ** 2))
            if not np.isfinite(loss):
                raise InvalidParameterError("Ошибка обучения: уменьшите lr.")
            self.history.append(loss)
            if progress and (epoch == 1 or epoch % 100 == 0
                             or epoch == epochs):
                progress(epoch, loss)
        return self.history

    def predict_raw(self, features):
        """Normalize raw inputs using stored training parameters, then
        predict.

        Args:
           features: Numeric feature matrix; columns must match the model.

        Returns:
            Network outputs after applying the stored training scaler.
        """
        features = self._features(features)
        if self.preprocessing is None:
            raise InvalidParameterError("Сначала загрузите обучающие данные.")
        center = np.asarray(self.preprocessing["center"], dtype=float)
        scale = np.asarray(self.preprocessing["scale"], dtype=float)
        return self.forward((features - center) / scale)

    def save(self, path):
        """Save architecture, weights, scaler and history in a JSON file.

        Args:
            path: Input or output filesystem path used by this operation.

        Returns:
            None; the JSON model is written to the supplied path.
        """
        payload = {
            "format_version": 1, "sizes": self.sizes,
            "activation": self.activation_name, "seed": self.seed,
            "history": self.history, "preprocessing": self.preprocessing,
            "layers": [{"weights": layer.weights.tolist(),
                        "biases": layer.biases.tolist()}
                       for layer in self.layers],
        }
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False,
                                       indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as error:
            raise ModelFileError(f"Не удалось сохранить модель: {error}")

    @classmethod
    def load(cls, path):
        """Validate and restore a JSON model without executing serialized
        code.

        Args:
            path: Input or output filesystem path used by this operation.

        Returns:
            Validated NeuralNetwork restored from the JSON file.
        """
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload["format_version"] != 1:
                raise ValueError("Неизвестный формат модели")
            model = cls(payload["sizes"], payload["activation"],
                        payload["seed"])
            if len(payload["layers"]) != len(model.layers):
                raise ValueError("Неверное число слоев")
            for layer, saved in zip(model.layers, payload["layers"]):
                for key in ("weights", "biases"):
                    value = np.asarray(saved[key], dtype=float)
                    if (value.shape != getattr(layer, key).shape
                            or not np.isfinite(value).all()):
                        raise ValueError("Некорректные параметры слоя")
                    setattr(layer, key, value)
            model.history = [float(loss) for loss in payload["history"]]
            if not np.isfinite(model.history).all():
                raise ValueError("Некорректная история ошибок")
            state = payload["preprocessing"]
            if state is not None:
                center = np.asarray(state["center"], dtype=float)
                scale = np.asarray(state["scale"], dtype=float)
                if (center.shape != (model.sizes[0],)
                        or scale.shape != center.shape
                        or not np.isfinite(center).all()
                        or not np.isfinite(scale).all() or (scale <= 0).any()):
                    raise ValueError("Некорректные параметры нормализации")
            model.preprocessing = state
            return model
        except (OSError, ValueError, TypeError, KeyError,
                InvalidLayerSizeError, InvalidParameterError) as error:
            raise ModelFileError(f"Не удалось загрузить модель: {error}")
