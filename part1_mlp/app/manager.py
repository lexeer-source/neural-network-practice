"""CSV validation, independent train/test split and train-only
standardization.
"""

import csv
from pathlib import Path

import numpy as np

from .exceptions import DataFileError, InvalidParameterError


def read_numeric_csv(path):
    """Return (headers, numeric matrix); reject missing/nonfinite/ragged rows.
    """
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            headers = next(reader)
            if len(headers) < 1 or any(not name.strip() for name in headers):
                raise ValueError("Пустой заголовок CSV")
            rows = []
            for line_number, row in enumerate(reader, 2):
                if not row or len(row) != len(headers):
                    raise ValueError(f"Неверная строка CSV: {line_number}")
                rows.append([float(value) for value in row])
            matrix = np.asarray(rows, dtype=float)
            if not rows or not np.isfinite(matrix).all():
                raise ValueError("Пустые данные или нечисловые значения")
        return headers, matrix
    except (OSError, UnicodeError, ValueError, StopIteration) as error:
        raise DataFileError(f"Ошибка CSV: {error}")


class DatasetManager:
    """Own numeric features and binary targets read from the last CSV column.
    """

    def __init__(self):
        """Create an unloaded manager; split() fits the training scaler."""
        self.features = None
        self.targets = None
        self.center = None
        self.scale = None
        self.feature_names = []

    def load_csv(self, path):
        """Load a headered CSV with >=1 features and a final 0/1 target
        column.
        """
        names, matrix = read_numeric_csv(path)
        if matrix.shape[1] < 2 or len(matrix) < 2:
            raise DataFileError("Нужны признаки, цель и минимум две строки.")
        if not np.isin(matrix[:, -1], [0, 1]).all():
            raise DataFileError("Этот учебный пример использует цели 0 и 1.")
        self.feature_names = names[:-1]
        self.features = matrix[:, :-1]
        self.targets = matrix[:, -1:]
        self.center = self.scale = None
        return self

    def split(self, test_fraction=0.2, seed=42):
        """Return scaled train/test arrays; fit center/scale on train rows
        only.
        """
        if self.features is None:
            raise DataFileError("Данные еще не загружены.")
        if not 0 < test_fraction < 1:
            raise InvalidParameterError("Доля test должна быть между 0 и 1.")
        order = np.random.default_rng(seed).permutation(len(self.features))
        test_count = max(1, min(len(order) - 1,
                               round(len(order) * test_fraction)))
        self.test_indices = order[:test_count]
        self.train_indices = order[test_count:]
        raw_train = self.features[self.train_indices]
        self.center = raw_train.mean(axis=0)
        self.scale = raw_train.std(axis=0)
        self.scale[self.scale == 0] = 1
        return (self.transform(raw_train),
                self.transform(self.features[self.test_indices]),
                self.targets[self.train_indices],
                self.targets[self.test_indices])

    def transform(self, features):
        """Apply the fitted train scaler without modifying its parameters."""
        features = np.asarray(features, dtype=float)
        if (self.center is None or features.ndim != 2
                or features.shape[1] != len(self.center)
                or not np.isfinite(features).all()):
            raise DataFileError("Неверные признаки или scaler еще не обучен.")
        return (features - self.center) / self.scale

    def state(self):
        """Return serializable training normalization parameters and names."""
        if self.center is None:
            raise DataFileError("Сначала разделите данные на train/test.")
        return {"center": self.center.tolist(), "scale": self.scale.tolist(),
                "feature_names": self.feature_names}
