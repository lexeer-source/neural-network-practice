"""Interactive entry point: python -m part1_mlp.app.main."""

import os
from pathlib import Path

from .exceptions import InvalidParameterError, PracticeError
from .manager import DatasetManager, read_numeric_csv
from .models import NeuralNetwork
from .utils import save_loss_plot


def ask_number(prompt, converter):
    """Read one numeric value and convert invalid input to an application
    error.
    """
    try:
        return converter(input(prompt))
    except ValueError as error:
        raise InvalidParameterError(f"Неверное число: {error}")


class ConsoleApplication:
    """Compose the interactive network and its independent dataset manager."""

    def __init__(self):
        """Initialize an empty application without reading files or training.
        """
        self.network = None
        self.manager = DatasetManager()
        self.split_data = None

    def create_network(self):
        """Ask for layer sizes and activation, then initialize a fresh model.
        """
        try:
            sizes = [int(item) for item in
                     input("Размеры слоев (например 2 8 1): ").split()]
        except ValueError as error:
            raise InvalidParameterError(f"Неверный размер слоя: {error}")
        activation = input("Активация скрытых слоев (sigmoid/relu): ").strip()
        self.network = NeuralNetwork(sizes, activation)
        print(f"Создана сеть {sizes}, активация {activation}.")

    def load_data(self):
        """Ask for CSV and test fraction; preserve an existing model's scaler.
        """
        path = input("Путь к CSV: ").strip()
        fraction = ask_number("Доля тестовой выборки (например 0.2): ", float)
        manager = DatasetManager().load_csv(path)
        split_data = manager.split(fraction)
        self.manager, self.split_data = manager, split_data
        print(f"Данные: {len(manager.features)} строк, "
              f"{manager.features.shape[1]} признака; "
              f"train={len(split_data[0])}, test={len(split_data[1])}.")
        print("Нормализация рассчитана только на train.")

    def _require_network(self):
        if self.network is None:
            raise InvalidParameterError("Сначала создайте или загрузите сеть.")

    def train_network(self):
        """Ask for optimizer parameters; train and display held-out accuracy.
        """
        self._require_network()
        if self.split_data is None:
            raise InvalidParameterError("Сначала загрузите данные.")
        rate = ask_number("Скорость обучения: ", float)
        epochs = ask_number("Число эпох: ", int)
        batch_size = ask_number("Размер батча: ", int)
        train_features, test_features, train_targets, test_targets = (
            self.split_data)
        self.network.train(
            train_features, train_targets, rate, epochs, batch_size,
            progress=lambda epoch, loss: print(
                f"Эпоха {epoch:4d}: MSE={loss:.6f}"))
        self.network.preprocessing = self.manager.state()
        predicted = self.network.forward(test_features)
        accuracy = ((predicted >= 0.5) == test_targets).mean()
        print(f"Точность на test: {accuracy:.2%}")

    def predict(self):
        """Read raw feature vectors manually or from a headered feature CSV."""
        self._require_network()
        source = input("Ввод: 1 вручную, 2 из CSV: ").strip()
        if source == "1":
            try:
                rows = [[float(value) for value in row.split()]
                        for row in input(
                            "Признаки через пробел; строки через ';': "
                        ).split(";")]
            except ValueError as error:
                raise InvalidParameterError(f"Неверные признаки: {error}")
        elif source == "2":
            path = input("Путь к CSV без целевой колонки: ")
            _, rows = read_numeric_csv(path)
        else:
            raise InvalidParameterError("Выберите 1 или 2.")
        predicted = self.network.predict_raw(rows)
        for index, values in enumerate(predicted, 1):
            print(f"Пример {index}: выход={values.round(6).tolist()}, "
                  f"класс={(values >= 0.5).astype(int).tolist()}")

    def save_model(self):
        """Save weights, scaler and training history at a user-supplied path.
        """
        self._require_network()
        self.network.save(input("Путь к JSON-модели: ").strip())
        print("Модель сохранена.")

    def load_model(self):
        """Load and validate a user-supplied JSON model."""
        path = input("Путь к JSON-модели: ").strip()
        self.network = NeuralNetwork.load(path)
        print(f"Модель загружена: {self.network.sizes}.")

    def plot_history(self):
        """Save a loss chart without requiring a graphical desktop."""
        self._require_network()
        default = Path(os.getenv("NN_OUTPUT_DIR", "data/output")) / "loss.png"
        path = input(f"Путь к графику [{default}]: ").strip() or default
        print(f"График: {save_loss_plot(self.network.history, path)}")

    def change_activation(self):
        """Switch hidden activation and warn that retraining may be needed."""
        self._require_network()
        self.network.change_activation(input("Новая активация: ").strip())
        print("Активация изменена; для новых результатов обучите сеть заново.")

    def run(self):
        """Run all menu actions; report input/file/dimension errors
        gracefully.
        """
        actions = {
            "1": self.create_network, "2": self.load_data,
            "3": self.train_network, "4": self.predict,
            "5": self.save_model, "6": self.load_model,
            "7": self.plot_history, "8": self.change_activation,
        }
        while True:
            print("\n1 Создать сеть | 2 CSV | 3 Обучить | 4 Предсказание")
            print("5 Сохранить | 6 Загрузить | 7 График | "
                  "8 Активация | 0 Выход")
            try:
                choice = input("Выбор: ").strip()
                if choice == "0":
                    print("Работа завершена.")
                    return
                if choice not in actions:
                    raise InvalidParameterError("Неизвестный пункт меню.")
                actions[choice]()
            except PracticeError as error:
                print(f"Ошибка: {error}")
            except (EOFError, KeyboardInterrupt):
                print("\nРабота завершена пользователем.")
                return


if __name__ == "__main__":
    ConsoleApplication().run()
