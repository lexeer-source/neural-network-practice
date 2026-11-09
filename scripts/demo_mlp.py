"""Run a reproducible MLP experiment without replacing the interactive menu."""

import argparse
import json
from pathlib import Path

import numpy as np

from part1_mlp.app.manager import DatasetManager
from part1_mlp.app.models import NeuralNetwork
from part1_mlp.app.utils import save_loss_plot


def run(csv_path, output, activation, epochs):
    """Train once with fixed parameters; save metrics, model and loss chart."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manager = DatasetManager().load_csv(csv_path)
    train_features, test_features, train_targets, test_targets = (
        manager.split())
    model = NeuralNetwork([2, 8, 1], activation=activation, seed=42)
    print(f"Network: {model.sizes}, hidden={activation}, output=sigmoid")
    print(f"CSV: {len(manager.features)} rows; train={len(train_features)}, "
          f"test={len(test_features)}; scaler fitted on train only.")
    initial, _ = model.loss_and_gradients(train_features, train_targets)
    print(f"Initial training MSE: {initial:.6f}")
    rate = 1.0 if activation == "sigmoid" else 0.1
    model.train(train_features, train_targets, rate, epochs, 16,
                progress=lambda epoch, loss: print(
                    f"epoch={epoch:4d} training_MSE={loss:.6f}"))
    model.preprocessing = manager.state()
    test_prediction = model.forward(test_features)
    accuracy = float(((test_prediction >= 0.5) == test_targets).mean())
    model_path = output / f"mlp_{activation}.json"
    model.save(model_path)
    restored = NeuralNetwork.load(model_path)
    raw_test = manager.features[manager.test_indices]
    restored_prediction = restored.predict_raw(raw_test)
    equivalent = bool(np.array_equal(test_prediction, restored_prediction))
    print(f"Held-out test accuracy: {accuracy:.2%}")
    print(f"Saved/reloaded predictions identical: {equivalent}")
    for values in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        score = float(restored.predict_raw([values])[0, 0])
        print(f"Input {values}: output={score:.6f}, class={int(score >= .5)}")
    metrics = {
        "dataset": "seeded synthetic noisy XOR", "rows": len(manager.features),
        "train_rows": len(train_features), "test_rows": len(test_features),
        "architecture": model.sizes, "hidden_activation": activation,
        "epochs": epochs, "learning_rate": rate, "batch_size": 16,
        "initial_training_mse": initial,
        "final_training_mse": model.history[-1],
        "test_mse": float(np.mean((test_prediction - test_targets) ** 2)),
        "test_accuracy": accuracy, "saved_predictions_identical": equivalent,
        "train_test_indices_disjoint": not bool(set(manager.train_indices)
                                                & set(manager.test_indices)),
    }
    (output / f"metrics_{activation}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    save_loss_plot(model.history, output / f"loss_{activation}.png")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/input/xor.csv")
    parser.add_argument("--output", default="data/output")
    parser.add_argument("--activation", choices=["sigmoid", "relu"],
                        default="sigmoid")
    parser.add_argument("--epochs", type=int, default=1200)
    args = parser.parse_args()
    run(args.csv, args.output, args.activation, args.epochs)
