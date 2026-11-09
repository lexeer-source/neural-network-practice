"""Optional loss plotting, kept separate from numerical network code."""

from pathlib import Path

from .exceptions import InvalidParameterError, ModelFileError


def save_loss_plot(history, path):
    """Save epoch/MSE chart to path; return the actual image path."""
    if not history:
        raise InvalidParameterError("История обучения пока пуста.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(range(1, len(history) + 1), history, color="#0072B2")
        axis.set(xlabel="Epoch", ylabel="Training MSE",
                 title="MLP training loss")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path
    except (OSError, ImportError, ValueError) as error:
        raise ModelFileError(f"Не удалось сохранить график: {error}")
