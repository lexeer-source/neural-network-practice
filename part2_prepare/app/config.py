"""CLI defaults read from environment variables without absolute paths."""

import os
from dataclasses import dataclass
from pathlib import Path


class PreparationError(Exception):
    """A preparation input, configuration or output is invalid."""


@dataclass
class Settings:
    """Default input path, output directory, extensions and chunk settings."""

    path: Path
    output_dir: Path
    log_level: str
    image_size: int
    chunk_size: int
    extensions: str

    @classmethod
    def from_env(cls):
        """Return settings from NN_* variables, rejecting invalid numbers.

        Args:
            No explicit arguments.

        Returns:
            Validated Settings instance built from NN_* variables.
        """
        try:
            settings = cls(
                Path(os.getenv("NN_DATA_PATH", "data/input/images")),
                Path(os.getenv("NN_OUTPUT_DIR", "data/output")),
                os.getenv("NN_LOG_LEVEL", "INFO").upper(),
                int(os.getenv("NN_IMAGE_SIZE", "64")),
                int(os.getenv("NN_CHUNK_SIZE", "256")),
                os.getenv("NN_EXT", "png,jpg,jpeg"))
            if settings.image_size <= 0 or settings.chunk_size <= 0:
                raise ValueError("Размер изображения и блока должны быть > 0")
            if settings.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
                raise ValueError("Неверный NN_LOG_LEVEL")
            return settings
        except ValueError as error:
            raise PreparationError(f"Ошибка настроек: {error}")
