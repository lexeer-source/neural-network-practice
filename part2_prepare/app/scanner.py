"""Recursive scanning and bounded-memory image/CSV to NPY conversion."""

import csv
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import PreparationError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".csv"}
LOGGER = logging.getLogger(__name__)


def normalize_extensions(value):
    """Parse comma-separated extensions and reject unsupported formats."""
    extensions = {"." + part.strip().lower().lstrip(".")
                  for part in value.split(",") if part.strip()}
    if not extensions or not extensions <= SUPPORTED_EXTENSIONS:
        raise PreparationError("Поддерживаются CSV, PNG, JPG, BMP и TIFF.")
    if ".csv" in extensions and extensions & IMAGE_EXTENSIONS:
        raise PreparationError("Обрабатывайте CSV отдельно от изображений.")
    return extensions


def scan_files(directory, extensions):
    """Yield matching regular files recursively, with deterministic ordering.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise PreparationError(f"Каталог не найден: {directory}")
    for current, directories, filenames in os.walk(
            directory, followlinks=False):
        directories.sort()
        for name in sorted(filenames):
            path = Path(current) / name
            if path.suffix.lower() in extensions and not path.is_symlink():
                yield path


def load_image(path, image_size):
    """Return a grayscale float32 square normalized to [0, 1]."""
    try:
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (image_size, image_size), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise PreparationError(f"Ошибка изображения {path}: {error}")


def csv_chunks(path, chunk_size):
    """Yield numeric rows in chunks; validate header, row widths and
    finiteness.
    """
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if not header or any(not name.strip() for name in header):
                raise ValueError("Пустой заголовок CSV")
            buffer = []
            row_count = 0
            for row_number, row in enumerate(reader, 2):
                if len(row) != len(header):
                    raise ValueError(f"Неверная ширина строки {row_number}")
                values = [float(value) for value in row]
                if not np.isfinite(values).all():
                    raise ValueError(f"Не конечное число: строка {row_number}")
                buffer.append(values)
                row_count += 1
                if len(buffer) == chunk_size:
                    yield header, np.asarray(buffer, dtype=np.float32)
                    buffer = []
            if buffer:
                yield header, np.asarray(buffer, dtype=np.float32)
            if row_count == 0:
                raise ValueError("CSV не содержит строк данных")
    except (OSError, ValueError, UnicodeError, StopIteration) as error:
        raise PreparationError(f"Ошибка CSV {path}: {error}")


def prepare(directory, extensions, output, image_size=64, chunk_size=256):
    """Convert files to one memory-mapped NPY and return JSON-ready
    statistics.
    """
    started = time.perf_counter()
    if image_size <= 0 or chunk_size <= 0:
        raise PreparationError("Размер изображения и блока должны быть > 0.")
    paths = list(scan_files(directory, extensions))
    if not paths:
        raise PreparationError("Подходящих файлов не найдено.")
    tabular = extensions == {".csv"}
    minimum = maximum = None
    header = None
    sample_count = 0
    if tabular:
        for path in paths:
            for names, block in csv_chunks(path, chunk_size):
                if header is None:
                    header = names
                    minimum, maximum = block.min(axis=0), block.max(axis=0)
                elif header != names:
                    raise PreparationError("Заголовки CSV различаются.")
                minimum = np.minimum(minimum, block.min(axis=0))
                maximum = np.maximum(maximum, block.max(axis=0))
                sample_count += len(block)
        sample_shape = (len(header),)
        scale = maximum - minimum
        scale[scale == 0] = 1
    else:
        sample_count, sample_shape = len(paths), (image_size, image_size)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".npy":
        raise PreparationError("Выходной файл должен иметь расширение .npy.")
    needed = sample_count * int(np.prod(sample_shape)) * 4 + 4096
    if shutil.disk_usage(output.parent).free < needed:
        raise PreparationError("Недостаточно места на диске для NPY.")
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".npy", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    mapped = None
    try:
        mapped = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32,
            shape=(sample_count, *sample_shape))
        offset = 0
        for path in paths:
            if tabular:
                for _, block in csv_chunks(path, chunk_size):
                    mapped[offset:offset + len(block)] = (
                        block - minimum) / scale
                    offset += len(block)
            else:
                mapped[offset] = load_image(path, image_size)
                offset += 1
            LOGGER.info("Prepared %s", path.name)
        if offset != sample_count:
            raise PreparationError("Входные файлы изменились при обработке.")
        mapped.flush()
        del mapped
        mapped = None
        temporary.replace(output)
    except (OSError, ValueError, PreparationError) as error:
        if mapped is not None:
            del mapped
        temporary.unlink(missing_ok=True)
        raise PreparationError(f"Подготовка прервана: {error}")
    statistics = {
        "file_count": len(paths), "samples": sample_count,
        "shape": [sample_count, *sample_shape], "dtype": "float32",
        "input_bytes": sum(path.stat().st_size for path in paths),
        "output_bytes": output.stat().st_size,
        "seconds": time.perf_counter() - started, "output": str(output),
        "normalization": ("global column min-max" if tabular
                          else "pixels / 255"),
        "chunk_size": chunk_size,
    }
    if tabular:
        statistics["columns"] = header
        statistics["minimum"] = minimum.tolist()
        statistics["scale"] = scale.tolist()
        statistics["scope"] = "feature preprocessing without train/test split"
    output.with_suffix(".manifest.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8")
    return statistics
