"""Create explicitly synthetic XOR samples, feature CSVs and image patterns."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def generate(destination, seed=42, images=24, size=64):
    """Write seeded independent XOR samples and generated grayscale images.

    Args:
        destination: Directory in which synthetic input files are written.
        seed: Integer seed for reproducible data or transformation choices.
        images: Positive number of synthetic image files to generate.
        size: Positive side length of the processed square images.

    Returns:
        None; seeded CSV and PNG files are written to destination.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    rows = []
    for _ in range(240):
        bits = generator.integers(0, 2, size=2)
        features = bits + generator.normal(0, 0.07, size=2)
        rows.append([*features, int(bits[0] != bits[1])])
    with (destination / "xor.csv").open(
            "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["signal_left", "signal_right", "target"])
        writer.writerows(rows)
    csv_dir = destination / "tables"
    csv_dir.mkdir(exist_ok=True)
    for part in range(2):
        with (csv_dir / f"features_{part}.csv").open(
                "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(["signal_left", "signal_right"])
            subset = rows[part * 120:(part + 1) * 120]
            writer.writerows([row[:2] for row in subset])
    row_grid, column_grid = np.mgrid[:size, :size]
    for index in range(images):
        directory = destination / "images" / f"group_{index % 2}"
        directory.mkdir(parents=True, exist_ok=True)
        pattern = ((row_grid // (3 + index % 8)
                    + column_grid // (4 + index % 5)) % 2) * 180 + 30
        pixels = np.clip(pattern + generator.normal(
            0, 6, pattern.shape), 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(directory / f"sample_{index:03d}.png")
    print(f"Synthetic XOR: 240 rows; images: {images}, size: {size}x{size}.")
    print(f"Saved to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/input")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--size", type=int, default=64)
    options = parser.parse_args()
    if options.images <= 0 or options.size <= 0:
        parser.error("images and size must be positive")
    generate(options.output, options.seed, options.images, options.size)
