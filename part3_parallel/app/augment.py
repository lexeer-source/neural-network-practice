"""Deterministic image transformations suitable for spawn worker processes."""

import numpy as np


def augment_image(index, image, variants, seed):
    """Return (index, original + seeded rotations/reflections/noise variants).

    Args:
        index: Stable source image index, also used to seed its variants.
        image: Normalized two-dimensional grayscale NumPy array.
        variants: Number of augmented copies added to each original image.
        seed: Integer seed for reproducible data or transformation choices.

    Returns:
        Pair of source index and original plus augmented arrays.
    """
    result = np.empty((variants + 1, *image.shape), dtype=np.float32)
    result[0] = image
    for variant in range(variants):
        generator = np.random.default_rng(
            np.random.SeedSequence([seed, index, variant]))
        transformed = np.rot90(image, int(generator.integers(0, 4)))
        if generator.random() < 0.5:
            transformed = np.fliplr(transformed)
        noise = generator.normal(0, 0.025, image.shape).astype(np.float32)
        result[variant + 1] = np.clip(transformed + noise, 0, 1)
    return index, result
