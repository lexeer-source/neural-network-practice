"""Measure the same seeded workload in sequential and parallel execution."""

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import statistics
import sys

from part2_prepare.app.config import PreparationError, Settings
from part2_prepare.app.logging_config import configure_logging
from part2_prepare.app.scanner import normalize_extensions
from .producer_consumer import run_parallel, run_sequential


def file_digest(path):
    """Return SHA256 by reading small chunks rather than the entire file.

    Args:
        path: Input or output filesystem path used by this operation.

    Returns:
        Hexadecimal SHA256 digest of the file bytes.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark(path, output, size=64, variants=3, readers=3, processes=2,
              repeats=3, seed=42, extensions="png"):
    """Return measured medians; verify identical arrays before writing report.

    Args:
        path: Input or output filesystem path used by this operation.
        output: Destination path for an array or directory for run results.
        size: Positive side length of the processed square images.
        variants: Number of augmented copies added to each original image.
        readers: Positive number of file-reading threads.
        processes: Positive number of augmentation worker processes.
        repeats: Positive number of sequential/parallel timing comparisons.
        seed: Integer seed for reproducible data or transformation choices.
        extensions: Supported extensions; strings use comma-separated names.

    Returns:
        Measured timings and equality checks also saved as JSON.
    """
    if min(size, variants, readers, processes, repeats) <= 0:
        raise PreparationError("Параметры сравнения должны быть > 0.")
    extensions = normalize_extensions(extensions)
    if ".csv" in extensions:
        raise PreparationError("Аугментация в части 3 принимает изображения.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sequential_path = output / "sequential.npy"
    parallel_path = output / "parallel.npy"
    measured = {"sequential": [], "parallel": []}
    run_info = {}
    for repeat in range(repeats):
        order = ["sequential", "parallel"]
        if repeat % 2:
            order.reverse()
        for mode in order:
            if mode == "sequential":
                result = run_sequential(path, extensions, sequential_path,
                                        size, variants, seed)
            else:
                result = run_parallel(path, extensions, parallel_path, size,
                                      variants, seed, readers, processes)
            measured[mode].append(result["seconds"])
            run_info[mode] = result
            print(f"repeat={repeat + 1} mode={mode} "
                  f"files={result['files']} seconds={result['seconds']:.6f}",
                  flush=True)
        if file_digest(sequential_path) != file_digest(parallel_path):
            raise PreparationError("Результаты двух режимов различны.")
    sequential = statistics.median(measured["sequential"])
    parallel = statistics.median(measured["parallel"])
    report = {
        "python": platform.python_version(), "seed": seed,
        "image_size": size, "variants_per_image": variants,
        "reader_threads": readers, "processes": processes, "repeats": repeats,
        "file_count": run_info["sequential"]["files"],
        "output_samples": run_info["sequential"]["samples"],
        "seconds": measured,
        "median_sequential_seconds": sequential,
        "median_parallel_seconds": parallel,
        "speedup_sequential_over_parallel": sequential / parallel,
        "outputs_identical": True,
        "output_sha256": file_digest(parallel_path),
        "threads_joined": run_info["parallel"]["threads_joined"],
        "remaining_child_processes": len(mp.active_children()),
        "timing_scope": ("scan, read, normalize, augment, write, "
                         "worker lifecycle"),
        "cache_control": "OS cache not reset; execution order alternates",
        "interpretation": ("parallel faster" if parallel < sequential
                           else "sequential faster; process overhead"),
    }
    (output / "performance_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    """Parse ENV/CLI configuration and return a useful nonzero error code.

    Args:
        No explicit arguments.

    Returns:
        Integer exit code: zero for success, two for a reported failure.
    """
    try:
        settings = Settings.from_env()
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--path", type=Path, default=settings.path)
        parser.add_argument("--output-dir", type=Path,
                            default=settings.output_dir)
        parser.add_argument("--ext", default="png")
        parser.add_argument("--size", type=int, default=settings.image_size)
        parser.add_argument("--variants", type=int, default=3)
        parser.add_argument("--readers", type=int, default=3)
        parser.add_argument("--processes", type=int, default=2)
        parser.add_argument("--repeats", type=int, default=3)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--log-level", default=settings.log_level,
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        args = parser.parse_args()
        configure_logging(args.log_level)
        benchmark(args.path, args.output_dir, args.size, args.variants,
                  args.readers, args.processes, args.repeats, args.seed,
                  args.ext)
        return 0
    except PreparationError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
