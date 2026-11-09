"""Argparse entry point: prepare and doctor commands."""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

from .config import PreparationError, Settings
from .logging_config import configure_logging


def doctor():
    """Return Python/dependency/CUDA/cwd diagnostics; GPU is optional."""
    packages = {}
    for name in ["numpy", "PIL", "matplotlib"]:
        try:
            module = importlib.import_module(name)
            packages[name] = {"available": True,
                              "version": getattr(module, "__version__", "?")}
        except ImportError:
            packages[name] = {"available": False}
    frameworks = {name: importlib.util.find_spec(name) is not None
                  for name in ["tensorflow", "torch"]}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False)
        cuda = {"available": result.returncode == 0,
                "message": (result.stdout or result.stderr).strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        cuda = {"available": False, "message": type(error).__name__}
    return {"python": platform.python_version(), "executable": sys.executable,
            "libraries": packages, "optional_frameworks": frameworks,
            "cuda": cuda, "gpu_required": False,
            "working_directory": str(Path.cwd())}


def main():
    """Parse CLI overrides, run the command and return zero or a clear error.
    """
    try:
        settings = Settings.from_env()
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--log-level", default=settings.log_level,
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        commands = parser.add_subparsers(dest="command", required=True)
        preparation = commands.add_parser("prepare")
        preparation.add_argument("--path", type=Path, default=settings.path)
        preparation.add_argument("--ext", default=settings.extensions)
        preparation.add_argument("--output", type=Path,
                                 default=settings.output_dir / "prepared.npy")
        preparation.add_argument("--size", type=int,
                                 default=settings.image_size)
        preparation.add_argument("--chunk-size", type=int,
                                 default=settings.chunk_size)
        commands.add_parser("doctor")
        args = parser.parse_args()
        configure_logging(args.log_level)
        if args.command == "doctor":
            result = doctor()
        else:
            from .scanner import normalize_extensions, prepare
            result = prepare(args.path, normalize_extensions(args.ext),
                             args.output, args.size, args.chunk_size)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.command == "doctor":
            return 0 if all(item["available"] for item in
                            result["libraries"].values()) else 2
        return 0
    except PreparationError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
