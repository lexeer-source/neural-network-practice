"""One path producer, several file readers and a bounded process pool."""

import logging
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time

import numpy as np

from part2_prepare.app.config import PreparationError
from part2_prepare.app.scanner import load_image, scan_files
from .augment import augment_image

LOGGER = logging.getLogger(__name__)


def _put_cancellable(channel, item, cancelled):
    while not cancelled.is_set():
        try:
            channel.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _create_output(path, count, size, variants):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    mapped = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32,
        shape=(count * (variants + 1), size, size))
    return mapped, temporary


def run_sequential(directory, extensions, output, size, variants, seed):
    """Read, augment and write in one process; include all stages in timing.

    Args:
        directory: Root directory scanned recursively for input files.
        extensions: Supported extensions; strings use comma-separated names.
        output: Destination path for an array or directory for run results.
        size: Positive side length of the processed square images.
        variants: Number of augmented copies added to each original image.
        seed: Integer seed for reproducible data or transformation choices.

    Returns:
        Dictionary containing elapsed seconds and item counts.
    """
    started = time.perf_counter()
    paths = list(scan_files(directory, extensions))
    if not paths:
        raise PreparationError("Нет изображений для обработки.")
    mapped, temporary = _create_output(output, len(paths), size, variants)
    try:
        for index, path in enumerate(paths):
            _, result = augment_image(index, load_image(path, size),
                                      variants, seed)
            start = index * (variants + 1)
            mapped[start:start + variants + 1] = result
        mapped.flush()
        del mapped
        mapped = None
        temporary.replace(output)
    finally:
        if mapped is not None:
            del mapped
        temporary.unlink(missing_ok=True)
    return {"seconds": time.perf_counter() - started, "files": len(paths),
            "samples": len(paths) * (variants + 1)}


def run_parallel(directory, extensions, output, size=64, variants=3,
                 seed=42, readers=3, processes=2, queue_size=8):
    """Run bounded thread/Pool pipeline; join all workers on success or error.

    Args:
        directory: Root directory scanned recursively for input files.
        extensions: Supported extensions; strings use comma-separated names.
        output: Destination path for an array or directory for run results.
        size: Positive side length of the processed square images.
        variants: Number of augmented copies added to each original image.
        seed: Integer seed for reproducible data or transformation choices.
        readers: Positive number of file-reading threads.
        processes: Positive number of augmentation worker processes.
        queue_size: Positive bound for queued or pending work items.

    Returns:
        Timing, counts and status after all workers have stopped.
    """
    if min(size, variants, readers, processes, queue_size) <= 0:
        raise PreparationError("Параметры конвейера должны быть > 0.")
    started = time.perf_counter()
    paths_queue = queue.Queue(maxsize=queue_size)
    ready_queue = queue.Queue(maxsize=queue_size)
    cancelled = threading.Event()
    lock = threading.Lock()
    pending_slots = threading.BoundedSemaphore(queue_size)
    counts = {"read": 0, "completed": 0}
    errors = []
    mapped = temporary = pool = None
    workers = []

    def producer():
        """Scan once and deliver ordered paths followed by stop signals.

        Args:
            No explicit arguments; uses instance state or console input.

        Returns:
            None; updates or displays application state.
        """
        try:
            paths = list(scan_files(directory, extensions))
            _put_cancellable(ready_queue, ("header", len(paths), None),
                             cancelled)
            for index, path in enumerate(paths):
                if not _put_cancellable(paths_queue, (index, path), cancelled):
                    break
            LOGGER.info("Producer scanned %d files", len(paths))
        except (PreparationError, OSError) as error:
            _put_cancellable(ready_queue, ("error", str(error), None),
                             cancelled)
        finally:
            for _ in range(readers):
                _put_cancellable(paths_queue, None, cancelled)

    def consumer():
        """Read one image at a time and report failures without abandoning
        queues.

        Args:
            No explicit arguments; uses instance state or console input.

        Returns:
            None; updates or displays application state.
        """
        try:
            while not cancelled.is_set():
                try:
                    task = paths_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if task is None:
                        break
                    index, path = task
                    image = load_image(path, size)
                    with lock:
                        counts["read"] += 1
                    LOGGER.info("Read %s", path.name)
                    _put_cancellable(ready_queue, ("image", index, image),
                                     cancelled)
                except (PreparationError, OSError, ValueError) as error:
                    _put_cancellable(ready_queue, ("error", str(error), None),
                                     cancelled)
                finally:
                    paths_queue.task_done()
        finally:
            _put_cancellable(ready_queue, ("done", None, None), cancelled)

    def completed(result):
        """Place a completed process result at its deterministic output
        offset.

        Args:
            result: Completed worker result: source index and image batch.

        Returns:
            None; updates or displays application state.
        """
        try:
            index, images = result
            start = index * (variants + 1)
            mapped[start:start + variants + 1] = images
            with lock:
                counts["completed"] += 1
        except (OSError, ValueError, IndexError) as error:
            with lock:
                errors.append(str(error))
        finally:
            pending_slots.release()

    def failed(error):
        """Record an asynchronous worker failure and free its pending slot.

        Args:
            error: Exception reported by an asynchronous augmentation worker.

        Returns:
            None; updates or displays application state.
        """
        with lock:
            errors.append(str(error))
        pending_slots.release()

    try:
        # spawn works on Windows too and does not fork a multithreaded parent.
        pool = mp.get_context("spawn").Pool(processes=processes)
        workers = [threading.Thread(target=producer, name="NN-producer")]
        workers.extend(threading.Thread(target=consumer,
                                        name=f"NN-reader-{index + 1}")
                       for index in range(readers))
        for worker in workers:
            worker.start()
        kind, count, _ = ready_queue.get(timeout=30)
        if kind != "header" or count == 0:
            raise PreparationError(f"Нет входных данных: {count}")
        mapped, temporary = _create_output(output, count, size, variants)
        finished_readers = 0
        while finished_readers < readers:
            kind, value, image = ready_queue.get(timeout=30)
            if kind == "done":
                finished_readers += 1
            elif kind == "error":
                with lock:
                    errors.append(value)
            elif kind == "image":
                if not pending_slots.acquire(timeout=60):
                    raise PreparationError("Истекло время ожидания процессов.")
                pool.apply_async(augment_image, (value, image, variants, seed),
                                 callback=completed, error_callback=failed)
        paths_queue.join()
        for worker in workers:
            worker.join()
        pool.close()
        pool.join()
        pool = None
        if errors or counts["read"] != count or counts["completed"] != count:
            raise PreparationError("; ".join(errors) or "Неполная обработка")
        mapped.flush()
        del mapped
        mapped = None
        temporary.replace(output)
        LOGGER.info("Processed %d files; readers and processes stopped", count)
        return {"seconds": time.perf_counter() - started, "files": count,
                "samples": count * (variants + 1), "counters": counts.copy(),
                "threads_joined": all(not worker.is_alive()
                                      for worker in workers)}
    except (queue.Empty, OSError, ValueError) as error:
        raise PreparationError(f"Ошибка конвейера: {error}")
    finally:
        cancelled.set()
        for worker in workers:
            worker.join(timeout=10)
        if pool is not None:
            pool.terminate()
            pool.join()
        if mapped is not None:
            del mapped
        if temporary is not None:
            temporary.unlink(missing_ok=True)
