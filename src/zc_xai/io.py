"""Small, dependency-light helpers for reproducible experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    """Hash an array's dtype, shape, and C-order bytes reproducibly."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def sha256_json(document: dict[str, Any]) -> str:
    """Hash a JSON-compatible mapping with stable key ordering."""

    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return document


@contextmanager
def atomic_output_path(path: Path, *, overwrite: bool) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically publish it."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}\nUse --overwrite to replace it."
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.incomplete-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"No output was written to {temporary_path}.")
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        if overwrite:
            temporary_path.replace(destination)
        else:
            os.link(temporary_path, destination)
            temporary_path.unlink()
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, document: dict[str, Any], *, overwrite: bool) -> None:
    with (
        atomic_output_path(path, overwrite=overwrite) as temporary_path,
        temporary_path.open("w", encoding="utf-8") as stream,
    ):
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_npz(
    path: Path, *, overwrite: bool, compressed: bool = True, **arrays: Any
) -> None:
    """Atomically write named NumPy arrays without suffix surprises."""

    saver = np.savez_compressed if compressed else np.savez
    with (
        atomic_output_path(path, overwrite=overwrite) as temporary_path,
        temporary_path.open("wb") as stream,
    ):
        saver(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
