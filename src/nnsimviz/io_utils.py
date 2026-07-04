"""Save/load utilities for configs and simulation outputs.

Kept intentionally small: JSON for configs (human-readable, diff-friendly)
and CSV for the weight matrix and activity, so results can be inspected or
reloaded later. Depends only on configs.py, simulation.py, and the stdlib +
NumPy.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .configs import ProjectConfig
from .simulation import SimulationResult


# --------------------------------------------------------------------------- #
# Config <-> JSON
# --------------------------------------------------------------------------- #
def config_to_json(config: ProjectConfig) -> str:
    """Serialize a ProjectConfig to a pretty JSON string."""
    return json.dumps(config.to_dict(), indent=2)


def config_from_json(text: str) -> ProjectConfig:
    """Rebuild a ProjectConfig from a JSON string."""
    return ProjectConfig.from_dict(json.loads(text))


def save_config(config: ProjectConfig, path: str | Path) -> None:
    """Write a ProjectConfig to a JSON file."""
    Path(path).write_text(config_to_json(config), encoding="utf-8")


def load_config(path: str | Path) -> ProjectConfig:
    """Read a ProjectConfig from a JSON file."""
    return config_from_json(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Arrays -> CSV (as strings, for Streamlit download buttons)
# --------------------------------------------------------------------------- #
def weight_matrix_to_csv(weight_matrix: np.ndarray) -> str:
    """Return the weight matrix as a CSV string."""
    buf = io.StringIO()
    np.savetxt(buf, weight_matrix, delimiter=",", fmt="%.6f")
    return buf.getvalue()


def activity_to_csv(result: SimulationResult) -> str:
    """Return activity (neurons x timesteps) as a CSV string with a time header."""
    buf = io.StringIO()
    header = "," + ",".join(f"t={t:.3f}" for t in result.time)
    rows = [header]
    for i, row in enumerate(result.activity):
        rows.append(f"neuron_{i}," + ",".join(f"{v:.6f}" for v in row))
    buf.write("\n".join(rows))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Imported weight matrices
# --------------------------------------------------------------------------- #
def validate_weight_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    """Validate and return a recurrent weight matrix as a float NumPy array.

    The simulator expects W to be square, finite, and 2-D.
    W[i, j] is the connection from source neuron j to target neuron i.
    """
    W = np.asarray(weight_matrix, dtype=float)

    if W.ndim != 2:
        raise ValueError("Imported weight matrix must be 2-D.")

    if W.shape[0] != W.shape[1]:
        raise ValueError(
            "Imported weight matrix must be square, "
            f"got shape {W.shape}."
        )

    if W.shape[0] == 0:
        raise ValueError("Imported weight matrix cannot be empty.")

    if not np.isfinite(W).all():
        raise ValueError("Imported weight matrix must contain only finite values.")

    return W


def load_weight_matrix_from_upload(filename: str, data: bytes) -> np.ndarray:
    """Load a weight matrix from uploaded CSV, NPY, or NPZ bytes.

    Supported formats:
        .csv: numeric matrix, comma-separated
        .npy: single NumPy array
        .npz: array named 'W', or a file containing exactly one array
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".csv":
        text = data.decode("utf-8")
        matrix = np.loadtxt(io.StringIO(text), delimiter=",")
        return validate_weight_matrix(matrix)

    if suffix == ".npy":
        matrix = np.load(io.BytesIO(data), allow_pickle=False)
        return validate_weight_matrix(matrix)

    if suffix == ".npz":
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            if "W" in loaded.files:
                matrix = loaded["W"]
            elif len(loaded.files) == 1:
                matrix = loaded[loaded.files[0]]
            else:
                raise ValueError(
                    "NPZ import must contain an array named 'W', "
                    "or exactly one array."
                )
        return validate_weight_matrix(matrix)

    raise ValueError(
        "Unsupported network import format. Use .csv, .npy, or .npz."
    )