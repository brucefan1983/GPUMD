#!/usr/bin/env python3
"""NumPy-only semantic checks for GPUMD regression outputs."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


class PostCheckError(RuntimeError):
    pass


def _close(actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float, label: str) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise PostCheckError(f"{label}: shape differs {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise PostCheckError(f"{label}: non-finite value")
    difference = np.abs(actual - expected)
    scale = np.maximum(np.abs(actual), np.abs(expected))
    relative = np.divide(difference, scale, out=np.zeros_like(difference), where=scale != 0)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        flat = int(np.argmax(difference - (atol + rtol * scale)))
        index = np.unravel_index(flat, actual.shape)
        raise PostCheckError(
            f"{label}: mismatch at {index}: {actual[index]:.17g} != {expected[index]:.17g} "
            f"(rtol={rtol:.3e}, atol={atol:.3e})"
        )
    return {
        "max_abs": float(np.max(difference)) if difference.size else 0.0,
        "max_rel": float(np.max(relative)) if relative.size else 0.0,
    }


def _comment_fields(comment: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for token in shlex.split(comment):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def read_extxyz(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise PostCheckError(f"missing XYZ file: {path.name}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    lines = text.splitlines()
    frames: List[Dict[str, Any]] = []
    cursor = 0
    while cursor < len(lines):
        try:
            natoms = int(lines[cursor].strip())
        except (ValueError, IndexError) as exc:
            raise PostCheckError(f"{path.name}: invalid atom-count line at {cursor + 1}") from exc
        if cursor + 1 >= len(lines):
            raise PostCheckError(f"{path.name}: missing comment line")
        comment = lines[cursor + 1]
        fields = _comment_fields(comment)
        spec = fields.get("Properties")
        if not spec:
            raise PostCheckError(f"{path.name}: missing Properties field")
        parts = spec.split(":")
        if len(parts) % 3 != 0:
            raise PostCheckError(f"{path.name}: malformed Properties field")
        descriptors: List[Tuple[str, str, int]] = []
        for index in range(0, len(parts), 3):
            try:
                count = int(parts[index + 2])
            except ValueError as exc:
                raise PostCheckError(f"{path.name}: malformed property count") from exc
            descriptors.append((parts[index], parts[index + 1], count))
        rows = lines[cursor + 2 : cursor + 2 + natoms]
        if len(rows) != natoms:
            raise PostCheckError(f"{path.name}: truncated frame")
        columns: Dict[str, List[Any]] = {name: [] for name, _, _ in descriptors}
        expected_columns = sum(count for _, _, count in descriptors)
        for row_number, row in enumerate(rows, start=cursor + 3):
            tokens = row.split()
            if len(tokens) != expected_columns:
                raise PostCheckError(
                    f"{path.name}: line {row_number} has {len(tokens)} columns, expected {expected_columns}"
                )
            offset = 0
            for name, kind, count in descriptors:
                values = tokens[offset : offset + count]
                offset += count
                if kind == "S":
                    parsed: Any = values[0] if count == 1 else values
                elif kind == "I":
                    parsed = [int(value) for value in values]
                    parsed = parsed[0] if count == 1 else parsed
                else:
                    parsed = [float(value) for value in values]
                    parsed = parsed[0] if count == 1 else parsed
                columns[name].append(parsed)
        arrays: Dict[str, np.ndarray] = {}
        for name, kind, _ in descriptors:
            dtype = object if kind == "S" else float
            arrays[name] = np.asarray(columns[name], dtype=dtype)
        frames.append({"natoms": natoms, "fields": fields, "arrays": arrays})
        cursor += natoms + 2
    return frames


def _loadtxt(path: Path) -> np.ndarray:
    try:
        data = np.loadtxt(path)
    except (OSError, ValueError) as exc:
        raise PostCheckError(f"cannot load numeric text {path.name}: {exc}") from exc
    return np.atleast_2d(data)


def _check_numeric_row(spec: Mapping[str, Any], workdir: Path) -> Dict[str, Any]:
    data = _loadtxt(workdir / spec["output"])
    row = int(spec.get("row", 0))
    start = int(spec.get("start_column", 0))
    expected = np.asarray(spec["expected"], dtype=float)
    actual = data[row, start : start + expected.size]
    metric = _close(actual, expected, float(spec["rtol"]), float(spec["atol"]), spec["output"])
    metric["checked_values"] = int(expected.size)
    return metric


def _active_uncertainty_from_forces(forces: np.ndarray) -> np.ndarray:
    variance = np.var(forces, axis=0)
    return np.sqrt(np.sum(variance, axis=-1))


def _check_active_uncertainty(spec: Mapping[str, Any], workdir: Path) -> Dict[str, Any]:
    observer_count = int(spec["observers"])
    interval = int(spec["check_interval"])
    threshold = float(spec["threshold"])
    rtol = float(spec["rtol"])
    atol = float(spec["atol"])
    observer_frames = [read_extxyz(workdir / f"observer{i}.xyz") for i in range(observer_count)]
    frame_counts = {len(frames) for frames in observer_frames}
    if len(frame_counts) != 1:
        raise PostCheckError(f"observer frame counts differ: {sorted(frame_counts)}")
    total_frames = frame_counts.pop()
    selected_indices = list(range(interval - 1, total_frames, interval))
    forces = np.asarray(
        [[frame["arrays"]["forces"] for frame in frames] for frames in observer_frames],
        dtype=float,
    )
    sampled_forces = forces[:, selected_indices, :, :]
    per_atom = _active_uncertainty_from_forces(sampled_forces)
    maximum = np.max(per_atom, axis=1)
    active_out = _loadtxt(workdir / "active.out")
    if active_out.shape[0] != len(selected_indices):
        raise PostCheckError(
            f"active.out rows {active_out.shape[0]} != expected {len(selected_indices)}"
        )
    metric = _close(active_out[:, 1], maximum, rtol, atol, "active.out uncertainty")

    active_frames = read_extxyz(workdir / "active.xyz")
    active_mask = maximum > threshold
    expected_indices = np.flatnonzero(active_mask)
    if len(active_frames) != len(expected_indices):
        raise PostCheckError(
            f"active.xyz frames {len(active_frames)} != threshold-selected {len(expected_indices)}"
        )
    selection = spec["expected_selection"]
    if selection == "all" and len(expected_indices) != len(maximum):
        raise PostCheckError("threshold did not select every checked structure")
    if selection == "none" and len(expected_indices) != 0:
        raise PostCheckError("threshold unexpectedly selected structures")
    if selection == "partial" and not (0 < len(expected_indices) < len(maximum)):
        raise PostCheckError("threshold did not select a strict subset of structures")

    for output_index, sample_index in enumerate(expected_indices):
        frame = active_frames[output_index]
        fields = frame["fields"]
        header_uncertainty = float(fields["uncertainty"])
        _close(
            np.asarray([header_uncertainty]),
            np.asarray([maximum[sample_index]]),
            rtol,
            max(atol, 5.0e-8),
            f"active.xyz frame {output_index} header uncertainty",
        )
        arrays = frame["arrays"]
        expected_observer_frame = observer_frames[0][selected_indices[sample_index]]["arrays"]
        for name, enabled in (("vel", spec["has_velocity"]), ("forces", spec["has_force"])):
            if enabled:
                if name not in arrays:
                    raise PostCheckError(f"active.xyz frame {output_index} missing {name}")
                _close(
                    arrays[name],
                    expected_observer_frame[name],
                    rtol,
                    atol,
                    f"active.xyz frame {output_index} {name}",
                )
            elif name in arrays:
                raise PostCheckError(f"active.xyz frame {output_index} unexpectedly contains {name}")
        if spec["has_atom_uncertainty"]:
            if "uncertainty" not in arrays:
                raise PostCheckError("active.xyz missing per-atom uncertainty")
            _close(
                arrays["uncertainty"],
                per_atom[sample_index],
                rtol,
                max(atol, 5.0e-8),
                f"active.xyz frame {output_index} per-atom uncertainty",
            )
        elif "uncertainty" in arrays:
            raise PostCheckError("active.xyz unexpectedly contains per-atom uncertainty")
    metric.update({"checked_samples": len(maximum), "selected_frames": len(active_frames)})
    return metric


def _compute_msd(positions: np.ndarray, window: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    memory = np.zeros((window, positions.shape[1], 3), dtype=float)
    msd = np.zeros((window, 3), dtype=float)
    origins = 0
    for step, current in enumerate(positions):
        correlation_step = step % window
        memory[correlation_step] = current
        if step < window - 1:
            continue
        origins += 1
        for tau in range(window):
            previous = memory[(correlation_step - tau) % window]
            msd[tau] += np.sum((current - previous) ** 2, axis=0)
    return msd / (origins * positions.shape[1])


def _check_msd_from_xyz(spec: Mapping[str, Any], workdir: Path) -> Dict[str, Any]:
    msd = _loadtxt(workdir / spec["output"])
    window = int(spec["window"])
    if msd.shape[0] != window:
        raise PostCheckError(f"{spec['output']}: {msd.shape[0]} rows != window {window}")
    max_abs = 0.0
    for item in spec["trajectories"]:
        frames = read_extxyz(workdir / item["xyz"])
        try:
            positions = np.asarray([frame["arrays"]["unwrapped_position"] for frame in frames])
        except KeyError as exc:
            raise PostCheckError(f"{item['xyz']}: missing unwrapped_position") from exc
        expected = _compute_msd(positions, window)
        start = int(item["column_start"])
        actual = msd[:, start : start + 3]
        metric = _close(actual, expected, float(spec["rtol"]), float(spec["atol"]), item["xyz"])
        max_abs = max(max_abs, metric["max_abs"])
    return {"trajectories": len(spec["trajectories"]), "max_abs": max_abs}


CHECKS = {
    "active_uncertainty": _check_active_uncertainty,
    "msd_from_xyz": _check_msd_from_xyz,
}


def _relative_path(value: Any, owner: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{owner} must be a safe relative path")


def validate_spec(spec: Any, owner: str) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{owner} must be an object")
    name = spec.get("name")
    if name not in CHECKS:
        raise ValueError(f"{owner}.name is unsupported: {name!r}")

    required_by_name = {
        "active_uncertainty": {
            "name", "observers", "check_interval", "threshold", "expected_selection",
            "has_velocity", "has_force", "has_atom_uncertainty", "rtol", "atol",
        },
        "msd_from_xyz": {
            "name", "output", "window", "trajectories", "rtol", "atol",
        },
    }
    required = required_by_name[name]
    missing = required - set(spec)
    unknown = set(spec) - required
    if missing:
        raise ValueError(f"{owner} missing key(s): {sorted(missing)}")
    if unknown:
        raise ValueError(f"{owner} has unknown key(s): {sorted(unknown)}")

    for tolerance in ("rtol", "atol"):
        value = spec[tolerance]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{owner}.{tolerance} must be a finite non-negative number")

    if name == "active_uncertainty":
        for key in ("observers", "check_interval"):
            if not isinstance(spec[key], int) or isinstance(spec[key], bool) or spec[key] <= 0:
                raise ValueError(f"{owner}.{key} must be a positive integer")
        if spec["expected_selection"] not in {"all", "partial", "none"}:
            raise ValueError(f"{owner}.expected_selection is invalid")
        for key in ("has_velocity", "has_force", "has_atom_uncertainty"):
            if not isinstance(spec[key], bool):
                raise ValueError(f"{owner}.{key} must be boolean")
        if not isinstance(spec["threshold"], (int, float)) or not math.isfinite(spec["threshold"]):
            raise ValueError(f"{owner}.threshold must be finite")
    elif name == "msd_from_xyz":
        _relative_path(spec["output"], f"{owner}.output")
        if not isinstance(spec["window"], int) or isinstance(spec["window"], bool) or spec["window"] <= 0:
            raise ValueError(f"{owner}.window must be a positive integer")
        trajectories = spec["trajectories"]
        if not isinstance(trajectories, list) or not trajectories:
            raise ValueError(f"{owner}.trajectories must be a non-empty list")
        for index, item in enumerate(trajectories):
            if not isinstance(item, dict) or set(item) != {"xyz", "column_start"}:
                raise ValueError(f"{owner}.trajectories[{index}] requires xyz and column_start")
            _relative_path(item["xyz"], f"{owner}.trajectories[{index}].xyz")
            if not isinstance(item["column_start"], int) or isinstance(item["column_start"], bool) or item["column_start"] < 0:
                raise ValueError(f"{owner}.trajectories[{index}].column_start must be non-negative")


def run(specs: Sequence[Mapping[str, Any]], workdir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for spec in specs:
        name = spec["name"]
        try:
            metric = CHECKS[name](spec, workdir)
        except (OSError, ValueError, KeyError, IndexError, PostCheckError) as exc:
            if isinstance(exc, PostCheckError):
                raise
            raise PostCheckError(f"{name}: {exc}") from exc
        results.append({"name": name, "metrics": metric})
    return results
