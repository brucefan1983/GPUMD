"""PPPM initial mesh and trajectory checks, separate from numerical tolerances."""

import math
import re
from pathlib import Path

import numpy as np

from post_checks import read_extxyz


NUMBER = r"[-+0-9.eE]+"
INITIAL = re.compile(
    rf"PPPM mesh: (\d+) x (\d+) x (\d+) \(target spacing ({NUMBER}) A; "
    rf"actual spacing ({NUMBER}) ({NUMBER}) ({NUMBER}) A\)\."
)


def validate_spec(spec):
    required = {"spacing", "initial_mesh"}
    optional = {"frames", "last_cell"}
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - required - optional:
        raise ValueError("pppm requires spacing and initial_mesh; optional frames, last_cell")
    if not isinstance(spec["spacing"], (int, float)) or isinstance(spec["spacing"], bool) or not math.isfinite(spec["spacing"]) or spec["spacing"] <= 0:
        raise ValueError("pppm spacing must be finite and positive")
    meshes = [] if spec["initial_mesh"] is None else [spec["initial_mesh"]]
    for mesh in meshes:
        if not isinstance(mesh, list) or len(mesh) != 3 or any(type(n) is not int or n < 16 for n in mesh):
            raise ValueError("pppm meshes must contain three integers >= 16")
    if "frames" in spec and (type(spec["frames"]) is not int or spec["frames"] < 1):
        raise ValueError("pppm frames must be positive")
    if "last_cell" in spec:
        cell = spec["last_cell"]
        if "frames" not in spec or not isinstance(cell, list) or len(cell) != 9 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in cell):
            raise ValueError("pppm last_cell requires frames and nine finite values")


def good_size(n):
    if n < 16 or n % 2:
        return False
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def strip_mesh_lines(data):
    """Only used for a case whose candidate mesh contract has already passed."""
    kept = []
    for line in data.splitlines(keepends=True):
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not INITIAL.fullmatch(text):
            kept.append(line)
    return b"".join(kept)


def check(spec, result):
    text = Path(result["stdout_path"]).read_text(encoding="utf-8")
    initial = []
    for line in text.splitlines():
        if line.startswith("PPPM "):
            match = INITIAL.fullmatch(line)
            if not match:
                raise ValueError("unexpected or malformed PPPM mesh output")
            initial.append(match.groups())
    expected = spec["initial_mesh"]
    if expected is None:
        if initial:
            raise ValueError("PPPM without a force evaluation must not create a mesh")
        return {"initial_mesh": None}
    if len(initial) != 1:
        raise ValueError(f"expected one initial mesh line, got {len(initial)}")
    mesh = [int(x) for x in initial[0][:3]]
    spacing = float(initial[0][3])
    actual = [float(x) for x in initial[0][4:]]
    if mesh != expected or not all(good_size(n) for n in mesh):
        raise ValueError(f"initial mesh {mesh} != {expected}, or invalid FFT size")
    if spacing != spec["spacing"] or not all(math.isfinite(h) and 0 < h <= spacing * (1 + 1e-12) for h in actual):
        raise ValueError("invalid target or actual PPPM spacing")
    if "frames" in spec:
        frames = read_extxyz(Path(result["workdir"]) / "state.xyz")
        if len(frames) != spec["frames"]:
            raise ValueError(f"state.xyz has {len(frames)} frames, expected {spec['frames']}")
        if "last_cell" in spec:
            cell = np.array([float(x) for x in frames[-1]["fields"]["Lattice"].split()])
            if not np.allclose(cell, spec["last_cell"], rtol=0, atol=1e-8):
                raise ValueError("final cell does not match the prescribed deformation")
    return {"initial_mesh": mesh}
