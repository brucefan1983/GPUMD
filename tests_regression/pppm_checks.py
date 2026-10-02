"""PPPM mesh contracts, separate from baseline/candidate numerical tolerances.

The initial mesh check is permanent. The diagnostic trace check is temporary
V1 validation for GPUMD_PPPM_DIAGNOSTICS and can be removed with that macro.
"""

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
DIAGNOSTIC = re.compile(
    rf"PPPM diagnostic: (\d+) (\d+) (\d+); thickness "
    rf"({NUMBER}) ({NUMBER}) ({NUMBER}); rebuild ([01])"
)


def validate_spec(spec):
    required = {"spacing", "initial_mesh"}
    optional = {"mesh_sequence", "frames", "last_cell"}
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - required - optional:
        raise ValueError("pppm requires spacing and initial_mesh; optional mesh_sequence, frames, last_cell")
    if not isinstance(spec["spacing"], (int, float)) or isinstance(spec["spacing"], bool) or not math.isfinite(spec["spacing"]) or spec["spacing"] <= 0:
        raise ValueError("pppm spacing must be finite and positive")
    meshes = [] if spec["initial_mesh"] is None else [spec["initial_mesh"]]
    if "mesh_sequence" in spec:
        if not isinstance(spec["mesh_sequence"], list) or not spec["mesh_sequence"]:
            raise ValueError("pppm mesh_sequence must be nonempty")
        meshes += spec["mesh_sequence"]
        if spec["mesh_sequence"][0] != spec["initial_mesh"]:
            raise ValueError("pppm mesh_sequence must start at initial_mesh")
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
        if not INITIAL.fullmatch(text) and not DIAGNOSTIC.fullmatch(text):
            kept.append(line)
    return b"".join(kept)


def check(spec, result, require_diagnostics=False):
    text = Path(result["stdout_path"]).read_text(encoding="utf-8")
    initial = []
    trace = []
    for line in text.splitlines():
        if line.startswith("PPPM mesh:"):
            match = INITIAL.fullmatch(line)
            if not match:
                raise ValueError("malformed PPPM initial mesh line")
            initial.append(match.groups())
        elif line.startswith("PPPM diagnostic:"):
            match = DIAGNOSTIC.fullmatch(line)
            if not match:
                raise ValueError("malformed PPPM diagnostic line")
            trace.append(match.groups())
    expected = spec["initial_mesh"]
    if expected is None:
        if initial or trace:
            raise ValueError("PPPM without a force evaluation must not create a mesh")
        return {"initial_mesh": None, "diagnostic_evaluations": 0}
    if len(initial) != 1:
        raise ValueError(f"expected one initial mesh line, got {len(initial)}")
    mesh = [int(x) for x in initial[0][:3]]
    spacing = float(initial[0][3])
    actual = [float(x) for x in initial[0][4:]]
    if mesh != expected or not all(good_size(n) for n in mesh):
        raise ValueError(f"initial mesh {mesh} != {expected}, or invalid FFT size")
    if spacing != spec["spacing"] or not all(math.isfinite(h) and 0 < h <= spacing * (1 + 1e-12) for h in actual):
        raise ValueError("invalid target or actual PPPM spacing")
    previous = None
    sequence = []
    for record in trace:
        current = [int(x) for x in record[:3]]
        thickness = [float(x) for x in record[3:6]]
        rebuilt = int(record[6])
        if not all(good_size(n) for n in current):
            raise ValueError("invalid diagnostic FFT size")
        if not all(math.isfinite(h) and 0 < h <= n * spacing * (1 + 1e-12) for h, n in zip(thickness, current)):
            raise ValueError("diagnostic mesh exceeds target spacing")
        if previous and any(n < old for n, old in zip(current, previous)):
            raise ValueError("PPPM mesh shrank")
        changed = current != previous
        if rebuilt != int(changed):
            raise ValueError("PPPM rebuilt a mesh unnecessarily or missed a required rebuild")
        if changed:
            sequence.append(current)
        previous = current
    if trace and sequence[0] != expected:
        raise ValueError("initial and diagnostic meshes disagree")
    if "mesh_sequence" in spec:
        if require_diagnostics and not trace:
            raise ValueError("missing temporary PPPM diagnostics; rebuild with -DGPUMD_PPPM_DIAGNOSTICS")
        if trace and sequence != spec["mesh_sequence"]:
            raise ValueError(f"mesh sequence {sequence} != {spec['mesh_sequence']}")
    if "frames" in spec:
        frames = read_extxyz(Path(result["workdir"]) / "state.xyz")
        if len(frames) != spec["frames"]:
            raise ValueError(f"state.xyz has {len(frames)} frames, expected {spec['frames']}")
        if "last_cell" in spec:
            cell = np.array([float(x) for x in frames[-1]["fields"]["Lattice"].split()])
            if not np.allclose(cell, spec["last_cell"], rtol=0, atol=1e-8):
                raise ValueError("final cell does not match the prescribed deformation")
    return {
        "initial_mesh": mesh,
        "diagnostic_evaluations": len(trace),
        "mesh_sequence": sequence,
        "dynamic_trace_checked": bool(trace) if "mesh_sequence" in spec else None,
    }
