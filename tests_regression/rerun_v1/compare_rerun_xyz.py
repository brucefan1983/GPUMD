#!/usr/bin/env python3
"""Compare GPUMD reference and rerun Extended XYZ frames numerically."""

import math
import re
import sys


def read_frames(path):
    frames = []
    with open(path) as stream:
        while True:
            count_line = stream.readline()
            if not count_line:
                return frames
            if not count_line.strip():
                continue
            count = int(count_line)
            comment = stream.readline().strip()

            def attr(key):
                match = re.search(
                    r"(?:^|\s)" + re.escape(key) + r'=("[^"]*"|[^ \t]+)', comment
                )
                if not match:
                    raise ValueError(f"missing {key} in {path}")
                return match.group(1).strip('"')

            descriptions = attr("Properties").split(":")
            offsets, widths = {}, {}
            cursor = 0
            for i in range(0, len(descriptions), 3):
                name, _, width = descriptions[i], descriptions[i + 1], int(descriptions[i + 2])
                offsets[name], widths[name] = cursor, width
                cursor += width
            rows = [stream.readline().split() for _ in range(count)]
            if any(len(row) != cursor for row in rows):
                raise ValueError(f"wrong atom column count in {path}")

            def column(name):
                start, width = offsets[name], widths[name]
                return [row[start : start + width] for row in rows]

            frames.append(
                {
                    "count": count,
                    "step": int(attr("Step")),
                    "time": float(attr("Time")),
                    "energy": float(attr("energy")),
                    "virial": [float(value) for value in attr("virial").split()],
                    "lattice": [float(value) for value in attr("Lattice").split()],
                    "pbc": attr("pbc").split(),
                    "species": [row[0] for row in column("species")],
                    "positions": [float(value) for row in column("pos") for value in row],
                    "forces": [float(value) for row in column("forces") for value in row],
                }
            )


def main(reference_path, rerun_path, expected_frames, dt_fs):
    reference = read_frames(reference_path)
    rerun = read_frames(rerun_path)
    if len(reference) != 1:
        raise SystemExit(f"reference must contain exactly one frame, found {len(reference)}")
    if len(rerun) != expected_frames:
        raise SystemExit(f"expected {expected_frames} rerun frames, found {len(rerun)}")
    ref = reference[0]
    max_energy = max_force = max_virial = max_position = max_cell = 0.0
    force_sum_sq = virial_sum_sq = virial_ref_sum_sq = 0.0
    force_count = 0
    for index, frame in enumerate(rerun):
        expected_step = index + 1
        if frame["step"] != expected_step:
            raise SystemExit(f"output Step {frame['step']} != expected {expected_step}")
        if abs(frame["time"] - expected_step * dt_fs) > max(1e-7, 1e-7 * expected_step * dt_fs):
            raise SystemExit(f"output Time inconsistent with Step {expected_step}")
        if frame["count"] != ref["count"] or frame["species"] != ref["species"]:
            raise SystemExit("atom count/type/order changed")
        if frame["pbc"] != ref["pbc"]:
            raise SystemExit("PBC flags changed")
        if len(frame["forces"]) != len(ref["forces"]):
            raise SystemExit("force output is missing or has the wrong size")
        if len(frame["virial"]) != 9 or len(ref["virial"]) != 9:
            raise SystemExit("full 9-component Virial is missing")
        max_energy = max(max_energy, abs(frame["energy"] - ref["energy"]))
        for a, b in zip(frame["forces"], ref["forces"]):
            delta = a - b
            max_force = max(max_force, abs(delta))
            force_sum_sq += delta * delta
            force_count += 1
        for a, b in zip(frame["virial"], ref["virial"]):
            delta = a - b
            max_virial = max(max_virial, abs(delta))
            virial_sum_sq += delta * delta
            virial_ref_sum_sq += b * b
        for a, b in zip(frame["positions"], ref["positions"]):
            max_position = max(max_position, abs(a - b))
        for a, b in zip(frame["lattice"], ref["lattice"]):
            max_cell = max(max_cell, abs(a - b))

    force_rmse = math.sqrt(force_sum_sq / force_count)
    virial_relative = math.sqrt(virial_sum_sq / virial_ref_sum_sq) if virial_ref_sum_sq else 0.0
    print(
        f"PASS frames={expected_frames} max_dE={max_energy:.8g} eV "
        f"force_RMSE={force_rmse:.8g} eV/A force_max={max_force:.8g} eV/A "
        f"virial_max={max_virial:.8g} eV virial_relative={virial_relative:.8g} "
        f"position_max={max_position:.8g} A box_max={max_cell:.8g} A"
    )
    if max_energy > 1e-5 or force_rmse >= 1e-5 or max_virial > 1e-5:
        raise SystemExit("numerical consistency tolerance exceeded")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: compare_rerun_xyz.py REFERENCE RERUN EXPECTED_FRAMES DT_FS")
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
