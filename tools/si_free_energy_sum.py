#!/usr/bin/env python3
import argparse
import math
import sys

import yaml


REFERENCE_KEYS = ("spring_species", "spring_constants", "uf_self_pairs", "uf_cross_pairs")
META_KEYS = ("T", "V", "P", "N_total", "F_ref")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def close(a, b, name):
    if not math.isclose(float(a), float(b), rel_tol=1.0e-8, abs_tol=1.0e-10):
        raise ValueError(f"{name} differs between stage files: {a} vs {b}")


def require_keys(data, path):
    required = (
        "stage",
        "T",
        "V",
        "P",
        "N_total",
        "W_forward",
        "W_backward",
        "delta_F",
        "F_ref",
        "spring_species",
        "spring_constants",
        "uf_self_pairs",
        "uf_cross_pairs",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"{path} is missing required key '{key}'")


def stage_value(data):
    value = data["stage"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("stage values must be 1 and 2")
    return value


def ordered_stages(first, second):
    first_stage = stage_value(first)
    second_stage = stage_value(second)
    stages = {first_stage, second_stage}
    if stages != {1, 2}:
        raise ValueError("stage values must be 1 and 2")
    return (first, second) if first_stage == 1 else (second, first)


def validate_compatible(stage1, stage2):
    for key in REFERENCE_KEYS:
        if stage1[key] != stage2[key]:
            raise ValueError(f"reference definitions differ for '{key}'")
    for key in META_KEYS:
        close(stage1[key], stage2[key], key)


def summarize(stage1, stage2):
    f_ref = float(stage1["F_ref"])
    delta1 = float(stage1["delta_F"])
    delta2 = float(stage2["delta_F"])
    volume = float(stage1["V"])
    pressure = float(stage1["P"])
    f_target = f_ref + delta1 + delta2
    return {
        "stage1": {
            "W_forward": float(stage1["W_forward"]),
            "W_backward": float(stage1["W_backward"]),
            "delta_F": delta1,
        },
        "stage2": {
            "W_forward": float(stage2["W_forward"]),
            "W_backward": float(stage2["W_backward"]),
            "delta_F": delta2,
        },
        "F_ref": f_ref,
        "F_target": f_target,
        "G_target": f_target + pressure * volume,
        "T": float(stage1["T"]),
        "V": volume,
        "P": pressure,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize two-stage superionic TI YAML files.")
    parser.add_argument("stage_a")
    parser.add_argument("stage_b")
    parser.add_argument("-o", "--output", default="ti_superionic_summary.yaml")
    args = parser.parse_args(argv)

    try:
        first = load_yaml(args.stage_a)
        second = load_yaml(args.stage_b)
        require_keys(first, args.stage_a)
        require_keys(second, args.stage_b)
        stage1, stage2 = ordered_stages(first, second)
        validate_compatible(stage1, stage2)
        data = summarize(stage1, stage2)
        with open(args.output, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
