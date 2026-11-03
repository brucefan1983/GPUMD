"""Compare machine-readable TB outputs; never inspect terminal messages."""

import math
import re


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def near(a, b, atol=2e-5, rtol=2e-4):
    check(math.isfinite(a) and math.isfinite(b), "non-finite comparison")
    check(abs(a - b) <= atol + rtol * abs(b), f"{a:.10g} != {b:.10g}")


def numbers(path):
    rows = [[float(x) for x in line.split()] for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    check(rows and all(math.isfinite(x) for row in rows for x in row), f"invalid numbers in {path}")
    return rows


def check_training_outputs(directory, metadata):
    losses = numbers(directory / "loss.out")
    check([row[0] for row in losses] == [1, 2], "wrong epoch sequence")
    check(all(len(row) == 7 for row in losses), "wrong loss column count")
    bands = numbers(directory / "band_train.out")
    count = sum(len(s["kpoints"]) for s in metadata["structures"])
    nb = metadata["band_max"]
    check(len(bands) == count and all(len(row) == 5 + 2*nb for row in bands), "incomplete band output")
    cursor = 0
    for sid, source in enumerate(metadata["structures"]):
        for kid, (kpoint, reference) in enumerate(zip(source["kpoints"], source["reference"])):
            row = bands[cursor]
            check(row[:2] == [sid, kid], "wrong structure/k-point index")
            for actual, expected in zip(row[2:5], kpoint):
                near(actual, expected, atol=2e-7, rtol=2e-6)
            for actual, expected in zip(row[5+nb:], reference):
                near(actual, expected, atol=2e-6, rtol=2e-6)
            cursor += 1
    validate_model(directory / "tb.txt", metadata)


def read_model(path):
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    check(rows and rows[0] == ["tb", "2"], f"unexpected model identifier/version in {path}")
    i = next(i for i, row in enumerate(rows) if row[0] == "parameters")
    header = rows[:i]
    count = int(rows[i][1])
    check(count > 0, "empty parameter block")
    values = rows[i+1:i+1+count]
    check(len(values) == count and all(len(row) == 1 for row in values), "invalid parameter block")
    parameters = [float(row[0]) for row in values]
    i += 1 + count
    check(i < len(rows) and rows[i][0] == "q_scaler", "missing q_scaler block")
    count = int(rows[i][1])
    values = rows[i+1:]
    check(count > 0 and len(values) == count and all(len(row) == 1 for row in values), "invalid q_scaler block")
    scalers = [float(row[0]) for row in values]
    check(all(math.isfinite(x) for x in parameters + scalers), "non-finite model data")
    check(all(x > 0 for x in scalers), "nonpositive descriptor scale")
    return header, parameters, scalers


def validate_model(path, metadata):
    header, _, _ = read_model(path)
    fields = {row[0]: row[1:] for row in header}
    check(fields["basis"] == [metadata["basis"]], "wrong model basis")
    check(fields["type"] == [str(len(metadata["elements"]))] + list(metadata["elements"]),
          "wrong model elements")
    check(fields["band_range"] == ["0", str(metadata["band_max"])], "wrong model band range")


def compare_outputs(old, new, atol=2e-5, rtol=2e-4, parameter_atol=None):
    """Compare training outputs, excluding the loss timing column.

    Integer indices/counts and categorical metadata must agree exactly.
    Floating values use |new-old| <= atol + rtol*|old|. Whitespace and
    decimal formatting do not matter. parameter_atol overrides atol only
    for model parameters. Return per-file maximum differences.
    """
    errors, maxima = [], {}

    def value(a, b, label, absolute_tolerance=atol):
        check(math.isfinite(a) and math.isfinite(b), "non-finite " + label)
        difference = abs(a-b)
        filename = label.split(":", 1)[0]
        maxima[filename] = max(maxima.get(filename, 0.0), difference)
        if difference > absolute_tolerance + rtol*abs(a) and len(errors) < 12:
            errors.append(f"{label}: old={a:.10g}, new={b:.10g}, abs_diff={difference:.3g}")

    def sequence(a, b, label, absolute_tolerance=atol):
        check(len(a) == len(b), label + " size differs")
        for i, (x, y) in enumerate(zip(a, b)):
            value(x, y, f"{label}[{i}]", absolute_tolerance)

    def tokens(a, b, label):
        check(len(a) == len(b), label + " token count differs")
        for i, (x, y) in enumerate(zip(a, b)):
            if re.fullmatch(r"[+-]?\d+", x) and re.fullmatch(r"[+-]?\d+", y):
                check(int(x) == int(y), label + " integer differs")
                continue
            try:
                u, v = float(x), float(y)
            except ValueError:
                check(x == y, label + " metadata differs")
            else:
                if math.isinf(u) or math.isinf(v):
                    check(u == v, label + " infinity differs")
                else:
                    value(u, v, f"{label}[{i}]")

    for filename in ("loss.out", "band_train.out"):
        a, b = numbers(old / filename), numbers(new / filename)
        check(len(a) == len(b), filename + " row count differs")
        for i, (x, y) in enumerate(zip(a, b)):
            check(len(x) == len(y), filename + " column count differs")
            indices = 1 if filename == "loss.out" else 2
            check(x[:indices] == y[:indices], filename + " indices differ")
            # loss.out: epoch, lr, total, mean-|parameter|, band, dispersion, time.
            sequence(x[indices:-1] if filename == "loss.out" else x[indices:],
                     y[indices:-1] if filename == "loss.out" else y[indices:], f"{filename}:row{i}")

    ha, pa, qa = read_model(old / "tb.txt")
    hb, pb, qb = read_model(new / "tb.txt")
    check(len(ha) == len(hb), "model header size differs")
    for a, b in zip(ha, hb):
        tokens(a, b, "tb.txt:header")
    sequence(pa, pb, "tb.txt:parameters", atol if parameter_atol is None else parameter_atol)
    sequence(qa, qb, "tb.txt:q_scaler")

    check(not errors, "numerical regression:\n" + "\n".join(errors))
    return maxima
