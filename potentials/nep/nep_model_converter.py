#!/usr/bin/env python3
# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD. If not, see <https://www.gnu.org/licenses/>.

"""Convert legacy NEP4 model files and optional restarts to the new NEP format."""

import argparse
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile


MODELS = {
    "": ("nep", 0, 0, 1, 1),
    "_temperature": ("nep_temperature", 1, 0, 1, 1),
    "_charge1": ("qnep1", 0, 1, 2, 1),
    "_charge2": ("qnep2", 0, 1, 2, 1),
    "_dipole": ("nep_dipole", 0, 0, 1, 1),
    "_polarizability": ("nep_polarizability", 0, 0, 1, 2),
    "_vdw": ("nep_vdw", 0, 1, 1, 1),
    "_charge_vdw": ("qnep2_vdw", 0, 2, 2, 1),
}
ELEMENTS = {
    element.lower(): element for element in (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
        "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
        "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au "
        "Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu"
    ).split()
}
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
FLOAT_MAX = float.fromhex("0x1.fffffep+127")


def records(stream):
    for line_number, line in enumerate(stream, 1):
        text = line.partition("#")[0].rstrip("\r\n")
        if any((ord(c) < 32 and c != "\t") or ord(c) >= 127 for c in text):
            raise ValueError(f"{stream.name}:{line_number}: invalid character")
        tokens = text.split()
        if tokens:
            yield f"{stream.name}:{line_number}", tokens


def next_record(source, filename):
    try:
        return next(source)
    except StopIteration:
        raise ValueError(f"{filename}: incomplete model header") from None


def read_field(source, filename, name, minimum, maximum):
    where, tokens = next_record(source, filename)
    if tokens[0].lower() != name or not minimum <= len(tokens) - 1 <= maximum:
        raise ValueError(f"{where}: malformed {name} line")
    return f"{where}: {name}", tokens[1:]


def integer(token, minimum, maximum, where):
    if not re.fullmatch(r"[0-9]+", token):
        raise ValueError(f"{where}: invalid integer '{token}'")
    value = int(token)
    if not minimum <= value <= maximum:
        raise ValueError(f"{where}: integer must be in [{minimum}, {maximum}]")
    return value


def float_value(token, where):
    if not NUMBER.fullmatch(token):
        raise ValueError(f"{where}: invalid number '{token}'")
    value = float(token)
    nonzero = any(c in "123456789" for c in re.split("[eE]", token)[0])
    if not math.isfinite(value) or abs(value) > FLOAT_MAX or (value == 0.0 and nonzero):
        raise ValueError(f"{where}: number outside the float range: '{token}'")
    rounded = struct.unpack("f", struct.pack("f", value))[0]
    if rounded == 0.0 and nonzero:
        raise ValueError(f"{where}: number underflows float: '{token}'")
    return rounded


def read_model(filename):
    with filename.open(encoding="utf-8-sig") as stream:
        source = records(stream)
        where, header = next_record(source, filename)
        tag = header[0].lower()
        if not tag.startswith("nep4"):
            raise ValueError(f"{where}: expected a legacy NEP4 model, found '{header[0]}'")
        suffix = tag[4:]
        has_zbl = suffix.startswith("_zbl")
        if has_zbl:
            suffix = suffix[4:]
        if suffix not in MODELS:
            raise ValueError(f"{where}: unsupported model '{header[0]}'")
        name, extra_inputs, extra_outputs, globals_count, ann_count = MODELS[suffix]
        tensor = name in ("nep_dipole", "nep_polarizability")
        if has_zbl and tensor:
            raise ValueError(f"{where}: ZBL is not supported for {name}")
        if len(header) < 3:
            raise ValueError(f"{where}: missing element count or element list")
        num_types = integer(header[1], 1, len(ELEMENTS), where)
        if len(header) != num_types + 2:
            raise ValueError(f"{where}: expected {num_types} element symbols")
        elements = []
        for token in header[2:]:
            element = ELEMENTS.get(token.lower())
            if element is None or element in elements:
                raise ValueError(f"{where}: unknown or duplicate element '{token}'")
            elements.append(element)

        fields = []
        flexible_zbl = False
        if has_zbl:
            where, values = read_field(source, filename, "zbl", 2, 3)
            inner, outer = [float_value(v, where) for v in values[:2]]
            flexible_zbl = inner == 0.0 and outer == 0.0
            if flexible_zbl:
                if len(values) != 2:
                    raise ValueError(f"{where}: flexible ZBL cannot have a cutoff factor")
            else:
                if inner < 0.0 or outer <= inner:
                    raise ValueError(f"{where}: ZBL requires 0 <= inner < outer")
                if len(values) == 3 and float_value(values[2], where) < 0.5:
                    raise ValueError(f"{where}: ZBL cutoff factor must be at least 0.5")
            fields.append(("zbl", values))

        where, values = read_field(source, filename, "cutoff", 4, 2 * num_types + 2)
        if len(values) not in (4, 2 * num_types + 2):
            raise ValueError(f"{where}: expected two cutoffs or one pair per element")
        if name in ("qnep1", "qnep2") and len(values) != 4:
            raise ValueError(f"{where}: {name} supports only uniform cutoffs")
        for index in range(0, len(values) - 2, 2):
            radial, angular = [float_value(v, where) for v in values[index:index + 2]]
            if not 3.0 <= angular <= radial <= 100.0:
                raise ValueError(f"{where}: cutoffs require 3 <= angular <= radial <= 100")
        for token in values[-2:]:
            integer(token, 0, 2147483647, where)
        fields.append(("cutoff", values))

        where, values = read_field(source, filename, "n_max", 2, 2)
        nr = integer(values[0], 0, 15, where)
        na = integer(values[1], 0, 12, where)
        fields.append(("n_max", values))
        where, values = read_field(source, filename, "basis_size", 2, 2)
        br = integer(values[0], 0, 16, where)
        ba = integer(values[1], 0, 12, where)
        fields.append(("basis_size", values))
        where, values = read_field(source, filename, "l_max", 3, 7)
        l_max = integer(values[0], 2, 8, where)
        flags = [integer(v, 0, 2 if i == 0 else 1, where) for i, v in enumerate(values[1:])]
        if flags[0] == 1:
            raise ValueError(f"{where}: q_222 must be 0 or 2")
        angular_dim = (na + 1) * (l_max + sum(v != 0 for v in flags))
        dim = nr + 1 + angular_dim + extra_inputs
        if angular_dim > 90 or dim > 103:
            raise ValueError(f"{where}: descriptor dimension exceeds the supported limit")
        fields.append(("l_max", values))
        where, values = read_field(source, filename, "ann", 2, 2)
        n1 = integer(values[0], 1, 120, where)
        n2 = integer(values[1], 0, 120, where)
        if n2 and name not in ("nep", "nep_temperature", "nep_dipole", "nep_polarizability"):
            raise ValueError(f"{where}: {name} supports only one hidden layer")
        fields.append(("ann", values))

        ann_size = (dim + 1) * n1 + (n1 + 2) * n2 if n2 else (dim + 2 + extra_outputs) * n1
        parameters = (num_types * ann_size + globals_count) * ann_count
        parameters += num_types**2 * ((nr + 1) * (br + 1) + (na + 1) * (ba + 1))
        zbl_count = 5 * num_types * (num_types + 1) if flexible_zbl else 0
        expected = parameters + dim + zbl_count
        data = []
        for where, tokens in source:
            if len(tokens) != 1:
                raise ValueError(f"{where}: expected one numeric value per legacy model data line")
            if len(data) == expected:
                raise ValueError(f"{where}: unexpected data after {expected} model values")
            float_value(tokens[0], where)
            data.append(tokens[0])
        if len(data) != expected:
            raise ValueError(
                f"{filename}: expected {expected} values ({parameters} parameters, "
                f"{dim} scalers, {zbl_count} ZBL parameters), found {len(data)}"
            )
        if flexible_zbl:
            for index in range(parameters + dim, expected, 10):
                inner = float_value(data[index], str(filename))
                outer = float_value(data[index + 1], str(filename))
                if inner < 0.0 or outer <= inner:
                    raise ValueError(f"{filename}: each flexible ZBL pair requires 0 <= inner < outer")

    return {
        "name": name,
        "elements": elements,
        "fields": fields,
        "parameters": parameters,
        "dim": dim,
        "zbl_count": zbl_count,
        "data": data,
        "references": 0 if tensor else num_types,
    }


def channel_major(values, model):
    fields = dict(model["fields"])
    nr, na = map(int, fields["n_max"])
    br, ba = map(int, fields["basis_size"])
    sizes = ((nr + 1) * (br + 1), (na + 1) * (ba + 1))
    pairs = len(model["elements"])**2
    offset = model["parameters"] - pairs * sum(sizes)
    reordered = list(values[:offset])
    for size in sizes:
        for pair in range(pairs):
            reordered.extend(values[offset + pair:offset + pairs * size:pairs])
        offset += pairs * size
    reordered.extend(values[offset:])
    return reordered


def model_text(model):
    elements = model["elements"]
    lines = [f"nep {len(elements)} " + " ".join(elements), "format_version 1", "model " + model["name"]]
    lines.extend(name + " " + " ".join(values) for name, values in model["fields"])
    if model["references"]:
        lines.append("energy_reference " + " ".join(["0"] * model["references"]))
    data = channel_major(model["data"], model)
    offset = 0
    for name, count in (
        ("parameters", model["parameters"]),
        ("q_scaler", model["dim"]),
        ("zbl_parameters", model["zbl_count"]),
    ):
        if count:
            lines.append(f"@ {name} {count}")
            lines.extend(data[offset:offset + count])
            offset += count
    return "\n".join(lines) + "\n"


def restart_text(filename, model):
    references = model["references"]
    lines = ["nep_restart", f"energy_reference {references}" + " 0" * references,
             f"parameters {model['parameters']}"]
    count = 0
    with filename.open(encoding="utf-8-sig") as stream:
        for where, tokens in records(stream):
            if len(tokens) != 2:
                raise ValueError(f"{where}: expected a legacy restart with two columns (mu sigma)")
            if count == model["parameters"]:
                raise ValueError(f"{where}: unexpected data after {count} mu/sigma pairs")
            float_value(tokens[0], where)
            if float_value(tokens[1], where) < 0.0:
                raise ValueError(f"{where}: sigma must be nonnegative")
            lines.append(" ".join(tokens))
            count += 1
    if count != model["parameters"]:
        raise ValueError(f"{filename}: expected {model['parameters']} mu/sigma pairs, found {count}")
    lines[3:] = channel_major(lines[3:], model)
    return "\n".join(lines) + "\n"


def same_file(first, second):
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def check_output_paths(inputs, outputs, force):
    for i, output in enumerate(outputs):
        if any(same_file(output, source) for source in inputs):
            raise ValueError(f"{output}: output must not overwrite an input file")
        if any(same_file(output, previous) for previous in outputs[:i]):
            raise ValueError("model and restart outputs must be different files")
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise ValueError(f"{output}: output must be a regular file")
        if output.exists() and not force:
            raise ValueError(f"{output}: output exists; use --force to overwrite it")
        if not output.parent.is_dir():
            raise ValueError(f"{output}: output directory does not exist")


def write_outputs(outputs, force):
    temporary = []
    try:
        for path, text in outputs:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                prefix=".nep_converter_", delete=False,
            ) as stream:
                temporary.append((Path(stream.name), path))
                stream.write(text)
        for source, target in temporary:
            if not force and target.exists():
                raise ValueError(f"{target}: output exists; use --force to overwrite it")
            os.replace(source, target)
    finally:
        for source, _ in temporary:
            if source.exists():
                source.unlink()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Supports standard NEP4, temperature, charge1/2, dipole, polarizability, "
            "vdW and charge-vdW models. Energy references are set to zero. "
            "Writes type-pair descriptors in channel-major order; NEP3 and NEP5 are not supported. "
            "Conversion does not extend native MD support or preserve an old training trajectory."
        ),
    )
    parser.add_argument("input", type=Path, help="legacy model file")
    parser.add_argument("output", type=Path, help="new-format model file")
    parser.add_argument("--restart", type=Path, help="optional matching legacy restart")
    parser.add_argument("--restart-output", type=Path, help="new-format restart file")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()
    if (args.restart is None) != (args.restart_output is None):
        parser.error("--restart and --restart-output must be supplied together")
    try:
        inputs = [args.input] + ([args.restart] if args.restart else [])
        outputs = [args.output] + ([args.restart_output] if args.restart_output else [])
        check_output_paths(inputs, outputs, args.force)
        model = read_model(args.input)
        results = [(args.output, model_text(model))]
        if args.restart:
            results.append((args.restart_output, restart_text(args.restart, model)))
        write_outputs(results, args.force)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    print(f"{args.input} -> {args.output}: {model['name']}, "
          f"{len(model['elements'])} elements, {model['parameters']} parameters")
    if args.restart:
        print(f"{args.restart} -> {args.restart_output}: {model['parameters']} mu/sigma pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
