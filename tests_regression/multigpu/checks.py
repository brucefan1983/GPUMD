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
# Major contributors: Zheyong Fan (NEP multi-GPU release tests).


"""Strict standard-library readers and independent MD output checks."""

from array import array
import hashlib
import math
from pathlib import Path
import re


TOLERANCES = {
    'pos': (0.00002, 0),
    'unwrapped_position': (0.00002, 0),
    'vel': (0.00002, 0),
    'force': (0.0005, 0.00002),
    'potential': (0.00005, 0),
    'virial': (0.002, 0.00002),
    'box': (0.000001, 0),
    'temperature': (0.003, 0.00002),
    'pressure': (0.003, 0.00002),
    'energy_solid': (0.0005, 0),
    'energy_water': (0.002, 0),
}


class CheckError(Exception):
    pass

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def numbers(values, label):
    try:
        result = [float(x) for x in values]
    except (TypeError, ValueError) as error:
        raise CheckError('{}: invalid numeric value ({})'.format(label, error))
    if not all(math.isfinite(x) for x in result):
        raise CheckError('{}: non-finite value'.format(label))
    return result

def metric(left, right, atol, rtol=0):
    if len(left) != len(right):
        raise CheckError('Array length differs: {} != {}'.format(len(left), len(right)))
    if not left:
        raise CheckError('Empty numeric comparison')
    maximum, scaled_squared, exceedances, first = 0.0, 0.0, 0, []
    for i, (a, b) in enumerate(zip(left, right)):
        if not math.isfinite(a) or not math.isfinite(b):
            raise CheckError('Non-finite numeric output at index {}'.format(i))
        error = abs(a - b)
        if not math.isfinite(error):
            raise CheckError('Numeric difference overflows at index {}'.format(i))
        if error > maximum:
            scaled_squared = 1.0 + scaled_squared * (maximum / error) ** 2
            maximum = error
        elif error:
            scaled_squared += (error / maximum) ** 2
        if error > atol + rtol * abs(b):
            exceedances += 1
            if len(first) < 4:
                first.append({'index': i, 'new': a, 'reference': b, 'abs_error': error})
    return {'values': len(left), 'max_abs': maximum, 'rms': maximum * math.sqrt(scaled_squared / len(left)),
            'exceedances': exceedances, 'atol': atol, 'rtol': rtol, 'first_mismatches': first}

def frames(path):
    aliases = {'forces': 'force', 'energy_atom': 'potential'}
    with Path(path).open(encoding='utf-8') as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                count = int(line.strip())
            except ValueError:
                raise CheckError('{}: malformed XYZ atom count'.format(path))
            if count <= 0:
                raise CheckError('{}: nonpositive XYZ atom count'.format(path))
            header = stream.readline()
            meta = {k.lower(): quoted if quoted else bare
                    for k, quoted, bare in re.findall(r'(\w+)=(?:"([^\"]*)"|([^\s]+))', header)}
            raw_schema = meta.get('properties', '').split(':')
            if not raw_schema or len(raw_schema) % 3:
                raise CheckError('{}: missing/malformed Properties'.format(path))
            schema, data, order = {}, {}, []
            for name, kind, width in zip(raw_schema[::3], raw_schema[1::3], raw_schema[2::3]):
                name = aliases.get(name, name)
                try:
                    width = int(width)
                except ValueError:
                    raise CheckError('{}: invalid XYZ property width for {}'.format(path, name))
                if name in schema or kind not in ('S', 'R', 'I') or width < 1:
                    raise CheckError('{}: invalid XYZ schema'.format(path))
                schema[name] = (kind, width)
                data[name] = [] if kind == 'S' else array('d')
                order.append(name)
            columns = sum(width for _, width in schema.values())
            for atom in range(count):
                fields = stream.readline().split()
                if len(fields) != columns:
                    raise CheckError('{}: incomplete/wrong-width atom row {}'.format(path, atom))
                offset = 0
                for name in order:
                    kind, width = schema[name]
                    values = fields[offset:offset + width]
                    if kind == 'S':
                        data[name].extend(values)
                    else:
                        parsed = numbers(values, '{} atom {} {}'.format(path, atom, name))
                        if kind == 'I' and any(x != int(x) for x in parsed):
                            raise CheckError('{}: noninteger {}'.format(path, name))
                        data[name].extend(parsed)
                    offset += width
            yield {'n': count, 'schema': schema, 'data': data, 'meta': meta}

def thermo_rows(path, expected=None):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            stripped = line.strip()
            # GPUMD writes a comment header at the start of each run segment.
            if not stripped or stripped.startswith('#'):
                continue
            row = numbers(line.split(), str(path))
            if len(row) != 18:
                raise CheckError('{}: expected 18 thermo columns, got {}'.format(path, len(row)))
            rows.append(row)
    if not rows or (expected is not None and len(rows) != expected):
        raise CheckError('{}: expected {} thermo rows, got {}'.format(path, expected, len(rows)))
    return rows


SCHEMA = {'species': ('S', 1), 'pos': ('R', 3), 'mass': ('R', 1),
          'vel': ('R', 3), 'force': ('R', 3), 'potential': ('R', 1),
          'virial': ('R', 9), 'group': ('I', 1), 'unwrapped_position': ('R', 3)}
RESTART_SCHEMA = {k: SCHEMA[k] for k in ('species', 'pos', 'mass', 'vel', 'group')}


def exact(condition, label):
    if not condition:
        raise CheckError(label)


def lattice(frame):
    box = numbers(frame['meta'].get('lattice', '').split(), 'XYZ lattice')
    exact(len(box) == 9, 'XYZ lattice must have nine entries')
    volume = (box[0] * (box[4] * box[8] - box[5] * box[7]) -
              box[1] * (box[3] * box[8] - box[5] * box[6]) +
              box[2] * (box[3] * box[7] - box[4] * box[6]))
    exact(math.isfinite(volume) and volume > 0, 'Nonpositive XYZ cell volume')
    return box


def select(values, indices, width):
    return [values[i * width + j] for i in indices for j in range(width)]


def check_identity(frame, initial, indices, schema):
    exact(frame['n'] == len(indices) and frame['schema'] == schema,
          'Unexpected XYZ atom count or property schema')
    lattice(frame)
    exact(frame['meta'].get('pbc') == initial['meta'].get('pbc'), 'PBC changed')
    for key in ('species', 'mass', 'group'):
        expected = select(initial['data'][key], indices, 1)
        exact(list(frame['data'][key]) == expected, key + ': atom identity/order changed')


def header(frame, key, width):
    values = numbers(frame['meta'].get(key, '').split(), 'XYZ header ' + key)
    exact(len(values) == width, 'Invalid XYZ header ' + key)
    return values


def check_run(directory, initial, spec):
    """Check each executable independently before any baseline comparison."""
    directory = Path(directory)
    n = initial['n']
    metrics = {}
    if spec['kind'] == 'energy':
        rows = thermo_rows(directory / 'thermo.out', spec['samples'])
        exact(all(r[0] >= 0 and r[1] >= 0 for r in rows), 'Negative temperature or kinetic energy')
        energy = [(r[1] + r[2]) / n for r in rows]
        drift = [x - energy[0] for x in energy]
        metrics['NVE_drift_eV_per_atom'] = metric(drift, [0.0] * len(drift),
                                                  *TOLERANCES[spec['name']])
        exact(not (directory / 'state.xyz').exists(), 'Unexpected trajectory in energy run')
        return metrics

    times = spec['times']
    rows = thermo_rows(directory / 'thermo.out', len(times))
    full = list(frames(directory / 'state.xyz'))
    exact(len(full) == len(times), 'Wrong number of full-system frames')
    indices = list(range(n))
    for i, (frame, time, row) in enumerate(zip(full, times, rows)):
        check_identity(frame, initial, indices, SCHEMA)
        prefix = 'frame:{}:'.format(i)
        metrics[prefix + 'time_fs'] = metric(header(frame, 'time', 1), [time], 1e-8)
        box = lattice(frame)
        if not spec['changing_box']:
            metrics[prefix + 'fixed_box'] = metric(box, lattice(initial), 1e-10)
        metrics[prefix + 'thermo_box'] = metric(box, row[9:], 1e-9, 5e-11)
        data = frame['data']
        pe = math.fsum(data['potential'])
        ke = .5 * math.fsum(data['mass'][j] * (data['vel'][3 * j + d] * 10.18051) ** 2
                             for j in range(n) for d in range(3))
        metrics[prefix + 'potential_sum'] = metric([pe / n], [row[2] / n], 1e-8, 5e-11)
        metrics[prefix + 'header_energy'] = metric([pe / n], [header(frame, 'energy', 1)[0] / n], 1e-8)
        # Berendsen thermo is evaluated before velocity rescaling, whereas
        # dump_xyz sees the rescaled velocities. Honor that established timing.
        temperature = row[0]
        expected_ke = row[1]
        if 'berendsen' in spec:
            settings = spec['berendsen']
            temperature += (settings['temperature'] - temperature) / settings['tau']
            expected_ke = 1.5 * n * 8.617343e-5 * temperature
        metrics[prefix + 'kinetic_sum'] = metric([ke / n], [expected_ke / n], 1e-7, 1e-8)
        metrics[prefix + 'temperature_from_ke'] = metric([2 * ke / (3 * n * 8.617343e-5)],
                                                         [temperature], 1e-5, 1e-8)
        total_virial = header(frame, 'virial', 9)
        # The header is symmetric; its upper triangle is the sum of the matching
        # per-atom components. Do not assume per-atom tensors are symmetric.
        for d in (0, 1, 2, 4, 5, 8):
            metrics[prefix + 'virial_sum:{}'.format(d)] = metric(
                [math.fsum(data['virial'][d::9]) / n], [total_virial[d] / n], 1e-8)
        header(frame, 'stress', 9)  # Reject absent/non-finite metadata, too.
        metrics[prefix + 'net_force_per_atom'] = metric(
            [math.fsum(data['force'][d::3]) / n for d in range(3)], [0.0] * 3, 1e-5)

    if spec['selected']:
        selected = list(frames(directory / 'selected.xyz'))
        exact(len(selected) == len(full), 'Wrong number of group frames')
        ids = [i for i, g in enumerate(initial['data']['group']) if g == 1]
        for i, (part, whole) in enumerate(zip(selected, full)):
            check_identity(part, initial, ids, SCHEMA)
            exact(part['meta'] == whole['meta'], 'Group/full header mismatch')
            for key, (_, width) in SCHEMA.items():
                exact(list(part['data'][key]) == select(whole['data'][key], ids, width),
                      'Group selection mismatch: frame {} {}'.format(i, key))
    restart = list(frames(directory / 'restart.xyz'))
    exact(len(restart) == 1, 'Restart should contain exactly one frame')
    check_identity(restart[0], initial, indices, RESTART_SCHEMA)
    for key in ('pos', 'vel'):
        metrics['restart:' + key] = metric(restart[0]['data'][key], full[-1]['data'][key], 1e-10)
    metrics['restart:box'] = metric(lattice(restart[0]), lattice(full[-1]), 1e-10)

    if spec['name'] == 'migration':
        # Partition bins follow the same geometric rule as the documented slab
        # method. Require observable crossings; a stationary fixture is inadequate.
        length = lattice(initial)[0]
        bins = int(length / 3)
        exact(bins >= 30, 'Migration box is too small for three NEP partitions')
        width = (bins // 3) * length / bins
        before = initial['data']['pos'][::3]
        after = full[-1]['data']['unwrapped_position'][::3]
        crossing = sum(math.floor(a / width) != math.floor(b / width)
                       for a, b in zip(before, after))
        exact(crossing > 0, 'Migration fixture did not cross any slab/PBC boundary')
        metrics['crossing_atoms'] = {'count': crossing, 'exceedances': 0}
    return metrics


def compare_runs(candidate, reference, spec, n):
    if spec['kind'] == 'energy':
        # Each path checks its short NVE drift independently. Compare the first
        # sampled total energy as well; this is not a long-time stability test.
        a = thermo_rows(Path(candidate) / 'thermo.out', spec['samples'])[0]
        b = thermo_rows(Path(reference) / 'thermo.out', spec['samples'])[0]
        return {'initial_total_energy_per_atom': metric([(a[1] + a[2]) / n],
                                                       [(b[1] + b[2]) / n],
                                                       *TOLERANCES['potential'])}
    checks = {}
    filenames = ['state.xyz', 'restart.xyz'] + (['selected.xyz'] if spec['selected'] else [])
    for filename in filenames:
        a = list(frames(Path(candidate) / filename))
        b = list(frames(Path(reference) / filename))
        exact(len(a) == len(b), filename + ': frame counts differ')
        for i, (left, right) in enumerate(zip(a, b)):
            prefix = '{}:{}:'.format(filename, i)
            exact(left['n'] == right['n'] and left['schema'] == right['schema'], prefix + 'schema differs')
            for key in left['data']:
                if key in ('species', 'group', 'mass'):
                    exact(left['data'][key] == right['data'][key], prefix + key + ' differs')
                else:
                    checks[prefix + key] = metric(left['data'][key], right['data'][key], *TOLERANCES[key])
            checks[prefix + 'box'] = metric(lattice(left), lattice(right), *TOLERANCES['box'])
    a = thermo_rows(Path(candidate) / 'thermo.out', len(spec['times']))
    b = thermo_rows(Path(reference) / 'thermo.out', len(spec['times']))
    for d in range(18):
        key = 'temperature' if d == 0 else ('potential' if d < 3 else ('pressure' if d < 9 else 'box'))
        factor = n if d in (1, 2) else 1
        checks['thermo:{}'.format(d)] = metric([r[d] / factor for r in a],
                                              [r[d] / factor for r in b], *TOLERANCES[key])
    return checks
