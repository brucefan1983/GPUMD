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


"""Deterministic NEP89 inputs; all coordinates and velocities use XYZ units."""

import math
from pathlib import Path
import random

from checks import CheckError, frames, header

POTENTIAL = 'potentials/nep/nep89_20250409/nep89_20250409.txt'
WATER = 'tests_pytest/fixtures/structures/water-nat63-from-md.xyz'
PROPERTIES = 'precision double mass velocity force potential virial group_labels unwrapped_position'
NAMES = ('axis_x', 'axis_y', 'axis_z', 'auto', 'triclinic', 'alloy',
         'nonuniform', 'empty', 'partial_pbc', 'migration', 'nvt', 'npt',
         'multi_run', 'append', 'restart', 'water', 'energy_solid', 'energy_water')
PARTITIONS = 3
PERFORMANCE = {'solid': (4, 5, 6), 'water': (27, 18, 36)}
WARMUP = 10
STEPS = 100
REPEATS = 3


def prepare_atoms(atoms, rng):
    rng.shuffle(atoms)  # Input IDs deliberately differ from spatial order.
    for i, atom in enumerate(atoms):
        atom[3] += i * 1e-8  # Unique mass tags independently check atom order.
    total_mass = math.fsum(atom[3] for atom in atoms)
    mean = [math.fsum(atom[3] * atom[2][j] for atom in atoms) / total_mass for j in range(3)]
    for atom in atoms:
        for j in range(3):
            atom[2][j] -= mean[j]


def crystal(variant='solid', axis=0):
    # For rc=6 A, each of three partitions needs >=10 bins of width rc/2.
    # The 130 A cutting direction leaves room for skew and NPT changes.
    a = 3.615
    dims = [8, 8, 8]
    dims[axis] = 12 * PARTITIONS
    box = [[a * dims[i] if i == j else 0.0 for j in range(3)] for i in range(3)]
    if variant == 'triclinic':
        box[1][0], box[2][1] = 1.2, 0.8
    rng = random.Random(1906)
    atoms = []
    basis = ((0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0))
    for ix in range(dims[0]):
        for iy in range(dims[1]):
            for iz in range(dims[2]):
                for b in basis:
                    frac = [(i + s + .015 + rng.uniform(-.002, .002)) / d
                            for i, s, d in zip((ix, iy, iz), b, dims)]
                    if variant == 'empty' and not (.18 / PARTITIONS < frac[0] < .78 / PARTITIONS):
                        continue
                    if variant == 'nonuniform' and frac[0] > .4 and rng.random() < .6:
                        continue
                    pos = [sum(frac[k] * box[k][j] for k in range(3)) for j in range(3)]
                    element = rng.choice(('Cu', 'Ni', 'Al')) if variant == 'alloy' else 'Cu'
                    mass = {'Cu': 63.546, 'Ni': 58.6934, 'Al': 26.9815385}[element]
                    sigma = math.sqrt(8.617343e-5 * 300 / mass) / 10.18051
                    atoms.append([element, pos, [rng.gauss(0, sigma) for _ in range(3)], mass])
    prepare_atoms(atoms, rng)
    if variant == 'migration':
        for atom in atoms:
            atom[2][axis] -= .5  # A/fs; rigid translation exercises slab/PBC crossings.
    return {'box': box, 'pbc': (True, False, True) if variant == 'partial_pbc'
            else (True, True, True), 'atoms': atoms}


def water(repo, repeats=(12, 2, 4)):
    # Reuse an MD snapshot of 21 water molecules; no melting/equilibration run.
    # Default copies give 6048 atoms in a 102.17 x 24.56 x 24.72 A periodic box.
    frame = only_frame(Path(repo) / WATER)
    species = frame['data'].get('species', [])
    if (frame['n'] != 63 or species.count('H') != 42 or species.count('O') != 21 or
            frame['schema'] != {'species': ('S', 1), 'pos': ('R', 3)} or
            frame['meta'].get('pbc', '').upper().split() != ['T', 'T', 'T']):
        raise CheckError('Expected the periodic 63-atom liquid-water snapshot: ' + WATER)
    lattice = header(frame, 'lattice', 9)
    box = [lattice[i:i + 3] for i in (0, 3, 6)]
    rng = random.Random(1906)
    atoms = []
    for ix in range(repeats[0]):
        for iy in range(repeats[1]):
            for iz in range(repeats[2]):
                shift = [sum(n * row[j] for n, row in zip((ix, iy, iz), box)) for j in range(3)]
                for i, element in enumerate(species):
                    pos = [frame['data']['pos'][3 * i + j] + shift[j] for j in range(3)]
                    mass = {'H': 1.008, 'O': 15.999}[element]
                    sigma = math.sqrt(8.617343e-5 * 300 / mass) / 10.18051
                    atoms.append([element, pos, [rng.gauss(0, sigma) for _ in range(3)], mass])
    prepare_atoms(atoms, rng)
    return {'box': [[value * copies for value in row] for copies, row in zip(repeats, box)],
            'pbc': (True, True, True), 'atoms': atoms}


def write_state(path, state):
    with Path(path).open('w', encoding='utf-8') as out:
        out.write('{}\nLattice="{}" pbc="{}" '.format(
            len(state['atoms']), ' '.join(format(x, '.17g') for row in state['box'] for x in row),
            ' '.join('T' if p else 'F' for p in state['pbc'])))
        out.write('Properties=species:S:1:pos:R:3:vel:R:3:mass:R:1:group:I:1\n')
        for i, (element, pos, vel, mass) in enumerate(state['atoms']):
            out.write(element + ' ' + ' '.join(format(x, '.17g') for x in pos + vel + [mass]) +
                      ' {}\n'.format(i % 3))


def only_frame(path):
    iterator = frames(path)
    first = next(iterator, None)
    if first is None or next(iterator, None) is not None:
        raise CheckError('{}: expected exactly one XYZ frame'.format(path))
    return first


def snapshot_to_input(source, destination):
    # Retain species, mass, velocities and group/ID order from a real restart.
    frame = only_frame(source)
    required = {'species': ('S', 1), 'pos': ('R', 3), 'vel': ('R', 3),
                'mass': ('R', 1), 'group': ('I', 1)}
    if frame['schema'] != required:
        raise CheckError('{}: unexpected restart schema'.format(source))
    Path(destination).write_bytes(Path(source).read_bytes())


def segment(steps=10, interval=5, ensemble='nve', selected=True, restart=True):
    text = 'ensemble {}\ndump_thermo {}\n'.format(ensemble, interval)
    text += 'dump_xyz {} state.xyz {}\n'.format(interval, PROPERTIES)
    if selected:
        text += 'dump_xyz {} selected.xyz group 0 1 {}\n'.format(interval, PROPERTIES)
    if restart:
        text += 'dump_restart {}\n'.format(steps)
    return text + 'run {}\n'.format(steps)


def correctness(name):
    axis = 'xyz'.index(name[-1]) if name.startswith('axis_') else 0
    direction = '' if name == 'auto' else ' ' + 'xyz'[axis]
    text = 'potential potential.txt{}\ntime_step .05\n'.format(direction)
    spec = {'name': name, 'axis': axis, 'executions': 1, 'times': [.25, .5],
            'steps': [10], 'selected': True, 'kind': 'short', 'changing_box': False}
    if name.startswith('energy_'):
        # A short NVE check; this does not establish long-time energy stability.
        dt = '.1' if name == 'energy_water' else '.25'
        text = 'potential potential.txt x\ntime_step {}\nensemble nve\n'.format(dt)
        text += 'dump_thermo 10\nrun 100\n'
        spec.update(kind='energy', times=[], selected=False, steps=[100], samples=10)
    elif name == 'multi_run':
        text += segment(5, 5)
        text += 'ensemble nve\nrun 5\nchange_box .01 .02 .03\n' + segment(5, 5)
        spec.update(times=[.25, .75], steps=[5, 5, 5], changing_box=True)
    else:
        ensemble = {'nvt': 'nvt_ber 300 300 100',
                    'npt': 'npt_ber 300 300 100 0 100 1000'}.get(name, 'nve')
        text += segment(ensemble=ensemble)
        spec['changing_box'] = name == 'npt'
        if name in ('nvt', 'npt'):
            spec['berendsen'] = {'temperature': 300, 'tau': 100}
        if name == 'append':
            spec.update(executions=2, times=[.25, .5, .25, .5])
    return text, spec


def performance(phase):
    tile = PERFORMANCE[phase]
    dt = '.25' if phase == 'water' else '.5'
    text = 'replicate {} {} {}\npotential potential.txt x\n'.format(*tile)
    text += 'time_step {}\nensemble nve\nrun {}\n'.format(dt, WARMUP)
    # One final thermo sample synchronizes GPU work before GPUMD stops its timer.
    text += 'ensemble nve\ndump_thermo {}\nrun {}\n'.format(STEPS, STEPS)
    return text
