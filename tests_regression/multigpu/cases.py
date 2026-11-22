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

from checks import CheckError, frames

POTENTIAL = 'potentials/nep/nep89_20250409/nep89_20250409.txt'
PROPERTIES = 'precision double mass velocity force potential virial group_labels unwrapped_position'
NAMES = ('axis_x', 'axis_y', 'axis_z', 'auto', 'triclinic', 'alloy',
         'nonuniform', 'empty', 'partial_pbc', 'migration', 'nvt', 'npt',
         'multi_run', 'append', 'restart', 'liquid', 'energy_solid', 'energy_liquid')
TILES = {'small': (1, 2, 2), 'medium': (2, 3, 3), 'large': (4, 4, 4), 'xlarge': (4, 5, 6)}


def crystal(partitions, variant='solid', axis=0):
    # For rc=6 A, each partition needs >=10 bins of width rc/2. Leave room for
    # skew and NPT changes. Dimensions depend on the largest partition count.
    a = 3.80 if variant == 'liquid_seed' else 3.615
    dims = [8, 8, 8]
    dims[axis] = 12 * partitions
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
                    if variant == 'empty' and not (.18 / partitions < frac[0] < .78 / partitions):
                        continue
                    if variant == 'nonuniform' and frac[0] > .4 and rng.random() < .6:
                        continue
                    pos = [sum(frac[k] * box[k][j] for k in range(3)) for j in range(3)]
                    element = rng.choice(('Cu', 'Ni', 'Al')) if variant == 'alloy' else 'Cu'
                    mass = {'Cu': 63.546, 'Ni': 58.6934, 'Al': 26.9815385}[element]
                    sigma = math.sqrt(8.617343e-5 * 300 / mass) / 10.18051
                    atoms.append([element, pos, [rng.gauss(0, sigma) for _ in range(3)], mass])
    rng.shuffle(atoms)  # Input IDs deliberately differ from spatial order.
    mean = [math.fsum(atom[3] * atom[2][j] for atom in atoms) /
            math.fsum(atom[3] for atom in atoms) for j in range(3)]
    for i, atom in enumerate(atoms):
        for j in range(3):
            atom[2][j] -= mean[j]
        if variant == 'migration':
            atom[2][axis] -= .5  # A/fs; rigid translation exercises slab/PBC crossings.
        atom[3] += i * 1e-8  # Small unique mass tags independently check atom order.
    return {'box': box, 'pbc': (True, False, True) if variant == 'partial_pbc'
            else (True, True, True), 'atoms': atoms}


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
        text = 'potential potential.txt x\ntime_step .25\nensemble nve\n'
        text += 'dump_thermo 10\nrun 2000\n'
        spec.update(kind='energy', times=[], selected=False, steps=[2000], samples=200)
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


def liquid_preparation():
    # Melt a Cu crystal, cool to 2000 K, then measure diffusion at constant T.
    text = ('potential potential.txt\nvelocity 3000 seed 1906\ntime_step 1\n'
            'ensemble nvt_ber 3000 3000 100\nrun 5000\n'
            'ensemble nvt_ber 3000 2000 100\nrun 2000\n'
            'ensemble nvt_ber 2000 2000 100\ndump_thermo 100\n'
            'dump_xyz 1000 diffusion.xyz precision double mass group_labels unwrapped_position\n'
            'dump_restart 3000\nrun 3000\n')
    return text


def performance(size, profile, warmup, steps):
    tile = TILES[size]
    text = 'replicate {} {} {}\npotential potential.txt x\ntime_step .5\n'.format(*tile)
    text += 'ensemble nve\nrun {}\nensemble nve\n'.format(warmup)
    if profile == 'output':
        text += 'dump_thermo {}\ndump_xyz {} trajectory.xyz\n'.format(max(1, steps // 5), steps)
    text += 'run {}\n'.format(steps)
    # An untimed final run checks finite thermodynamics and flushes GPU work.
    text += 'ensemble nve\ndump_thermo 1\nrun 1\n'
    return text
