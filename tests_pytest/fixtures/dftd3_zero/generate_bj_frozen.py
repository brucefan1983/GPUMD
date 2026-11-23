"""Freeze the output of the unmodified upstream binary for the BJ-unchanged tests of test_dftd3_zero.py.

Writes bj_frozen.json next to this file (tiny24.xyz is stored). Run it ONCE, with a gpumd binary built from
the unmodified upstream source (before any zero-damping change; the commit is recorded):

    python generate_bj_frozen.py <path/to/baseline/gpumd> <upstream-commit> <dftd3.cu-sha256>

For each case it stores the dump.xyz lines 2.. (header with energy and virial, then species,
positions and forces, `precision double`) of a static run with `dftd3 pbe rc rcn`, and
the 'Add DFT-D3:' block of the standard output (the first of 3 runs; on a structure where the baseline
itself fluctuates the repeats differ in the last digits, see below). The potential is the all-zero NEP of make_zero_nep.py (zero energy, force and
virial exactly), so the output is the D3 contribution alone: NEP89 itself is not bit-reproducible
run to run on a GPU. Never regenerate it with a changed binary.

Run-to-run spread of the baseline (`bj_spread.json`). The D3 energy, virial and forces are accumulated with
double atomicAdd in a thread-order that is not fixed on the GPU, so for several structures the UNCHANGED baseline
binary itself returns slightly different last digits from run to run (tiny24 and benzene_dimer: one single
output over 40 runs; water27, graphite, water27_tri: fluctuating). The BJ test is therefore bitwise only where the
baseline is deterministic; elsewhere the deviation of the new binary from the frozen output must not exceed
SAFETY x the spread (max - min over N repeats of the unchanged baseline, frozen run included), taken per group
(energy, largest virial component, largest force component):

    python generate_bj_frozen.py spread <path/to/baseline/gpumd> [N=20]

writes bj_spread.json (adds data only; bj_frozen.json is not touched).

The `stress="..."` field of the dump header is removed: with a zero potential it varies from run
to run (also on the same binary) while energy, virial and forces do not.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ase.io import read, write

HERE = Path(__file__).resolve().parent
BJ_DIR = HERE      # structures: *.xyz next to this script
NEP = HERE / 'zero_nep_H_C_O_Cu.txt'     # all-zero NEP (make_zero_nep.py): the output is D3 alone


def tiny24():
    atoms = read(BJ_DIR / 'water27.xyz')[:24]
    atoms.set_cell([40.0, 40.0, 40.0])
    atoms.set_pbc(True)
    atoms.center()
    write(HERE / 'tiny24.xyz', atoms, format='extxyz')


CASES = {  # name: (structure, (rc, rc_cn), note)
    'water27': ('water27.xyz', (12, 6), 'small-box, cubic'),
    'graphite': ('graphite.xyz', (12, 6), 'small-box, hexagonal'),
    'water27_tri': ('water27_tri.xyz', (12, 6), 'small-box, triclinic'),
    'benzene_dimer': ('benzene_dimer.xyz', (12, 6), 'large-box path, non-periodic'),
    'tiny24': ('tiny24.xyz', (12, 6), 'large-box path, 24 atoms in a 40 A cell'),
}


def run(exe, structure, d3_line):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        read(structure).write(tmp / 'model.xyz', format='extxyz')
        lines = [f'potential {NEP}'] + ([d3_line] if d3_line else []) + \
            ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
        (tmp / 'run.in').write_text('\n'.join(lines) + '\n')
        r = subprocess.run([exe], cwd=tmp, capture_output=True, text=True, check=True)
        out = r.stdout.splitlines()
        block = []
        if d3_line:
            i = out.index('    Add DFT-D3:')
            while i < len(out) and out[i].strip():
                block.append(out[i])
                i += 1
        lines = (tmp / 'dump.xyz').read_text().splitlines()[1:]
        lines[0] = re.sub(r' stress="[^"]*"', '', lines[0])   # varies run to run (see docstring)
        return lines, block


SAFETY = 3.0     # tolerance in the test = SAFETY x measured spread


def quantities(lines):
    """(energy, virial[9], forces[3N]) of the dump lines (header, then species/positions/forces)."""
    energy = float(re.search(r'energy=([-0-9.eE+]+)', lines[0]).group(1))
    virial = [float(x) for x in re.search(r'virial="([^"]+)"', lines[0]).group(1).split()]
    forces = [float(x) for line in lines[1:] for x in line.split()[4:7]]
    return energy, virial, forces


def spread_main(exe, n):
    frozen = json.load(open(HERE / 'bj_frozen.json'))
    out = {'_provenance': {'binary': frozen['_provenance']['source'], 'repeats': n, 'safety_factor': SAFETY,
                           'gpu_and_driver': frozen['_provenance']['gpu_and_driver'],
                           'note': 'spread = max - min over the repeats and the frozen run; per group'}}
    for name, (fname, cut, note) in CASES.items():
        structure = HERE / fname
        samples = [quantities(frozen[name]['D3'])]
        for _ in range(n):
            lines, _ = run(exe, structure, f'dftd3 pbe {cut[0]} {cut[1]}')
            assert [l.split()[0] for l in lines[1:]] == [l.split()[0] for l in frozen[name]['D3'][1:]]
            samples.append(quantities(lines))

        def rng(values):
            return max(values) - min(values)

        e = rng([s[0] for s in samples])
        v = max(rng([s[1][k] for s in samples]) for k in range(9))
        f = max(rng([s[2][k] for s in samples]) for k in range(len(samples[0][2])))
        distinct = len({tuple(map(float.hex, [s[0]] + s[1] + s[2])) for s in samples})
        out[name] = {'energy': e, 'virial': v, 'force': f, 'distinct_outputs': distinct}
        print(name, out[name])
    json.dump(out, open(HERE / 'bj_spread.json', 'w'), indent=1)


if __name__ == '__main__' and sys.argv[1] == 'spread':
    spread_main(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
elif __name__ == '__main__':
    exe, commit, sha = sys.argv[1], sys.argv[2], sys.argv[3]
    tiny24()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    out = {'_provenance': {'binary': 'unmodified upstream gpumd (BJ damping only)',
                           'source': commit, 'dftd3_cu_sha256': sha, 'gpu_and_driver': gpu,
                           'note': 'bitwise reference; see generate_bj_frozen.py'}}
    for name, (fname, cut, note) in CASES.items():
        structure = HERE / fname if (HERE / fname).exists() else BJ_DIR / fname
        entry = {'structure': fname, 'cutoffs': list(cut), 'note': note}
        runs = [run(exe, structure, f'dftd3 pbe {cut[0]} {cut[1]}') for _ in range(3)]
        entry['D3'], entry['stdout_block'] = runs[0]
        out[name] = entry
        print(name, 'ok')
    json.dump(out, open(HERE / 'bj_frozen.json', 'w'), indent=0)
