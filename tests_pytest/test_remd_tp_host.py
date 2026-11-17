"""CPU regressions using production C++ definitions, without a CUDA dependency.

The harness substitutes GPU types and exchange velocity/pressure operations.
It tests the real host algorithms, not GPU integration or kernel execution.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
REPLICA = SRC / 'integrate/replica'


def definition(source, signature):
    """Extract a brace-balanced production definition; fail if it disappears."""
    start = source.index(signature)
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        if source[end] == '{':
            depth += 1
        elif source[end] == '}':
            depth -= 1
        end += 1
    return source[start:end]


def build_harness(directory):
    common = (REPLICA / 'replica_common.cu').read_text()
    header = (REPLICA / 'replica_scr.cuh').read_text()
    tp = (REPLICA / 'replica_remd_tp.cu').read_text()
    box = (SRC / 'model/box.cu').read_text()
    velocity = (REPLICA / 'replica_scr.cu').read_text()
    svr = (SRC / 'integrate/svr_utilities.cuh').read_text()
    declarations = [
        definition((SRC / 'model/box.cuh').read_text(), 'class Box') + ';',
        definition(header, 'struct SCR_Barostat') + ';',
        definition(tp, 'struct Barostat_Input') + ';',
        definition(tp, 'enum Direction') + ';',
        definition((REPLICA / 'replica_common.cuh').read_text(), 'template <typename T>\nbool parse_scalar('),
        'void generate_canonical_velocities(double, const std::vector<double>&, std::vector<double>&, unsigned int);',
        re.search(r'constexpr int restart_format_version = \d+;', tp).group(),
        (SRC / 'integrate/svr_state.cuh').read_text(),
        definition(svr, 'struct SVR_Noise') + ';',
    ]
    definitions = [definition(common, name) for name in (
        'bool values_match(', 'std::string trim_left(', 'bool extract_quoted_value(',
        'bool extract_token_value(')]
    definitions += [definition(velocity, name) for name in ('SCR_Barostat::SCR_Barostat(',
        'double SCR_Barostat::update_box(', 'std::string SCR_Barostat::export_state(',
        'bool SCR_Barostat::import_state(')]
    definitions += ['namespace replica {\n' + definition(common, 'void validate_xyz_pbc(') + '\n}']
    definitions += [definition(box, name) for name in ('void Box::get_inverse(', 'double Box::get_volume(')]
    definitions += [definition(velocity, name) for name in ('static void zero_linear_momentum(', 'void generate_canonical_velocities(')]
    definitions += [definition(svr, name) for name in (
        'static double gasdev(', 'static double gamdev(',
        'static double\nresamplekin_sumnoises(', 'static SVR_Noise\ndraw_resamplekin_noise(')]
    definitions += [definition((REPLICA / 'replica_thermo_kernels.cuh').read_text(),
                               'static __device__ double svr_velocity_factor(')]
    definitions += [definition((REPLICA / 'ensemble_bdp_replica.cu').read_text(),
                               '__global__ void gpu_resamplekin_and_scale(')]
    definitions += [definition(tp, name) for name in (
        'void validate_barostat_box(', 'void REMD_TP_Driver::read_restart_metadata(',
        'void REMD_TP_Driver::read_restart_xyz(', 'void REMD_TP_Driver::set_condition(',
        'void REMD_TP_Driver::build_phases(', 'void REMD_TP_Driver::attempt_pair(',
        'void REMD_TP_Driver::attempt_exchanges(')]
    source = Path(__file__).with_name('remd_tp_host_harness.cpp').read_text()
    source = source.replace('// @production declarations', '\n'.join(declarations))
    source = source.replace('// @production definitions', '\n'.join(definitions))
    path = directory / 'host.cpp'
    path.write_text(source)
    binary = directory / 'host'
    command = shlex.split(os.environ.get('CXX', 'c++'))
    command += ['-std=c++14', '-O2', '-Wall', '-Wextra', '-Wno-unused-parameter']
    command += shlex.split(os.environ.get('REPLICA_HOST_CXXFLAGS', ''))
    command += ['-I', str(SRC), str(path), '-o', str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return binary


class RemdTPHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='remd-tp-host-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        try:
            cls.binary = build_harness(cls.directory)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(error.stderr) from error
        cls.rng = subprocess.check_output([str(cls.binary), 'rng'], text=True)
        cls.version = re.search(r'constexpr int restart_format_version = (\d+);',
                               (REPLICA / 'replica_remd_tp.cu').read_text()).group(1)

    def setUp(self):
        self.prefix = self.directory / 'checkpoint'
        self.meta = Path(str(self.prefix) + '_restart.meta')
        self.xyz = Path(str(self.prefix) + '_replica_0_restart.xyz')
        self.lines = [
            f'format_version {self.version}', 'replicas 2', 'completed_steps 100',
            'exchange_count 10', 'exchange_interval 10', 'time_step 1', 'potential_hash 123',
            'barostat 100 2 1000 10 z', 'temperatures 300 400', 'pressure_table 0 0',
            'replica_to_label 0 1', 'pair_statistics 0 10 10 10 0 0 0',
            'pair_statistics 1 0 0 0 0 0 0', f'exchange_rng {self.rng}',
            f'bdp_rng 0 1 0 0 {self.rng}', f'bdp_rng 1 1 0 0 {self.rng}',
            f'scr_state 0 1 100 1 300 0 0 0 0 {self.rng} 0 1 0',
            f'scr_state 1 1 100 1 300 0 0 0 0 {self.rng} 0 1 0',
        ]
        self.header = ('replica_mode=remd_tp replica=0 condition_label=0 step=100 '
                       'target_temperature=300 target_pressure=0 potential_energy=0 '
                       'pbc="T T T" Lattice="30 0 0 0 30 0 0 0 31" '
                       'Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3')
        self.atoms = ['C 1 2 3 12 .1 .2 .3', 'C 2 3 4 12 -.1 -.2 -.3']

    def run_case(self, command, error=None):
        result = subprocess.run([str(self.binary), command, str(self.prefix)]
                                if command in ('metadata', 'xyz') else [str(self.binary), command],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 1 if error else 0, result.stdout + result.stderr)
        if error:
            self.assertIn(error, result.stderr)
        return result.stdout

    def write_meta(self, lines=None):
        self.meta.write_text('\n'.join(self.lines if lines is None else lines) + '\n')

    def write_xyz(self, header=None, atoms=None):
        self.xyz.write_text('2\n' + (self.header if header is None else header) + '\n' +
                            '\n'.join(self.atoms if atoms is None else atoms) + '\n')

    def test_valid_checkpoint(self):
        self.write_meta()
        self.run_case('metadata')
        self.write_meta(list(reversed(self.lines)))
        self.run_case('metadata')
        self.write_xyz()  # Coupled z length legitimately differs from model.xyz.
        self.run_case('xyz')

    def test_every_metadata_record_required(self):
        for i, line in enumerate(self.lines):
            with self.subTest(key=line.split()[:2]):
                self.write_meta(self.lines[:i] + self.lines[i + 1:])
                self.run_case('metadata', 'incomplete')

    def test_every_metadata_record_unique(self):
        for line in self.lines:
            with self.subTest(key=line.split()[:2]):
                self.write_meta(self.lines + [line])
                self.run_case('metadata', 'restart')

    def test_metadata_mismatch_and_corruption(self):
        changes = {
            'format_version': '2', 'replicas': '3', 'completed_steps': '-1',
            'exchange_count': '9', 'exchange_interval': '11', 'time_step': '2',
            'potential_hash': '456', 'barostat': '100 2 1000 11 z',
            'temperatures': '300 401', 'pressure_table': '0 .1',
            'replica_to_label': '0 0', 'exchange_rng': self.rng + ' trailing',
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                self.write_meta([f'{key} {value}' if x.startswith(key + ' ') else x
                                 for x in self.lines])
                self.run_case('metadata', 'REMD-TP')
        for value in ('0 -1 0 0 0 0 0', '0 1 2 1 0 0 0', '0 1 0 2 0 0 0',
                      '0 11 0 0 0 0 0', '0 1 0 nan 0 0 0'):
            with self.subTest(statistics=value):
                self.write_meta([f'pair_statistics {value}' if x.startswith('pair_statistics 0 ')
                                 else x for x in self.lines])
                self.run_case('metadata', 'pair_statistics')

    def test_metadata_truncation_and_trailing_data(self):
        for key in ('temperatures', 'pressure_table', 'barostat', 'replica_to_label'):
            for suffix in ('', ' extra'):
                with self.subTest(key=key, suffix=suffix):
                    lines = [((x.rsplit(' ', 1)[0] if not suffix else x + suffix)
                              if x.startswith(key + ' ') else x) for x in self.lines]
                    self.write_meta(lines)
                    self.run_case('metadata', 'REMD-TP')

    def test_xyz_pbc(self):
        for pbc in ('F F T', 'T T', 'T T T F', 'X T T'):
            with self.subTest(pbc=pbc):
                self.write_xyz(self.header.replace('T T T', pbc))
                self.run_case('xyz', 'pbc')
        self.write_xyz(self.header.replace('pbc="T T T"', ''))
        self.run_case('xyz', 'missing pbc')

    def test_xyz_lattice(self):
        original = '30 0 0 0 30 0 0 0 31'
        for lattice in ('0 0 0 0 30 0 0 0 31', '-30 0 0 0 30 0 0 0 31',
                        '30 0 0 5 30 0 0 0 31', '30 0 0 0 30 0 0 0',
                        original + ' 1', 'nan 0 0 0 30 0 0 0 31',
                        '31 0 0 0 30 0 0 0 31', '1e200 0 0 0 1e200 0 0 0 1e200'):
            with self.subTest(lattice=lattice):
                self.write_xyz(self.header.replace(original, lattice))
                self.run_case('xyz', 'REMD-TP')

    def test_xyz_header_consistency(self):
        for key, wrong in (('replica', '1'), ('condition_label', '1'), ('step', '99'),
                           ('target_temperature', '301'), ('target_pressure', '.1'),
                           ('replica_mode', 'remd'), ('potential_energy', 'nan'),
                           ('Properties', 'species:S:1:vel:R:3:mass:R:1:pos:R:3')):
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    header = re.sub(r'\b' + key + r'=\S+', '' if missing else f'{key}={wrong}',
                                    self.header)
                    self.write_xyz(header)
                    self.run_case('xyz', 'metadata')

    def test_xyz_atom_data(self):
        for atoms in (self.atoms[:1], self.atoms + [self.atoms[0]],
                      ['C 1 2 3 12 .1 .2', self.atoms[1]],
                      ['C 1 2 3 12 .1 .2 .3 extra', self.atoms[1]],
                      ['C nan 2 3 12 .1 .2 .3', self.atoms[1]],
                      ['C 1 2 3 12 1e308 .2 .3', self.atoms[1]],
                      ['Si 1 2 3 12 .1 .2 .3', self.atoms[1]],
                      ['C 1 2 3 13 .1 .2 .3', self.atoms[1]]):
            with self.subTest(atoms=atoms):
                self.write_xyz(atoms=atoms)
                self.run_case('xyz', 'REMD-TP')

    def test_scr_conditional_moments(self):
        print(self.run_case('scr_moments').strip())

    def test_scr_rng_and_cached_gaussian_roundtrip(self):
        self.run_case('scr_roundtrip')

    def test_scr_all_coupled_directions(self):
        self.run_case('scr_directions')

    def test_canonical_velocity_distribution(self):
        print(self.run_case('canonical').strip())

    def test_bdp_kinetic_distribution(self):
        print(self.run_case('bdp').strip())

    def test_coupled_ideal_gas_distribution(self):
        print(self.run_case('ideal_gas').strip())

    def test_temperature_exchange_preserves_pending_updates(self):
        self.run_case('exchange')

    def test_two_dimensional_exchange_phases(self):
        self.run_case('phases')


if __name__ == '__main__':
    unittest.main(verbosity=2)
