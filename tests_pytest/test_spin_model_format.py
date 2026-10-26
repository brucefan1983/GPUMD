"""Exercise the production Spin model parser without CUDA or text normalization."""
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from spin_fixtures import load_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def codec(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    root = tmp_path_factory.mktemp('spin-codec')
    main = root / 'main.cpp'
    main.write_text('''#include "utilities/nep_spin_model_file.cuh"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char** argv) {
  std::ifstream input(argv[1]);
  NEP_Spin_Model_File model;
  std::string error;
  if (!model.read(input, error)) { std::cerr << error; return 1; }
  std::cout << std::setprecision(17);
  auto dump = [](const char* name, const auto& values) {
    std::cout << name;
    for (auto value : values) std::cout << ' ' << value;
    std::cout << '\\n';
  };
  dump("energy_reference", model.energy_reference);
  dump("parameters", model.parameters);
  dump("q_scaler", model.q_scaler);
  dump("spin_cutoff", model.spin_cutoff);
  dump("spin_dof_type", model.spin_dof_type);
  dump("spin_env_type", model.spin_env_type);
  std::cout << "dimensions " << model.dim_struct << ' ' << model.dim_spin << '\\n';
  std::cout << "zbl " << model.enable_zbl << ' ' << model.zbl_rc_inner << ' '
            << model.zbl_rc_outer << ' ' << model.typewise_cutoff_zbl_factor << '\\n';
}
''')
    binary = root / 'codec'
    subprocess.run([compiler, '-std=c++14', '-pedantic-errors', '-I' + str(ROOT / 'src'), '-x', 'c++',
        str(main), str(ROOT / 'src/utilities/nep_spin_model_file.cu'),
        str(ROOT / 'src/utilities/nep_file.cu'), '-o', str(binary)], check=True)
    return binary


def parse(codec, path, model):
    path.write_text(model)
    result = subprocess.check_output([codec, path], text=True)
    return {line.split()[0]: list(map(float, line.split()[1:])) for line in result.splitlines()}


def float32(value):
    return struct.unpack('f', struct.pack('f', float(value)))[0]


@pytest.mark.parametrize('fixture', ['o3c2_uniform', 'o3c2_typewise', 'o3c3_uniform'])
def test_parameter_and_scaler_order(codec, tmp_path, fixture):
    model, _ = load_fixture(fixture)
    parsed = parse(codec, tmp_path / 'model.nep', model)
    lines = model.splitlines()
    for name in ('parameters', 'q_scaler'):
        start = next(i for i, line in enumerate(lines) if line.startswith('@ ' + name + ' '))
        count = int(lines[start].split()[2])
        assert parsed[name] == [float32(value) for value in lines[start + 1:start + count + 1]]
    reference = next(line.split()[1:] for line in lines if line.startswith('energy_reference '))
    assert parsed['energy_reference'] == list(map(float, reference))
    assert parsed['dimensions'] == [30, 91 if fixture == 'o3c3_uniform' else 55]
    assert parsed['spin_cutoff'] == ([4, 6] if fixture == 'o3c2_typewise' else [6, 6])
    assert parsed['spin_dof_type'] == [1, 0]
    assert parsed['spin_env_type'] == [1, 1]


def test_defaults_and_universal_zbl(codec, tmp_path):
    model, _ = load_fixture('o3c2_uniform')
    model = model.replace('spin_dof_type Fe\n', '').replace('spin_env_type Fe Ge\n', '')
    model = model.replace('model nep_spin\n', 'model nep_spin\nzbl 1 2 0.75\n')
    parsed = parse(codec, tmp_path / 'model.nep', model)
    assert parsed['spin_dof_type'] == parsed['spin_env_type'] == [1, 1]
    assert parsed['zbl'] == [1, 1, 2, 0.75]


@pytest.mark.parametrize('edit', [
    lambda s: s.replace('model nep_spin\n', 'model nep_spin3\n'),
    lambda s: s.replace('spin_version 1', 'spin_mode 3'),
    lambda s: s.replace('spin_version 1', 'spin_version 2'),
    lambda s: s.replace('spin_version 1', 'spin_version 1.0'),
    lambda s: s.replace('spin_version 1\n', ''),
    lambda s: s.replace('spin_version 1', 'spin_version 1\nspin_version 1'),
    lambda s: s.replace('energy_reference', 'unknown_reference'),
    lambda s: s.replace('format_version 1', 'format_version 2'),
    lambda s: s + '@ unexpected 1\n1\n',
    lambda s: s.replace('spin_projection_size 16', 'spin_projection_size 15'),
    lambda s: s.replace('spin_scaler 1', 'spin_scaler 0'),
    lambda s: s.replace('spin_compress 2', 'spin_compress 9'),
    lambda s: s.replace('spin_cutoff 6.0', 'spin_cutoff 0'),
    lambda s: s.replace('spin_dof_type Fe', 'spin_dof_type Fe Fe'),
    lambda s: s.replace('spin_env_type Fe Ge', 'spin_env_type Ge'),
    lambda s: s.replace('ann 30 0', 'ann 30 1'),
    lambda s: s.replace('@ parameters ', '@ q_scaler ', 1),
    lambda s: s[:s.index('@ q_scaler')] + '@ q_scaler 1\n1\n',
    lambda s: s[:s.index('@ q_scaler')] + '@ q_scaler 85\n' + '0\n' * 85,
    lambda s: s[:s.index('@ q_scaler')] + '@ q_scaler 85\n' + '1e-100\n' * 85,
    lambda s: 'nep4_spin3 2 Fe Ge\nspin_mode 3 9\n',
])
def test_reject_invalid_models(codec, tmp_path, edit):
    model, _ = load_fixture('o3c2_uniform')
    path = tmp_path / 'invalid.nep'
    path.write_text(edit(model))
    result = subprocess.run([codec, path], capture_output=True, text=True)
    assert result.returncode == 1 and result.stderr


def test_reject_flexible_zbl(codec, tmp_path):
    model, _ = load_fixture('o3c2_uniform')
    path = tmp_path / 'flexible-zbl.nep'
    path.write_text(model.replace('model nep_spin\n', 'model nep_spin\nzbl 0 0\n'))
    result = subprocess.run([codec, path], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'NEP-spin does not support flexible ZBL' in result.stderr


@pytest.mark.parametrize('cutoff', ['6.0', '7.0', '5.0 7.0'])
@pytest.mark.parametrize('spin_first', [False, True])
def test_spin_cutoff_bound(codec, tmp_path, cutoff, spin_first):
    model, _ = load_fixture('o3c2_uniform')
    model = model.replace('spin_cutoff 6.0', 'spin_cutoff ' + cutoff)
    if spin_first:
        line = next(line for line in model.splitlines() if line.startswith('cutoff '))
        model = model.replace(line + '\n', '').replace(
            'spin_cutoff ' + cutoff + '\n', 'spin_cutoff ' + cutoff + '\n' + line + '\n')
    path = tmp_path / 'model.nep'
    path.write_text(model)
    result = subprocess.run([codec, path], capture_output=True, text=True)
    if cutoff == '6.0':
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 1
        assert 'spin_cutoff must not exceed the NEP radial cutoff' in result.stderr
