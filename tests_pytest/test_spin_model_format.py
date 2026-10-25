"""Exercise the production Spin3 block codec without CUDA."""
import shutil
import subprocess
from pathlib import Path

import pytest
from spin_model_format import block_model
from validate_nep_spin3_runtime import load_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'fixtures' / 'nep_spin3'


@pytest.fixture(scope='module')
def codec(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    root = tmp_path_factory.mktemp('spin-codec')
    main = root / 'main.cpp'
    main.write_text('''#include "utilities/nep_spin_model_file.cuh"
#include <iostream>
int main(int argc, char** argv) {
  try {
    if (!is_nep_spin_model_file(argv[1])) return 2;
    std::cout << read_nep_spin_model_input(argv[1]);
  }
  catch (const std::exception& error) { std::cerr << error.what(); return 1; }
}
''')
    binary = root / 'codec'
    subprocess.run([compiler, '-std=c++14', '-I' + str(ROOT / 'src'), '-x', 'c++',
        str(main), str(ROOT / 'src/utilities/nep_spin_model_file.cu'),
        str(ROOT / 'src/utilities/nep_file.cu'), '-o', str(binary)], check=True)
    return binary


@pytest.mark.parametrize('fixture', ['o3c2_uniform', 'o3c2_typewise', 'o3c3_uniform'])
@pytest.mark.parametrize('legacy_name', [False, True], ids=['versioned', 'old-block'])
def test_legacy_and_blocks(codec, tmp_path, fixture, legacy_name):
    legacy, _ = load_fixture(fixture)
    old = tmp_path / "legacy.nep"
    old.write_text(legacy)
    assert subprocess.check_output([codec, old], text=True) == legacy
    new = tmp_path / 'new.nep'
    new.write_text(block_model(legacy, legacy_name=legacy_name))
    normalized = subprocess.check_output([codec, new], text=True)
    old_lines, new_lines = legacy.splitlines(), normalized.splitlines()
    old_start = next(i + 1 for i, x in enumerate(old_lines) if x.startswith('ANN '))
    new_start = next(i + 1 for i, x in enumerate(new_lines) if x.startswith('ANN '))
    assert [float(x) for x in new_lines[new_start:]] == pytest.approx(
        [float(x) for x in old_lines[old_start:]], rel=6e-8, abs=1e-15)
    old_ref = next(x.split()[1:] for x in old_lines if x.startswith('spin_baseline '))
    new_ref = next(x.split()[1:] for x in new_lines if x.startswith('spin_baseline '))
    assert new_ref == old_ref


@pytest.mark.parametrize('edit', [
    lambda s: s.replace('energy_reference', 'unknown_reference'),
    lambda s: s.replace('spin_version 1', 'spin_version 2'),
    lambda s: s.replace('spin_version 1', 'spin_version 0'),
    lambda s: s.replace('spin_version 1', 'spin_version -1'),
    lambda s: s.replace('spin_version 1', 'spin_version 1.0'),
    lambda s: s.replace('spin_version 1', 'spin_version 1 2'),
    lambda s: s.replace('spin_version 1\n', ''),
    lambda s: s.replace('spin_version 1', 'spin_version 1\nspin_version 1'),
    lambda s: s + 'spin_mode 3\n',
    lambda s: s.replace('model nep_spin\n', 'model nep_spin3\n'),
    lambda s: s.replace('format_version 1', 'format_version 2'),
    lambda s: s + '@ unexpected 1\n1\n',
    lambda s: s.replace('spin_scaler 1', 'spin_scaler 1\nspin_scaler 1'),
])
def test_reject_invalid_blocks(codec, tmp_path, edit):
    model = block_model(next(FIXTURES.glob('*.nep')).read_text())
    path = tmp_path / 'invalid.nep'
    path.write_text(edit(model))
    result = subprocess.run([codec, path], capture_output=True, text=True)
    assert result.returncode != 0


@pytest.mark.parametrize('edit', [
    lambda s: s.replace('spin_mode 3', 'spin_mode 2'),
    lambda s: s.replace('spin_mode 3\n', ''),
    lambda s: s + 'spin_version 1\n',
])
def test_reject_invalid_old_blocks(codec, tmp_path, edit):
    legacy, _ = load_fixture('o3c2_uniform')
    path = tmp_path / 'invalid-old.nep'
    path.write_text(edit(block_model(legacy, legacy_name=True)))
    result = subprocess.run([codec, path], capture_output=True, text=True)
    assert result.returncode != 0
