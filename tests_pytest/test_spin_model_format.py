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
    main.write_text('''#include "utilities/nep_io/nep_spin_model_file.cuh"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char** argv) {
  std::ifstream input(argv[1]);
  NEP_Spin_Model_File model;
  std::string error;
  if (!model.read(input, error)) { std::cerr << error; return 1; }
  if (!NEP_Spin_Model_File::is_spin_model(argv[1])) {
    std::cerr << "Spin model was not recognized"; return 2;
  }
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
  dump("spin_type", model.spin_type);
  std::cout << "dimensions " << model.dim_struct << ' ' << model.dim_spin << '\\n';
  std::cout << "zbl " << model.enable_zbl << ' ' << model.zbl_rc_inner << ' '
            << model.zbl_rc_outer << ' ' << model.typewise_cutoff_zbl_factor << '\\n';
}
''')
    binary = root / 'codec'
    subprocess.run([compiler, '-std=c++14', '-pedantic-errors', '-I' + str(ROOT / 'src'), '-x', 'c++',
        str(main), str(ROOT / 'src/utilities/nep_io/nep_spin_model_file.cu'),
        str(ROOT / 'src/utilities/nep_io/nep_file.cu'), '-o', str(binary)], check=True)
    return binary


def parse(codec, path, model):
    path.write_text(model)
    result = subprocess.check_output([codec, path], text=True)
    return {line.split()[0]: list(map(float, line.split()[1:])) for line in result.splitlines()}


def float32(value):
    return struct.unpack('f', struct.pack('f', float(value)))[0]


@pytest.mark.parametrize('name,dimension,count', [
    ('c3l2s2',64,7270), ('c2l4s4',70,8151), ('c1l0s1',32,4081)])
def test_spin_model(codec, tmp_path, name, dimension, count):
    model = (Path(__file__).parent/'fixtures/nep_spin'/(name+'.nep')).read_text()
    parsed = parse(codec,tmp_path/'model.nep',model)
    assert parsed['dimensions'] == [30,dimension-30]
    assert len(parsed['parameters']) == count
    assert len(parsed['q_scaler']) == dimension
    assert parsed['spin_type'] == [1,0,0]


@pytest.mark.parametrize('old,new', [
    ('spin_version 1','spin_version 2'),
    ('spin_version 1','spin_version 1.0'),
    ('spin_soc 1','spin_soc 0'),
    ('spin_order 2','spin_order 5'),
    ('spin_l_max 2','spin_l_max 5'),
    ('spin_basis_size 8','spin_basis_size 7'),
    ('spin_type Fe','spin_dof_type Fe'),
])
def test_spin_reject_invalid(codec,tmp_path,old,new):
    model = (Path(__file__).parent/'fixtures/nep_spin/c3l2s2.nep').read_text()
    path = tmp_path/'invalid.nep';path.write_text(model.replace(old,new))
    result=subprocess.run([codec,path],capture_output=True,text=True)
    assert result.returncode == 1 and result.stderr


@pytest.mark.parametrize('channels,rank,order,dimension,projection', [
    (3,2,2,61,87), (1,3,2,44,36), (1,4,4,58,98), (2,1,4,42,54)])
def test_spin_nosoc_layout(codec,tmp_path,channels,rank,order,dimension,projection):
    template=(Path(__file__).parent/'fixtures/nep_spin/c3l2s2.nep').read_text().split('@ parameters')[0]
    for old,new in [('spin_soc 1','spin_soc 0'),('spin_compress 3',f'spin_compress {channels}'),
                    ('spin_l_max 2',f'spin_l_max {rank}'),('spin_order 2',f'spin_order {order}')]:
        template=template.replace(old,new)
    count=3*32*(dimension+2)+1+810+projection
    model=template+f'@ parameters {count}\n'+'0\n'*count+f'@ q_scaler {dimension}\n'+'1\n'*dimension
    parsed=parse(codec,tmp_path/'nosoc.nep',model)
    assert parsed['dimensions']==[30,dimension-30]
    assert len(parsed['parameters'])==count


@pytest.mark.parametrize('name', ['c3l2s2', 'c2l4s4', 'c1l0s1', 'c3l2s2_fege'])
def test_parameter_and_scaler_order(codec, tmp_path, name):
    model = (Path(__file__).parent/'fixtures/nep_spin'/(name+'.nep')).read_text()
    parsed = parse(codec, tmp_path/'model.nep', model)
    lines = model.splitlines()
    for key in ('parameters','q_scaler'):
        a = next(i for i,l in enumerate(lines) if l.startswith('@ '+key+' '))
        count = int(lines[a].split()[2])
        assert parsed[key] == [float32(v) for v in lines[a+1:a+1+count]]


@pytest.mark.parametrize('old,new', [
    ('spin_version 1', 'spin_version 1\nspin_version 1'),
    ('spin_version 1\n', ''),
    ('spin_type Fe', 'spin_type Fe Fe'),
    ('spin_type Fe', 'spin_type Unknown'),
    ('spin_cutoff 6', 'spin_cutoff 0'),
    ('spin_cutoff 6', 'spin_cutoff 2.999'),
    ('spin_cutoff 6', 'spin_cutoff 6 2.999 6'),
    ('spin_cutoff 6', 'spin_cutoff 7'),
    ('ann 32 0', 'ann 32 1'),
    ('spin_scaler 1','spin_scaler 0'),
    ('energy_reference','unknown_reference'),
    ('@ q_scaler 64', '@ q_scaler 63'),
    ('format_version 1', 'format_version 2'),
    ('model nep_spin', 'model nep_spin\nzbl 0 0'),
])
def test_model_validation(codec,tmp_path,old,new):
    model = (Path(__file__).parent/'fixtures/nep_spin/c3l2s2.nep').read_text()
    assert old in model
    path=tmp_path/'invalid.nep';path.write_text(model.replace(old,new))
    result=subprocess.run([codec,path],capture_output=True,text=True)
    assert result.returncode == 1 and result.stderr


@pytest.mark.parametrize("angular,minimum", [
    ("2 0 0 0 1 0 0", 3), ("2 0 0 0 0 1 0", 3), ("3 0 0 0 0 0 1", 4),
])
def test_structural_rank_for_mixed_invariants(codec, tmp_path, angular, minimum):
    model = (Path(__file__).parent / 'fixtures/nep_spin/c3l2s2.nep').read_text()
    path = tmp_path / "rank.nep"
    path.write_text(model.replace("l_max 4 2 0\n", f"l_max {angular}\n"))
    result = subprocess.run([codec, path], capture_output=True, text=True)
    assert result.returncode == 1
    assert f"structural l_max >= {minimum}" in result.stderr


def test_defaults_and_zbl(codec,tmp_path):
    model=(Path(__file__).parent/'fixtures/nep_spin/c3l2s2.nep').read_text()
    model=model.replace('spin_type Fe\n','').replace('model nep_spin\n','model nep_spin\nzbl 1 2 0.75\n')
    parsed=parse(codec,tmp_path/'model.nep',model)
    assert parsed['spin_type']==[1,1,1]
    assert parsed['zbl']==[1,1,2,.75]


def test_spin_cutoff_lower_bound(codec, tmp_path):
    model = (Path(__file__).parent/'fixtures/nep_spin/c3l2s2.nep').read_text()
    for cutoff, expected in [('3', [3, 3, 3]), ('3 4 6', [3, 4, 6])]:
        parsed = parse(codec, tmp_path/'cutoff.nep', model.replace('spin_cutoff 6', 'spin_cutoff '+cutoff))
        assert parsed['spin_cutoff'] == expected
