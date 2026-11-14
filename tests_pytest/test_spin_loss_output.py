"""Host-only check of the production loss writer, including optional columns."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def test_spin_loss_output(tmp_path):
    compiler = shutil.which('c++')
    assert compiler, 'a C++ compiler is required'
    source = (Path(__file__).parents[1] / 'src/main_nep/nep_spin/nep_spin.cu').read_text()
    writer = source.split('for (FILE* file : {stdout, training->fid_loss_out}) {', 1)[1]
    writer = 'for (FILE* file : {stdout, training->fid_loss_out}) {' + writer.split(
        '\n  if (training->has_test_set)', 1)[0]
    program = r'''
#include <cstdio>
#include <initializer_list>
int main(int, char** argv) {
  struct { bool split_mforce_loss, has_ediff_combinations; } para = {argv[2][0] == '1', argv[3][0] == '1'};
  struct { int generation; float total, l1, l2; } report = {999, .79744f, .00519f, .00811f};
  struct Error { float energy, force, virial, ediff; };
  Error train = {.01941f, .47204f, .15199f, .00000017f};
  Error test = {.01729f, .49314f, .12671f, .00000029f};
  float mt = .11534f, tt = .08302f, mv = .12807f, tv = .08047f;
  float mp_train = .197496742f, mt_train = .0212451406f;
  float mp_test = .219915941f, mt_test = .020569738f;
  struct { FILE* fid_loss_out; } files = {fopen(argv[1], "w")};
  auto* training = &files;
  if (!training->fid_loss_out) return 1;
''' + writer + r'''
  return fclose(training->fid_loss_out);
}
'''
    cpp = tmp_path / 'writer.cpp'
    cpp.write_text(program)
    binary = tmp_path / 'writer'
    subprocess.run([compiler, '-std=c++14', '-Wall', '-Wextra', '-Werror',
                    str(cpp), '-o', str(binary)], check=True)
    expected = [.79744, .00519, .00811, .01941, .47204, .15199, .11534,
                .08302, .01729, .49314, .12671, .12807, .08047]
    split = [.197496742, .0212451406, .219915941, .020569738]
    for magnetic_split in (False, True):
        for ediff in (False, True):
            loss = tmp_path / 'loss.out'
            stdout = subprocess.check_output(
                [str(binary), str(loss), str(int(magnetic_split)), str(int(ediff))], text=True)
            assert stdout == loss.read_text()
            fields = stdout.split()
            assert fields[0] == '1000'
            values = expected + (split if magnetic_split else [])
            values += [.00000017, .00000029] if ediff else []
            assert len(fields) == len(values) + 1
            for field, value in zip(fields[1:], values):
                assert re.fullmatch(r'-?\d+\.\d{5}', field), field
                assert abs(float(field) - value) < 5.1e-6, (field, value)
            if ediff:
                assert fields[-2:] == ['0.00000', '0.00000']


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='spin-loss-output-') as directory:
        test_spin_loss_output(Path(directory))
    print('Spin loss output: all four column layouts passed')
