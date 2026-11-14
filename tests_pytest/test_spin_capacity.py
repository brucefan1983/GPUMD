"""Exhaust the declared magnetic shapes through the production layout and reader."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "force/nep_spin/spin_layout.cuh"
#include "utilities/nep_io/nep_spin_model_file.cuh"
#include <cassert>
#include <iostream>
#include <sstream>
int main() {
  int count=0, extended=0, max_dim=0, max_components=0, max_spins=0;
  for(int c=1;c<=9;c++) for(int l=0;l<=4;l++)
  for(int o=1;o<=4;o++) for(int soc=0;soc<=1;soc++) {
    nep_spin::Layout v(c,l,o,soc);
    for(auto t:v.terms) {
      assert(t.geometry>=0 && t.geometry<(l+1)*(l+1));
      assert(t.spin>=0 && t.spin<v.spin_components);
      assert(t.output>=0 && t.output<v.components);
      assert(t.path>=0 && t.path<v.path_count);
    }
    for(auto s:v.sectors) {
      assert(s.offset>=0 && s.offset+s.size<=v.components);
      assert(s.norm_column>=0 && s.norm_column+c<=v.dimension);
      assert(s.scalar_column<0 || s.scalar_column+c<=v.dimension);
    }
    for(auto p:v.primitives) assert(p.offset+2*p.l+1<=v.spin_components);
    for(auto p:v.centers) {
      assert(p.primitive>=0 && p.primitive<v.spin_components);
      assert(p.output>=0 && p.output<v.components);
      assert(p.weight>=0 && p.weight<int(v.primitives.size())-1);
    }
    // Maximal existing structural shape: 13 radial + 90 angular coordinates.
    const int dim=103+v.dimension, projection=v.parameter_count(9);
    const int parameters=4*(dim+2)+1+13+81+projection;
    std::ostringstream text;
    text << "nep 1 Fe\nformat_version 1\nmodel nep_spin\nspin_version 1\n"
      "cutoff 6 5 128 128\nn_max 12 8\nbasis_size 0 8\nl_max 8 2 1\nann 4 0\n"
      "spin_basis_size 8\nspin_cutoff 6\nspin_scaler 1\nspin_type Fe\nenergy_reference 0\n"
      "spin_compress " << c << "\nspin_l_max " << l << "\nspin_order " << o
      << "\nspin_soc " << soc << "\n@ parameters " << parameters << '\n';
    for(int i=0;i<parameters;i++) text << "0\n";
    text << "@ q_scaler " << dim << '\n';
    for(int i=0;i<dim;i++) text << "1\n";
    std::istringstream input(text.str()); NEP_Spin_Model_File model; std::string error;
    if(!model.read(input,error)) { std::cerr << error; return 1; }
    assert(model.dim_struct==103 && model.dim_spin==v.dimension && model.dim==dim);
    assert(dim<=NEP_Spin_Model_File::MAX_DIM_SPIN);
    assert(v.components<=625 && v.spin_components<=35);
    std::cout << c << ' ' << l << ' ' << o << ' ' << soc << ' '
              << v.dimension << ' ' << projection << '\n';
    count++; extended+=v.dimension>96;
    max_dim=std::max(max_dim,dim); max_components=std::max(max_components,v.components);
    max_spins=std::max(max_spins,v.spin_components);
  }
  assert(count==360 && extended==55 && max_dim==339 && max_components==625 && max_spins==35);
}
'''


def layout_rows(directory):
    source, binary = directory / 'capacity.cpp', directory / 'capacity'
    source.write_text(SOURCE)
    subprocess.run(['c++', '-std=c++14', '-O2', '-I'+str(ROOT/'src'), str(source),
                    '-x', 'c++', str(ROOT/'src/utilities/nep_io/nep_spin_model_file.cu'),
                    str(ROOT/'src/utilities/nep_io/nep_file.cu'), '-o', str(binary)], check=True)
    return [tuple(map(int, line.split())) for line in
            subprocess.check_output([binary], text=True).splitlines()]


def test_all_declared_spin_shapes(tmp_path):
    assert len(layout_rows(tmp_path)) == 360


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        test_all_declared_spin_shapes(Path(directory))
    print('360 shapes: layout indices, maximal structural shape and model reader passed')
