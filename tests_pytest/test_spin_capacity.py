# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.

"""Exhaust the declared magnetic shapes through the production layout and reader."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "utilities/nep_spin/spin_layout.cuh"
#include "utilities/nep_io/nep_spin_model_file.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
void check_projection(const nep_spin::Layout& layout, bool transpose, bool paired = false, bool candidate = false) {
  const int channels = layout.channels;
  std::vector<float> weights(layout.path_count * channels);
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.25f + i;
  nep_spin::SpinProjection projection(layout, candidate ? nullptr : weights.data(), transpose, paired);
  const int harmonics = (layout.lmax + 1) * (layout.lmax + 1);
  const int outputs = transpose ? harmonics * layout.spin_components : layout.components;
  std::vector<std::vector<nep_spin::Term>> expected(outputs);
  for (auto term : layout.terms) {
    const int density = term.geometry * layout.spin_components + (term.neighbor ? term.spin : 0);
    expected[transpose ? density : term.output].push_back(term);
  }
  if (paired) {
    const auto& rows = projection.rows;
    if (std::all_of(rows.begin(), rows.end(), [](const nep_spin::ProjectionRow& row) { return row.count == 1; })) {
      for (int m = 0; m < outputs; ++m) {
        assert(rows[m].output == m);
        assert(rows[m].begin == (candidate ? m : (m / projection.width) * channels * projection.width + m % projection.width));
      }
    }
    // Multiply the packed and original tables by the same adjoint, separately for each site.
    for (int c = 0; c < channels; ++c) {
      std::vector<double> actual[2], reference[2];
      for (int site = 0; site < 2; ++site) {
        actual[site].resize(outputs);
        reference[site].resize(outputs);
      }
      for (auto term : layout.terms) {
        const float weight = weights[term.path * channels + c] * term.coefficient;
        reference[term.neighbor][term.geometry * layout.spin_components + term.spin] +=
          double(weight) * (1 + term.output);
      }
      for (auto row : projection.rows) {
        for (int k = 0; k < row.count; ++k) {
          const auto index = row.begin + (candidate ? k : k * channels + c) * projection.width;
          const auto term = projection.packed.at(index);
          float center = term.weight, neighbor = projection.neighbor_weights.at(index);
          if (candidate) {
            center *= weights.at(projection.paths.at(2 * index) * channels + c);
            neighbor *= weights.at(projection.paths.at(2 * index + 1) * channels + c);
          }
          actual[0][row.output] += double(center) * (1 + term.input);
          actual[1][row.output] += double(neighbor) * (1 + term.input);
        }
      }
      for (int site = 0; site < 2; ++site)
        for (int k = 0; k < outputs; ++k)
          assert(std::abs(actual[site][k] - reference[site][k]) < 1e-8);
    }
    return;
  }
  std::vector<int> visits(outputs);
  for (auto row : projection.rows) {
    const auto& terms = expected.at(row.output);
    assert(++visits[row.output] == 1 && row.count == int(terms.size()));
    for (int k = 0; k < row.count; ++k) for (int c = 0; c < channels; ++c) {
      const int index = row.begin + (candidate ? k : k * channels + c) * projection.width;
      auto packed = projection.packed.at(index);
      auto term = terms[k];
      const int density = term.geometry * layout.spin_components + (term.neighbor ? term.spin : 0);
      assert(packed.input == (transpose ? term.output : density));
      assert(packed.center_spin == (term.neighbor ? -1 : term.spin));
      if (candidate) {
        assert(projection.paths.at(index) == term.path && packed.weight == term.coefficient);
      } else {
        assert(packed.weight == weights[term.path * channels + c] * term.coefficient);
      }
    }
  }
  assert(std::count(visits.begin(), visits.end(), 1) == outputs);
}
void check_harmonic_derivatives() {
  // Compare coefficients of every monomial in the Cartesian derivative polynomials.
  double expected[25][3][5][5][5] = {}, actual[25][3][5][5][5] = {};
  for (auto h : nep_spin::harmonics) {
    for (int axis = 0; axis < 3; ++axis) {
      int p[] = {h.x, h.y, h.z};
      if (p[axis] == 0) continue;
      double coefficient = h.value * p[axis]--;
      expected[h.component][axis][p[0]][p[1]][p[2]] += coefficient;
    }
  }
  for (auto t : nep_spin::harmonic_derivatives)
    for (auto h : nep_spin::harmonics)
      if (t.input == h.component)
        actual[t.output][t.axis][h.x][h.y][h.z] += t.coefficient * h.value;
  for (int g = 0; g < 25; ++g) for (int axis = 0; axis < 3; ++axis)
    for (int x = 0; x <= 4; ++x) for (int y = 0; y <= 4-x; ++y)
      for (int z = 0; z <= 4-x-y; ++z)
        assert(std::abs(actual[g][axis][x][y][z] - expected[g][axis][x][y][z]) < 1e-12);
}
template<int Tile>
void check_radial_edge_indices() {
  constexpr int atoms = 5, neighbors = 37;
  for (int channels = 1; channels <= 9; ++channels) {
    std::vector<bool> used((neighbors + Tile - 1) / Tile * Tile * atoms * channels);
    for (int i = 0; i < atoms; ++i)
      for (int c = 0; c < channels; ++c)
        for (int j = 0; j < neighbors; ++j) {
          const auto index = nep_spin::radial_edge_index<Tile>(atoms, channels, i, c, j);
          assert(!used.at(index));
          used[index] = true;
        }
  }
}
int main() {
  check_harmonic_derivatives();
  check_radial_edge_indices<1>();
  check_radial_edge_indices<2>();
  check_radial_edge_indices<4>();
  check_radial_edge_indices<32>();
  int count=0, extended=0, max_dim=0, max_components=0, max_spins=0;
  for(int c=1;c<=9;c++) for(int l=0;l<=4;l++)
  for(int o=1;o<=4;o++) for(int soc=0;soc<=1;soc++) {
    nep_spin::Layout v(c,l,o,soc);
    check_projection(v, false);
    check_projection(v, false, false, true);
    check_projection(v, true);
    check_projection(v, true, false, true);
    check_projection(v, true, true);
    check_projection(v, true, true, true);
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
