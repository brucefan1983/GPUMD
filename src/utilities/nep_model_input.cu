/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "error.cuh"
#include "nep_file.cuh"
#include "nep_model_file.cuh"
#include "nep_model_input.cuh"
#include <iomanip>
#include <limits>
#include <locale>

bool NEP_Model_Input::is_new_format(const std::string& token)
{
  return token.size() == 3 && (token[0] == 'n' || token[0] == 'N') &&
         (token[1] == 'e' || token[1] == 'E') && (token[2] == 'p' || token[2] == 'P');
}

std::vector<std::string> NEP_Model_Input::read_header(std::ifstream& input)
{
  const std::streampos start = input.tellg();
  std::vector<std::string> tokens = ::get_tokens(input);
  if (!tokens.empty() && is_new_format(tokens[0])) {
    input.clear();
    input.seekg(start);
    std::vector<std::string> elements;
    std::string error;
    if (!NEP_File::read_header(input, elements, error)) {
      PRINT_INPUT_ERROR(error.c_str());
    }
    tokens = {"nep", std::to_string(elements.size())};
    tokens.insert(tokens.end(), elements.begin(), elements.end());
  }
  return tokens;
}

NEP_Model_Input::NEP_Model_Input(const std::string& filename) : file(filename)
{
  if (!file.is_open()) {
    const std::string error = "Cannot open potential file " + filename + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  const std::vector<std::string> header = read_header(file);
  file.clear();
  file.seekg(0);
  new_format = !header.empty() && is_new_format(header[0]);
  if (!new_format) {
    return;
  }

  NEP_Model_File model;
  std::string error;
  if (!model.read(file, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (model.num_neurons2 != 0) {
    PRINT_INPUT_ERROR("Native MD supports only one hidden layer.");
  }
  if (model.zbl_parameters.size() > 550) {
    PRINT_INPUT_ERROR("Native MD supports flexible ZBL for at most 10 active elements.");
  }
  if (
    model.max_radial < 1 || model.max_radial > 819 || model.max_angular < 1 ||
    model.max_angular > std::numeric_limits<int>::max() / 2) {
    PRINT_INPUT_ERROR("Invalid neighbor capacities for native MD.");
  }

  std::string tag = model.enable_zbl ? "nep4_zbl" : "nep4";
  if (model.name == "nep_temperature") {
    tag += "_temperature";
  } else if (model.name == "qnep1") {
    tag += "_charge1";
  } else if (model.name == "qnep2") {
    tag += "_charge2";
  } else if (model.name == "nep_dipole") {
    tag += "_dipole";
  } else if (model.name == "nep_polarizability") {
    tag += "_polarizability";
  } else if (model.name != "nep") {
    PRINT_INPUT_ERROR("This model is not supported by native MD.");
  }

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  output << tag << ' ' << model.num_types;
  for (const std::string& element : model.elements) {
    output << ' ' << element;
  }
  output << '\n';
  if (model.enable_zbl) {
    output << "zbl " << model.zbl_rc_inner << ' ' << model.zbl_rc_outer;
    if (model.use_typewise_cutoff_zbl) {
      output << ' ' << model.typewise_cutoff_zbl_factor;
    }
    output << '\n';
  }
  output << "cutoff";
  const int count = model.has_multiple_cutoffs ? model.num_types : 1;
  for (int t = 0; t < count; ++t) {
    output << ' ' << model.rc_radial[t] << ' ' << model.rc_angular[t];
  }
  output << ' ' << model.max_radial << ' ' << model.max_angular << '\n';
  output << "n_max " << model.n_max_radial << ' ' << model.n_max_angular << '\n';
  output << "basis_size " << model.basis_size_radial << ' ' << model.basis_size_angular << '\n';
  output << "l_max " << model.L_max << ' ' << 2 * model.has_q_222 << ' ' << model.has_q_1111;
  const int extra_flags[] = {model.has_q_112, model.has_q_123, model.has_q_233, model.has_q_134};
  int last_flag = 3;
  while (last_flag >= 0 && extra_flags[last_flag] == 0) {
    --last_flag;
  }
  for (int i = 0; i <= last_flag; ++i) {
    output << ' ' << extra_flags[i];
  }
  output << '\n';
  output << "ANN " << model.num_neurons1 << " 0\n";
  for (const float value : model.parameters) {
    output << value << '\n';
  }
  for (const float value : model.q_scaler) {
    output << value << '\n';
  }
  for (const float value : model.zbl_parameters) {
    output << value << '\n';
  }
  converted.str(output.str());
  reference.swap(model.energy_reference);
}

std::vector<std::string> NEP_Model_Input::get_tokens()
{
  std::string line;
  std::getline(new_format ? static_cast<std::istream&>(converted) : file, line);
  return ::get_tokens(line);
}

const std::vector<double>& NEP_Model_Input::energy_reference() const { return reference; }
