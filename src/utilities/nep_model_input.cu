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
  std::string input;
  if (!model.get_md_input(input, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  converted.str(input);
  reference.swap(model.energy_reference);
}

std::vector<std::string> NEP_Model_Input::get_tokens()
{
  std::string line;
  std::getline(new_format ? static_cast<std::istream&>(converted) : file, line);
  return ::get_tokens(line);
}

const std::vector<double>& NEP_Model_Input::energy_reference() const { return reference; }

bool NEP_Model_Input::channel_major() const { return new_format; }
