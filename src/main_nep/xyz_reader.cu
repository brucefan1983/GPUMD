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

#include "xyz_reader.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cctype>

static std::vector<std::string> get_properties(const XYZ_Record& record)
{
  for (const auto& token : record.tokens) {
    const std::string prefix = "properties=";
    if (token.substr(0, prefix.size()) == prefix) {
      std::string line = token.substr(prefix.size());
      std::replace(line.begin(), line.end(), ':', ' ');
      return get_tokens(line);
    }
  }
  return {};
}

int XYZ_Record::get_property_offset(const char* name, const char* alternative) const
{
  int offset = 0;
  int result = -1;
  for (int k = 0; k < properties.size() / 3; ++k) {
    if (properties[k * 3] == name || (alternative && properties[k * 3] == alternative)) {
      result = offset;
    }
    offset += get_int_from_token(properties[k * 3 + 2], filename.c_str(), comment_line_number);
  }
  return result;
}

bool XYZ_Reader::read(XYZ_Record& record)
{
  auto tokens = get_tokens(input);
  ++line;
  if (tokens.empty()) {
    return false;
  }
  if (tokens.size() != 1) {
    PRINT_INPUT_ERROR("The first line for each frame should have one value.");
  }
  record.num_atoms = get_int_from_token(tokens[0], filename.c_str(), line);
  if (record.num_atoms < 1) {
    PRINT_INPUT_ERROR("Number of atoms for each frame should >= 1.");
  }
  record.index = index++;
  record.filename = filename;
  std::getline(input, record.comment);
  record.comment_line_number = ++line;
  record.tokens = get_tokens_without_unwanted_spaces(record.comment);
  for (auto& token : record.tokens) {
    std::transform(
      token.begin(), token.end(), token.begin(), [](unsigned char c) { return std::tolower(c); });
  }
  if (record.tokens.empty()) {
    PRINT_INPUT_ERROR("The second line for each frame should not be empty.");
  }
  int num_columns = 0;
  record.properties = get_properties(record);
  for (int k = 0; k < record.properties.size() / 3; ++k) {
    num_columns += get_int_from_token(record.properties[k * 3 + 2], filename.c_str(), line);
  }
  record.atoms.resize(record.num_atoms);
  for (auto& atom : record.atoms) {
    atom = get_tokens(input);
    ++line;
    if (atom.size() != num_columns) {
      PRINT_INPUT_ERROR("Number of items for an atom line mismatches properties.");
    }
  }
  return true;
}
