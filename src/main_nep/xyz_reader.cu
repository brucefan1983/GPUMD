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

/*----------------------------------------------------------------------------80
Major contributors:
  Zheyong Fan: Initial implementation.
------------------------------------------------------------------------------*/

#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include "xyz_reader.cuh"
#include <algorithm>
#include <cctype>
#include <climits>

static void xyz_input_error(const XYZ_Record& record, const std::string& message)
{
  const std::string text =
    record.filename + " line " + std::to_string(record.comment_line_number) + ": " + message;
  PRINT_INPUT_ERROR(text.c_str());
}

static std::vector<std::string> get_properties(const XYZ_Record& record)
{
  std::vector<std::string> properties;
  bool found = false;
  for (const auto& token : record.tokens) {
    const std::string prefix = "properties=";
    if (token.substr(0, prefix.size()) == prefix) {
      if (found) {
        xyz_input_error(record, "Duplicate properties field.");
      }
      found = true;
      std::string line = token.substr(prefix.size());
      if (
        line.empty() || line.front() == ':' || line.back() == ':' ||
        line.find("::") != std::string::npos) {
        xyz_input_error(record, "Properties must contain complete name:type:columns triples.");
      }
      std::replace(line.begin(), line.end(), ':', ' ');
      properties = get_tokens(line);
      if (properties.size() % 3 != 0) {
        xyz_input_error(record, "Properties must contain complete name:type:columns triples.");
      }
    }
  }
  if (!found) {
    xyz_input_error(record, "Missing properties field.");
  }
  return properties;
}

int XYZ_Record::get_property_offset(
  const char* name, char type, int columns, const char* alternative) const
{
  int offset = 0;
  int result = -1;
  for (int k = 0; k < properties.size() / 3; ++k) {
    const int count =
      get_int_from_token(properties[k * 3 + 2], filename.c_str(), comment_line_number);
    if (properties[k * 3] == name || (alternative && properties[k * 3] == alternative)) {
      if (result >= 0) {
        xyz_input_error(*this, std::string("Duplicate property or alias for ") + name + ".");
      }
      if (properties[k * 3 + 1] != std::string(1, std::tolower(type)) || count != columns) {
        xyz_input_error(
          *this,
          std::string(name) + " must have property type " + type + ":" + std::to_string(columns) +
            ".");
      }
      result = offset;
    }
    offset += count;
  }
  return result;
}

void XYZ_Record::validate_array(int first, int count) const
{
  const std::string prefix = tokens[first].substr(0, tokens[first].find('=') + 1);
  const std::string name = prefix.substr(0, prefix.size() - 1);
  for (int n = 0; n < tokens.size(); ++n) {
    if (n != first && tokens[n].compare(0, prefix.size(), prefix) == 0) {
      xyz_input_error(*this, "Duplicate " + name + " field.");
    }
  }
  const std::string message =
    name + " must contain exactly " + std::to_string(count) + " values in double quotes.";
  if (
    tokens.size() - first < count || tokens[first].substr(prefix.size(), 1) != "\"" ||
    tokens[first + count - 1].back() != '"') {
    xyz_input_error(*this, message);
  }
  for (int n = 0; n < count; ++n) {
    std::string value = tokens[first + n].substr(n == 0 ? prefix.size() + 1 : 0);
    if (n == count - 1) {
      value.pop_back();
    }
    if (value.empty() || value.find('"') != std::string::npos) {
      xyz_input_error(*this, message);
    }
  }
}

bool XYZ_Reader::read(XYZ_Record& record)
{
  std::vector<std::string> tokens;
  while (tokens.empty()) {
    std::string frame_line;
    if (!std::getline(input, frame_line)) {
      if (input.eof() && !input.bad()) {
        return false;
      }
      const std::string message =
        "Failed to read " + filename + " at line " + std::to_string(line + 1) + ".";
      PRINT_INPUT_ERROR(message.c_str());
    }
    ++line;
    tokens = get_tokens(frame_line);
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
    const int count = get_int_from_token(record.properties[k * 3 + 2], filename.c_str(), line);
    if (count <= 0 || count > INT_MAX - num_columns) {
      xyz_input_error(
        record, "Properties must have positive column counts with a total <= INT_MAX.");
    }
    num_columns += count;
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
