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

#pragma once
#include <fstream>
#include <string>
#include <vector>

// One frame as written in an extended XYZ file, before model-specific interpretation.
struct XYZ_Record
{
  int index;
  int num_atoms;
  int comment_line_number;
  std::string filename;
  std::string comment;
  std::vector<std::string> tokens;
  std::vector<std::string> properties;
  std::vector<std::vector<std::string>> atoms;

  int get_property_offset(const char* name, const char* alternative = nullptr) const;
};

class XYZ_Reader
{
public:
  XYZ_Reader(std::ifstream& input, const std::string& filename)
    : input(input), filename(filename) {}
  bool read(XYZ_Record& record);

private:
  std::ifstream& input;
  std::string filename;
  int line = 0;
  int index = 0;
};
