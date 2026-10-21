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
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace nep_file
{

struct Field
{
  std::string name;
  std::vector<std::string> values;
  std::size_t line = 0;
};

struct Block
{
  std::string name;
  std::vector<double> values;
  std::size_t line = 0;
};

struct Document
{
  std::vector<std::string> elements;
  std::vector<Field> fields;
  std::vector<Block> blocks;
};

const Field* find_field(const Document& document, const std::string& name);
const Block* find_block(const Document& document, const std::string& name);

bool parse_size(const std::string& token, std::size_t& value);
bool parse_double(const std::string& token, double& value);

bool read(std::istream& input, Document& document, std::string& error);
bool write(std::ostream& output, const Document& document, std::string& error);

bool check_unhandled(
  const Document& document,
  const std::vector<std::string>& handled_fields,
  const std::vector<std::string>& handled_blocks,
  std::string& error);

}
