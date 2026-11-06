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

#include "nep_model_file.cuh"

bool NEP_Model_File::read_nep(
  const NEP_File& document,
  std::vector<std::string>& fields,
  std::vector<std::string>& blocks,
  std::string& error)
{
  if (!read_descriptor(document, fields, error) || !read_ann(document, fields, error)) {
    return false;
  }
  calculate_layout(name == "nep_temperature" ? 1 : 0, 0, 1, 1);
  return read_reference(document, fields, error) &&
         read_zbl(document, fields, blocks, error) &&
         read_parameters(document, blocks, error);
}

bool NEP_Model_File::get_nep_md_input(std::string& input, std::string& error) const
{
  std::string tag = enable_zbl ? "nep4_zbl" : "nep4";
  if (name == "nep_temperature") {
    tag += "_temperature";
  }
  return get_md_input(tag, input, error);
}

void NEP_Model_File::compact_nep(
  NEP_File& document, const std::vector<int>& active_to_full) const
{
  compact_parameters(document, active_to_full);
}
