/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>
#include <vector>

struct RunInputLine
{
  std::string raw_line;
  std::vector<std::string> tokens;
  int line_number = 0;
};

class RunInput
{
public:
  explicit RunInput(const std::string& filename);

  const std::vector<RunInputLine>& lines() const;
  const RunInputLine* find_first(const std::string& keyword) const;
  const RunInputLine* find_last(const std::string& keyword) const;
  std::vector<const RunInputLine*> find_all(const std::string& keyword) const;
  bool contains(const std::string& keyword) const;

private:
  std::vector<RunInputLine> lines_;
};
