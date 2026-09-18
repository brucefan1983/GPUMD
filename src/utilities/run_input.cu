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

#include "run_input.cuh"
#include "error.cuh"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <utility>

RunInput::RunInput(const std::string& filename)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << filename << "." << std::endl;
    exit(1);
  }

  std::string raw_line;
  int line_number = 0;
  while (std::getline(input, raw_line)) {
    RunInputLine line;
    line.raw_line = raw_line;
    line.tokens = get_tokens_without_comments(raw_line);
    line.line_number = ++line_number;
    lines_.emplace_back(std::move(line));
  }
}

const std::vector<RunInputLine>& RunInput::lines() const { return lines_; }

const RunInputLine* RunInput::find_first(const std::string& keyword) const
{
  for (const auto& line : lines_) {
    if (!line.tokens.empty() && line.tokens[0] == keyword) {
      return &line;
    }
  }
  return nullptr;
}

const RunInputLine* RunInput::find_last(const std::string& keyword) const
{
  for (auto line = lines_.rbegin(); line != lines_.rend(); ++line) {
    if (!line->tokens.empty() && line->tokens[0] == keyword) {
      return &(*line);
    }
  }
  return nullptr;
}

std::vector<const RunInputLine*> RunInput::find_all(const std::string& keyword) const
{
  std::vector<const RunInputLine*> matches;
  for (const auto& line : lines_) {
    if (!line.tokens.empty() && line.tokens[0] == keyword) {
      matches.push_back(&line);
    }
  }
  return matches;
}

bool RunInput::contains(const std::string& keyword) const
{
  return find_first(keyword) != nullptr;
}
