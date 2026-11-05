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

#include <cstdio>
#include <sys/stat.h>
#include <iomanip>
#include <sstream>
#include <string>

// Track frame boundaries and reserve a fresh numbered filename for each chunk.
// Each run starts a new sequence; pre-existing chunks are never overwritten.
class XYZFrameSplitter
{
public:
  XYZFrameSplitter() = default;

  void reset(const std::string& filename, int frames_per_file)
  {
    filename_ = filename;
    frames_per_file_ = frames_per_file;
    frames_in_file_ = 0;
    next_index_ = 1;
  }

  bool needs_new_file() const { return frames_in_file_ == 0; }

  void frame_written()
  {
    if (++frames_in_file_ == frames_per_file_) {
      frames_in_file_ = 0;
    }
  }

  std::string next_filename()
  {
    // Do not overwrite previous output, including across restarts or multiple run commands.
    for (;;) {
      const std::string candidate = numbered_filename(next_index_++);
      struct stat file_info;
      if (::stat(candidate.c_str(), &file_info) != 0) {
        return candidate;
      }
    }
  }

private:
  std::string numbered_filename(unsigned long long index) const
  {
    std::ostringstream suffix;
    suffix << '_' << std::setfill('0') << std::setw(4) << index;
    const auto slash = filename_.find_last_of("/\\");
    const auto dot = filename_.find_last_of('.');
    if (dot != std::string::npos && dot > 0 &&
        (slash == std::string::npos || dot > slash + 1)) {
      return filename_.substr(0, dot) + suffix.str() + filename_.substr(dot);
    }
    return filename_ + suffix.str();
  }

  std::string filename_;
  int frames_per_file_ = 0;
  int frames_in_file_ = 0;
  unsigned long long next_index_ = 1;
};
