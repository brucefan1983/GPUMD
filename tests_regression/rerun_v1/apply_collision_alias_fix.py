#!/usr/bin/env python3
"""Apply the reproduced rerun XYZ path-alias collision fix to a checkout."""

import sys
from pathlib import Path


OLD = '''bool Dump_XYZ::rerun_overwrites_input(const std::string& path) const
{
  return separated_ == 0 && split_frames_ == 0 && filename_ == path;
}'''

NEW = '''bool Dump_XYZ::rerun_overwrites_input(const std::string& path) const
{
  if (separated_ != 0 || split_frames_ > 0) return false;
  if (filename_ == path) return true;

#ifdef _WIN32
  char output_path[_MAX_PATH];
  char input_path[_MAX_PATH];
  if (_fullpath(output_path, filename_.c_str(), _MAX_PATH) == nullptr ||
      _fullpath(input_path, path.c_str(), _MAX_PATH) == nullptr) {
    return false;
  }
  return _stricmp(output_path, input_path) == 0;
#else
  char output_path[PATH_MAX];
  char input_path[PATH_MAX];
  if (realpath(filename_.c_str(), output_path) == nullptr ||
      realpath(path.c_str(), input_path) == nullptr) {
    return false;
  }
  return std::strcmp(output_path, input_path) == 0;
#endif
}'''


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: apply_collision_alias_fix.py CHECKOUT [CHECKOUT ...]")
    for checkout in sys.argv[1:]:
        path = Path(checkout) / "src/measure/dump_xyz.cu"
        source = path.read_text()
        if NEW in source:
            print(f"already patched: {path}")
            continue
        if source.count(OLD) != 1:
            raise SystemExit(f"expected one unpatched rerun_overwrites_input in {path}")
        source = source.replace(OLD, NEW, 1)
        include = "#include <cstring>\n"
        replacement = "#include <cstring>\n#include <stdlib.h>\n#ifndef _WIN32\n#include <limits.h>\n#endif\n"
        if source.count(include) != 1:
            raise SystemExit(f"could not locate expected include block in {path}")
        source = source.replace(include, replacement, 1)
        path.write_text(source)
        print(f"patched: {path}")


if __name__ == "__main__":
    main()
