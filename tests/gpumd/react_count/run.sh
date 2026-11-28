#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O2 -std=c++14 -x c++ test_core.cpp -o test_core
rm -rf out
mkdir -p out
./test_core water25.xyz out/water mutual
./test_core diamond80.xyz out/diamond reax
./test_core water25.xyz out/append mutual
./test_core water25.xyz out/append mutual
python3 - <<'PY'
from pathlib import Path

def rows(path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split())
    return out

water = rows(Path("out/water.species.out"))
assert any(r == ["1", "H2O", "25"] for r in water), water
bond = rows(Path("out/water.bond.out"))
assert any(r == ["1", "O-H", "50"] for r in bond), bond
diamond = rows(Path("out/diamond.coord.out"))
assert any(r == ["1", "C", "4", "80"] for r in diamond), diamond
append = rows(Path("out/append.species.out"))
assert sum(1 for r in append if r[1] == "H2O") == 2, append
print("react_count core regression: PASS")
PY
rm -rf out test_core
