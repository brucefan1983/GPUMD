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
import csv
from pathlib import Path

out = Path("out")
species = list(csv.DictReader(open(out / "water.species.csv")))
assert any(r["formula"] == "H2O" and r["count"] == "25" for r in species), species
coord = list(csv.DictReader(open(out / "diamond.coord.csv")))
assert any(
    r["element"] == "C" and r["coordination"] == "4" and r["count"] == "80"
    for r in coord
), coord
bond = list(csv.DictReader(open(out / "water.bond.csv")))
assert any(r["pair"] == "O-H" and r["count"] == "50" for r in bond), bond
append = list(csv.DictReader(open(out / "append.species.csv")))
assert sum(1 for r in append if r["formula"] == "H2O") == 2, append
print("react_count core regression: PASS")
PY
rm -rf out test_core
