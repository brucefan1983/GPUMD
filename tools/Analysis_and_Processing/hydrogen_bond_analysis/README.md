# Hydrogen Bond Analyzer for XYZ Trajectories

This script analyzes hydrogen bonds (H-bonds) in molecular dynamics or quantum chemistry trajectories provided in extended XYZ format. It supports both conventional and unconventional H-bonds, including pi-H bonds (e.g., benzene rings, alkynes), dihydrogen bonds, and metal-H bonds.

---

## 🧪 Theory

Hydrogen bonds are identified based on geometric criteria:

- The distance between donor and acceptor atoms (D...A) must be below a threshold (e.g., **< 3.5 Å**).
- The angle between Hydrogen–Donor–Acceptor must be sharp (e.g., **< 30°**).
- The Hydrogen–Donor bond distance must be below a cutoff (e.g., **< 1.2 Å**).

Unconventional H-bonds are also recognized by geometry:

- **π-H bonds**: H atom interacts with the π electron system (e.g., benzene ring, triple bond).
- **Dihydrogen bonds**: interaction between X–H and H–M (M = metal or boron).
- **Metal-H bonds**: H atom interacts with a metal atom.

---

## 🚀 Features

- Reads multi-frame `.xyz` trajectory files (supports periodic boundary conditions via `Lattice=`).
- Flexible donor and acceptor selection by **index** or **element type**.
- Automatic or manual detection of **π systems** and **metal centers**.
- **Parallel processing** for speed.
- Outputs per-frame H-bond counts and detailed geometric statistics.

---

## 📦 Usage

Show all options:

```bash
python hbond.py -h
```

## 📂 Outputs

- `hbond_count.txt`: Per-frame H-bond counts (regular and unconventional types)
- `hbond_geom_all.txt`: All H-bond geometries (frame, atom indices, distances, angles, type)
- Geometry distribution files: `Donor-Hydrogen_distribution.txt`, etc.

------

**Author:** chenzherui0124@foxmail.com