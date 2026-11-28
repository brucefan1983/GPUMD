# MEAM parameter files

Each subdirectory contains one LAMMPS-compatible MEAM parameter set and a
`meam.txt` descriptor for it:

| Directory | Descriptor | System |
| --- | --- | --- |
| `Fe` | `meam.txt` | single-element Fe with the `nn2(I,J)` second-nearest-neighbor extension |
| `Ge` | `meam.txt` | single-element Ge |
| `Ga_In` | `meam.txt` | Ga-In binary |
| `Al_Co_Fe` | `meam.txt` | Al-Co-Fe ternary with the `nn2(I,J)` extension |

`library.meam` holds the elemental parameters and the optional `*.meam` file
holds the pair-specific overrides. Both use the format of
[LAMMPS pair_style meam](https://docs.lammps.org/pair_meam.html) in metal
units (eV and Å). The `meam.txt` descriptor is read by GPUMD:

```text
meam 1 Fe
library.meam
Fe.meam
```

The first line gives the number of elements and their names, the second line
names the library file, and the third line names the override file (or `NULL`).
Relative paths are resolved from the descriptor's directory. An optional fourth
line with `library_elements` selects alternate library entries by name.

The parameter sets are adapted from the LAMMPS MEAM example and documentation
input files. They are intended for testing and for reproducing the LAMMPS
documentation examples; for production work, cite the original parameter
papers.

## References

* M. I. Baskes,
  Modified embedded-atom potentials for cubic materials and impurities,
  Phys. Rev. B **46**, 2727 (1992).
  https://doi.org/10.1103/PhysRevB.46.2727
* B.-J. Lee and M. I. Baskes,
  Second nearest-neighbor modified embedded-atom-method potential,
  Phys. Rev. B **62**, 8564 (2000).
  https://doi.org/10.1103/PhysRevB.62.8564
* B.-J. Lee, M. I. Baskes, H. Kim, and Y. K. Cho,
  Second nearest-neighbor modified embedded atom method potentials for bcc
  transition metals,
  Phys. Rev. B **64**, 184102 (2001).
  https://doi.org/10.1103/PhysRevB.64.184102
