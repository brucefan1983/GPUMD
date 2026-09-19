# Shared systems

`cu_grouped.xyz` is the 32-atom FCC Cu cell already used by the Master
examples, augmented with two grouping methods. Grouping method 0 has three
non-empty groups for heat baths, fixed/moving atoms, and TTM tests. Grouping
method 1 contains every atom in group 0.

Most Cu cases run `replicate 6 6 6` before loading
`potentials/eam/Cu_Zhou_2004.txt`, giving 6912 atoms and a box length of
43.38 Angstrom. This is large enough to represent a normal GPU workload while
remaining inexpensive for the short regression trajectories. The EAM cutoff
is 5 Angstrom and the candidate neighbor list adds a 1 Angstrom skin. Wall
cases use `replicate 12 6 6`, giving 13824 atoms and an x length of
86.76 Angstrom.

Triclinic and the compact Si `ti_liquid` regression case reuse the two-atom
primitive Si cell stored in `../phonon/si_primitive.xyz` and replicate it
`16 16 16` before loading `potentials/nep/Si_2022_NEP4_3body.txt`, giving
8192 atoms.

The group-based `heat_nhc`, `heat_bdp`, and `heat_lan` cases use the
self-contained `graphene_grouped_40400.xyz` fixture together with
`potentials/tersoff/Graphene_Lindsay_2010_modified.txt`. Group 0 is fixed, and
groups 1 and 8 are the heat source and sink.

`temperature_F_512.xyz` and `../potentials/temperature_F_nep.txt` are the
user-supplied temperature-dependent NEP fixture. The 512-atom fluorine model
has a 25.30329523 Angstrom cubic periodic box and deliberately stays on the
NEP large-box path. Their pinned SHA-256 values are:

- model: `44a76a943b6e1c419706e0f0b257200d38319194c80a60c589da977f7620b57c`;
- potential: `261e236885f88b785d832123ce8d7d3a8dbb220dec962ebbec26654ca481ee0b`.

`../potentials/C_2022_NEP4_MODIFIED.txt` is the upstream observer test's
`C_2022_NEP3_MODIFIED.txt` with only its model header updated from `nep3` to
`nep4`, matching the model families accepted by this baseline's
`Force::parse_potential()`. Its SHA-256 is
`7319c9a7c21aa4cc66ba66c91cf4d4540dade2a28acab90f2529b39c26bbd459`.
The unmodified companion is staged directly from
`potentials/nep/C_2022_NEP4.txt`.


Additional self-contained systems support semantic regression checks for
active learning, TNEP dipole/polarizability response output, observer species
validation, and grouped MSD. Their response or test-specific NEP models are
stored under `../models/`; ordinary public GPUMD potentials continue to be
staged from the repository-level `potentials/` directory when available.
Legacy NEP3 fixtures used by these tests were converted to the equivalent
NEP4 parameter layout before being added here.
