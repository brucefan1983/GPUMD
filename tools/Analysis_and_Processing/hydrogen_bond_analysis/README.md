# Hydrogen-bond analysis

`Hydrogen-bond-analysis.py` identifies hydrogen bonds using distance and
angle criteria. Optional criteria also cover π-H, dihydrogen, and metal-H
contacts. It requires Python 3, NumPy, SciPy, tqdm, and NetworkX.

## Input and usage

Use a multi-frame XYZ trajectory whose atom lines begin with
`element x y z`, with coordinates in Å. For periodic input, use an
axis-aligned orthorhombic `Lattice="..."` written in decimal notation;
the script applies periodicity in all three directions. Without `Lattice`,
it uses nonperiodic distances.

Run from the script directory:

```bash
python3 Hydrogen-bond-analysis.py -h
python3 Hydrogen-bond-analysis.py trajectory.xyz -dt O -at O -np 4
```

The second command selects oxygen donors and acceptors and uses four processes.
The default donor and acceptor element lists are both `O,N,F`.

| Option | Meaning (default) |
| --- | --- |
| `-R_cut` | Donor–acceptor distance cutoff in Å (3.5). |
| `-angle_cut` | H–donor–acceptor angle cutoff in degrees, with the vertex at the donor (30). |
| `-dH_cut` | Donor–H distance cutoff in Å (1.2). |
| `-max_H` | Maximum number of nearby H atoms considered per donor (2). |
| `-di`, `-ai` | Donor and acceptor indices, starting at 1; for example, `1-10,15`. Element filters still apply. |
| `-frame` | Frame index or inclusive range, starting at 0; for example, `0-100` (default: all frames). |
| `-un y` | Enable unconventional contacts (disabled by default). |
| `-pi_list`, `-metal_types` | With `-un y`, specify π-center atom groups or metal elements; omitted values are detected from the first frame. |
| `-analyze_geom n` | Disable geometry histograms (enabled by default). |

See `-h` for all options.

## Output

Files are written in the working directory and overwritten on subsequent runs:

- `hbond_count.txt`: counts for each frame, separated by contact type when
  unconventional contacts are enabled.
- `hbond_geom_all.txt`: atom indices, distances, angles, and contact types.
- `*_distribution.txt`: distance and angle histograms for detected contacts,
  when geometry analysis is enabled.

Author: Chen Zherui (chenzherui0124@foxmail.com).
