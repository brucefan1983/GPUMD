# Select XYZ frames

`split_xyz.py` selects frames from an extended XYZ dataset using Python 3
and its standard library. Each input frame's comment line must contain
`energy` or `energies` in lowercase.

| Option | Meaning |
| --- | --- |
| `-i`, `--input` | Input dataset (required). |
| `--ranges` | Comma-separated frame indices and ranges (required). Indices start at 0; `start:end` excludes `end`. |
| `--nframes` | Number of frames to select randomly. |
| `--perrange` | Select up to `--nframes` frames from each range; keep all frames in shorter ranges. |

Duplicate selections are removed and the output order is randomized, including
when `--nframes` is omitted. Selected XYZ frames go to standard output;
the selected indices are printed to standard error.

## Examples

Run from the script directory. These commands overwrite `output.xyz`;
use an output filename different from the input.

Select frames 0, 1, 4, 7, and 8:

```bash
python3 split_xyz.py -i input.xyz --ranges "0:2,4,7:9" > output.xyz
```

Select 10 frames in total from frames 0–99 and 200–299:

```bash
python3 split_xyz.py -i input.xyz --ranges "0:100,200:300" --nframes 10 > output.xyz
```

Select 10 frames from each of these ranges (20 in total):

```bash
python3 split_xyz.py -i input.xyz --ranges "0:100,200:300" --nframes 10 --perrange > output.xyz
```

Select 10, 5, and 10 frames from the three ranges (25 in total):

```bash
python3 split_xyz.py -i input.xyz --ranges "0:100,110:115,200:300" --nframes 10 --perrange > output.xyz
```

Author: Yong Wang (yongw@princeton.edu).
