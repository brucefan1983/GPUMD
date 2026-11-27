# Energy reference alignment

`Energy-Reference-Aligner.py` fits an energy offset for each element within
each selected `config_type` group, then subtracts the offsets weighted by
element counts from each structure's energy. It requires Python 3, NumPy,
ASE, and calorine.

## Configuration

Edit the `SETTINGS` section of the script:

- Set `input_xyz_file` (default: `train.xyz`) and a separate
  `output_xyz_file` (default: `output_aligned.xyz`).
- Select `ALIGNMENT_MODE` and its settings from the table below.
- Adjust the NES optimization settings or `nep_batch_size` if needed.

| Mode | Fitting target | Settings |
| --- | --- | --- |
| `REF_GROUP_ALIGNMENT` | The reference group's mean total energy. | Set `reference_group`; an empty `shift_groups` selects all other groups. |
| `ZERO_BASELINE_ALIGNMENT` | Zero total energy, minimizing the mean squared residual after subtracting elemental offsets. | Processes all groups. |
| `DFT_TO_NEP_ALIGNMENT` | The NEP energy calculated for each structure. | Set `nep_model_file`; processes all groups. |

Each input frame must contain an `energy=` header field, with element symbols
in the first atom column. Frames without `config_type` belong to
`default_group`. Use LF line endings; convert CRLF input with
`dos2unix train.xyz` before running.

## Run

From the script directory, after configuring the file paths:

```bash
python3 Energy-Reference-Aligner.py
```

The script prints the fitted elemental offsets and writes updated energies
and the original atom lines to `output_xyz_file`, overwriting that file if
it already exists.

Author: Chen Zherui (chenzherui0124@foxmail.com).
