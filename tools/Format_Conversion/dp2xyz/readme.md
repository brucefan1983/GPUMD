# DeePMD NumPy datasets to NEP XYZ

[dp2xyz-raw-npy-mix.py](dp2xyz-raw-npy-mix.py) converts DeePMD NumPy datasets, including mixed-type datasets, to extended XYZ for NEP. It requires Python 3 and NumPy.

The script searches the input directory recursively for systems containing `type.raw`. Each system should contain:

| Location | Files |
| --- | --- |
| System directory | `type.raw` and `type_map.raw` (chemical symbols in type-index order). |
| Each `set.*` directory | `box.npy`, `coord.npy`, `energy.npy`, and `force.npy`. |
| Optional arrays in `set.*` | `virial.npy`; `real_atom_types.npy` for frame-dependent atom types. |

Run from this directory, using the included datasets:

```bash
python3 dp2xyz-raw-npy-mix.py deepmd-npy nepxyz-from-npy
python3 dp2xyz-raw-npy-mix.py deepmd-mixed nepxyz-from-mixed
```

The two arguments are the input directory and output directory. All frames are combined into `train.xyz` in the output directory; an existing file is overwritten. The output includes energies, forces, cells, and available virials, with `pbc="T T T"`.

See the [DeePMD mixed-type format](https://docs.deepmodeling.com/projects/dpdata/en/master/systems/mixed.html) and [sample datasets](https://github.com/deepmodeling/AIS-Square/tree/main/datasets).
