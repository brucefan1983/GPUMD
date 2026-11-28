# VASP OUTCAR files to NEP XYZ

These Bash scripts recursively find files named `OUTCAR` beneath the input directory and combine the extracted configurations into extended XYZ. They require standard Unix text utilities and `dos2unix`.

| Script | Frames extracted | Output relative to the working directory |
| --- | --- | --- |
| [singleFrame-outcars2nep-exyz.sh](singleFrame-outcars2nep-exyz.sh) | Last frame of each OUTCAR; suitable for single-point calculations. | `NEPdataset/NEP-dataset.xyz` |
| [multipleFrames-outcars2nep-exyz.sh](multipleFrames-outcars2nep-exyz.sh) | Frames marked as electronically converged by VASP; suitable for AIMD and relaxation trajectories. | `NEPdataset-multiple_frames/NEP-dataset.xyz` |

Run from this directory, replacing `OUTCARs` with your input directory:

```bash
bash singleFrame-outcars2nep-exyz.sh OUTCARs
# Or extract multiple frames:
bash multipleFrames-outcars2nep-exyz.sh OUTCARs
```

Each script deletes and recreates its output directory, so save previous results before rerunning. Use input paths without spaces.

The output contains energies, forces, cell vectors, and virials when enabled. The single-frame script also includes Born effective charges when present. The multi-frame script records rejected single-point OUTCAR paths in `non_converged_files.txt`, replacing any previous log.
