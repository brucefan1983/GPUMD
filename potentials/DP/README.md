# `DP` model for `GPUMD`

## Necessary instructions

- This is a test version.
- DeePMD-kit model formats are supported through the DeePMD-kit backend enabled at compile time.
- TensorFlow `.pb` models and PyTorch `.pth`/`.pt2` models depend on the installed DeePMD-kit version and build options. See the PyTorch example in `examples/gpumd_dp_pytorch` for DPA2/DPA3/DPA4 usage.

## Installation Dependencies

- You must ensure that the new version of `DP` is installed and can run normally. This program contains `DP`-related dependencies.
- The installation environment requirements of `GPUMD` itself must be met.

## Run Test

This `DP` interface requires two files: a setting file and a `DP` potential file. The first file is very simple and is used to inform `GPUMD` of the atom number and types. For example, the `dp.txt` is shown in here for use the `potential dp.txt DP_POTENTIAL_FILE` command in the `run.in` file:

```dp 2 O H```

## Notice

The type list of setting file and potential file must be the same.

## References

- [Installation for `GPUMD-DP`](https://github.com/Kick-H/GPUMD/blob/master/doc/installation.rst)
- [Tutorials for `GPUMD-DP`](https://github.com/brucefan1983/GPUMD-Tutorials/tree/main/examples/14_DP)
- [`DeePMD-kit`](https://github.com/deepmodeling/deepmd-kit)
