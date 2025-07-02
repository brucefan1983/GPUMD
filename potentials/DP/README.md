# `DP` model for `GPUMD`

## Necessary instructions

- This is a test version.
- Only potential function files ending with `.pb` in deepmd are supported, that is, the potential function files of the tensorflow version generated using `dp --tf` freeze.

## Installation Dependencies

- You must ensure that the new version of `DP` is installed and can run normally. This program contains `DP`-related dependencies.
- The installation environment requirements of `GPUMD` itself must be met.

## Run Test

This `DP` interface requires two files: a setting file and a `DP` potential file. The first file is very simple and is used to inform `GPUMD` of the atom number and types. For example, the `dp.txt` is shown in here for use the `potential dp.txt DP_POTENTIAL_FILE.pb` command in the `run.in` file:

```dp 2 O H```

## Notice

The type list of setting file and potential file must be the same.

## References

- [Installation for `GPUMD-DP`](https://github.com/Kick-H/GPUMD/blob/master/doc/installation.rst)
- [Tutorials for `GPUMD-DP`](https://github.com/brucefan1983/GPUMD-Tutorials/tree/main/examples/14_DP)
- [`DeePMD-kit`](https://github.com/deepmodeling/deepmd-kit)
