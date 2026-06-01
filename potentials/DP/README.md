# `DP` model for `GPUMD`

## Necessary instructions

- This is a test version.
- Supports DeePMD-kit potential files with both TensorFlow and PyTorch backends:
  - `.pb` files: TensorFlow backend (generated using `dp --tf freeze`)
  - `.pth` files: PyTorch backend (generated using `dp --pt freeze`)
  - `.pt` files: PyTorch AOTInductor backend (recommended for DPA4-Neo and other modern models)
- The DeePMD-kit C++ API (`deepmd::DeepPot`) automatically detects the backend from the file extension at runtime.

## Installation Dependencies

- You must ensure that the new version of DeePMD-kit (v3.x or later) is installed and can run normally. This program contains DeePMD-related dependencies.
- For PyTorch backend support, you also need libtorch installed and linked during compilation.
- The installation environment requirements of `GPUMD` itself must be met.

## Compilation

To enable DeePMD support, modify the `src/makefile`:

1. Add `-DUSE_DEEPMD` to `CFLAGS`
2. Add DeePMD-kit include and library paths
3. For PyTorch backend, also add libtorch library paths

See the commented section in `src/makefile` for detailed instructions.

**Note:** The old `USE_TENSORFLOW` flag is still supported for backward compatibility, but `USE_DEEPMD` is now the recommended flag name.

## Run Test

This `DP` interface requires two files: a setting file and a `DP` potential file. The first file is very simple and is used to inform `GPUMD` of the atom number and types. For example, the `dp_settings.txt` is shown here for use with the `potential dp_settings.txt DP_POTENTIAL_FILE` command in the `run.in` file:

```dp 2 O H```

The potential file can be:
- `model.pb` (TensorFlow backend)
- `model.pth` (PyTorch backend)
- `model.pt` (PyTorch AOTInductor backend)

The `dp_settings.txt` format is the same regardless of which backend you use.

## Notice

The type list in the setting file and potential file must be the same.

## References

- [Installation for `GPUMD-DP`](https://github.com/Kick-H/GPUMD/blob/master/doc/installation.rst)
- [Tutorials for `GPUMD-DP`](https://github.com/brucefan1983/GPUMD-Tutorials/tree/main/examples/14_DP)
- [`DeePMD-kit`](https://github.com/deepmodeling/deepmd-kit)
