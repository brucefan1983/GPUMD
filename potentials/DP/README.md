# DeePMD-kit interface

Build GPUMD with `USE_DEEPMD` and the DeePMD-kit C++ libraries as described in the
[installation guide](../../doc/installation.rst#dp-potential-support).
Model formats (`.pb`, `.pth`, or `.pt2`) depend on the installed DeePMD-kit backend.

Create a setting file, for example `dp.txt`:

```text
dp 2 O H
```

Here, `2` is the number of atom types. The element order must match the model's type map.
Select the setting file and model in `run.in`:

```text
potential dp.txt frozen_model.pb
```

Use the appropriate model filename for your backend.
See the [PyTorch example](../../examples/gpumd_dp_pytorch/README.md),
[DP tutorial](https://github.com/brucefan1983/GPUMD-Tutorials/tree/main/examples/14_DP),
and [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/stable/).
