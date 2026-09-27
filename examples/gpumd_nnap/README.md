# NNAP potential

This example performs a one-step GPUMD calculation for Cu using the NNAP model in `../../potentials/nnap/Cu.json`.

This example requires an NNAP-enabled GPUMD build (`USE_NNAP` with Make or
`PKG_NNAP` with CMake) together with the required jse/JNI environment. See the
[installation guide](../../doc/installation.rst) for the build instructions.

Run:

```bash
/path/to/gpumd
```

The calculation writes energy, force, and virial information to `dump.1.xyz`.

`diffxyz.groovy` is an optional cross-check against the original Java/JSE NNAP implementation. It is kept because there is currently no equivalent Python reference implementation in this repository.
