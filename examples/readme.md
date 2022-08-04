# How to run the examples?

* First, compile the correct version you need:
  * If you want to use empirical potentials, do not add `-DUSE_FCP` or `-DUSE_NEP` to `CFLAGS` in the `makefile`.
  * If you want to use an FCP potential, add `-DUSE_FCP` to `CFLAGS` in the `makefile`.
  * If you want to use an NEP potential, add `-DUSE_NEP` to `CFLAGS` in the `makefile`.
  * It is not allowed to add `-DUSE_FCP` and `-DUSE_NEP` simultaneously to `CFLAGS`.

* Then, go to the directory of an example and type one of the following commands:
  * `path/to/gpumd`
  * `path/to/nep`
