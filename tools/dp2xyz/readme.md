# Add a dp training set that supports converting mixed types


## Converting `npy/raw` format

Sample code for converting `npy/raw` format:
```
python dp2xyz-raw-npy-mix.py deepmd-npy nepxyz-from-npy
```
Then you will find the converted `extxyz` training set in `nepxyz-from-npy` and be supported by `nep`.


## Converting `npy/raw/mixed` format

Sample code for converting `npy/raw/mixed` format:
```
python dp2xyz-raw-npy-mix.py deepmd-mixed nepxyz-from-mixed
```
Then you will find the converted `extxyz` training set in `nepxyz-from-mixed` and be supported by `nep`.


## Reference:

- [dp/npy/mixed format](https://docs.deepmodeling.com/projects/dpdata/en/master/systems/mixed.html)

- [Sample training set](https://github.com/deepmodeling/AIS-Square/tree/main/datasets)
