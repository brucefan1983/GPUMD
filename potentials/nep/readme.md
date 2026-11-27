# NEP models

Published models and datasets are maintained in [nep-data](https://gitlab.com/brucefan1983/nep-data).
The files here are provided for examples and tutorials.

| Model files | Scope | References |
| --- | --- | --- |
| [Si_2022_NEP4_3body.txt](Si_2022_NEP4_3body.txt), [Si_2022_NEP4_4body.txt](Si_2022_NEP4_4body.txt), [Si_2022_NEP4_5body.txt](Si_2022_NEP4_5body.txt) | Silicon; three-, four-, and five-body descriptors. | Data: [Bartók et al. (2018)](https://doi.org/10.1103/PhysRevX.8.041048). Model: [Fan et al. (2022)](https://doi.org/10.1063/5.0106617). |
| [C_2022_NEP4.txt](C_2022_NEP4.txt) | Mainly amorphous carbon. | Data: [Deringer et al. (2017)](https://doi.org/10.1103/PhysRevB.95.094203). Model: [Fan et al. (2022)](https://doi.org/10.1063/5.0106617). |
| [C_2024_NEP4.txt](C_2024_NEP4.txt) | General-purpose carbon model. | Data: [Rowe et al. (2020)](https://doi.org/10.1063/5.0005084). Model: [Fan et al. (2024)](https://doi.org/10.1088/1361-648X/ad31c2). |
| [Song-2024-UNEP-v1-AgAlAuCrCuMgMoNiPbPdPtTaTiVWZr.txt](Song-2024-UNEP-v1-AgAlAuCrCuMgMoNiPbPdPtTaTiVWZr.txt) | 16 elemental metals and their alloys. | [Song et al. (2024)](https://doi.org/10.1038/s41467-024-54554-x). |
| [NEP89 (20250409)](nep89_20250409/) | Inorganic and organic materials across 89 elements. | [Liang et al. (2026)](https://doi.org/10.1038/s43588-026-01009-6). |

See the [model format](nep_model_format.md) and [legacy model converter](nep_model_converter.py).
For supported conversion types and restart options, run from this directory:

```bash
python3 nep_model_converter.py --help
```
