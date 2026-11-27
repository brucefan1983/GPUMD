# EAM potential files

| File | Contents |
| --- | --- |
| [Cu_Zhou_2004.txt](Cu_Zhou_2004.txt) | Cu potential in the analytical Zhou form. |
| [Cu_Dai_2006.txt](Cu_Dai_2006.txt) | Cu potential in the analytical Dai form. |
| [eam_zhou_2004_Cu.txt](eam_zhou_2004_Cu.txt) | Cu potential in the analytical Zhou form. |
| [eam_zhou_2004_CuFeNi.txt](eam_zhou_2004_CuFeNi.txt) | Cu-Fe-Ni potential in the analytical Zhou form. |
| [eam_zhou_2004_AlCoCuFeNi.txt](eam_zhou_2004_AlCoCuFeNi.txt) | Al-Co-Cu-Fe-Ni potential in the analytical Zhou form. |
| [eam_zhou_2004_CoCrNi.txt](eam_zhou_2004_CoCrNi.txt) | Co-Cr-Ni potential in the analytical Zhou form. |
| [eam_zhou_2004_18Elements.txt](eam_zhou_2004_18Elements.txt) | Element-labeled parameter table for preparing Zhou-form potential files. |
| [eam_zhou_2004_18Elements_unified.txt](eam_zhou_2004_18Elements_unified.txt) | 18-element Zhou-form potential file with a GPUMD header. |
| [CuNi.eam.alloy](CuNi.eam.alloy) | Tabulated Cu-Ni potential in `eam/alloy` format, based on Zhou parameters. |

The element-labeled parameter table is reference data; use a potential file with the appropriate header in `run.in`. See the [EAM manual](../../doc/potentials/eam.rst) for file formats and usage.

## References

- Zhou form and parameters: X. W. Zhou et al., Phys. Rev. B **69**, 144113 (2004). [DOI](https://doi.org/10.1103/PhysRevB.69.144113).
- Dai form and parameters: X. D. Dai et al., Journal of Physics: Condensed Matter **18**, 4527 (2006). [DOI](https://doi.org/10.1088/0953-8984/18/19/008).
- Cr parameters: G. Anand, R. Goodall, and C. L. Freeman, Scripta Materialia **124**, 90–94 (2016). [DOI](https://doi.org/10.1016/j.scriptamat.2016.07.001).
- V parameters: S. Zhao et al., Acta Materialia **219**, 117233 (2021). [DOI](https://doi.org/10.1016/j.actamat.2021.117233).
