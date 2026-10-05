# Request for consent: relicense GPUMD source code under LGPL-3.0-or-later

We propose to relicense the **source code and header files under `src/`, including subdirectories**, from GPL to [LGPL-3.0-or-later](https://www.gnu.org/licenses/lgpl-3.0.html). Existing third-party licenses and notices will be preserved.

This change aims to make GPUMD easier to integrate with other software through the LGPL’s more flexible linking terms. GPUMD will remain free and open source, and contributors will continue to receive credit for their work.

This proposal is based on GPUMD 5.9, commit [`eb7fbbf17a78`](https://github.com/brucefan1983/GPUMD/commit/eb7fbbf17a7886a014584aa0105f4456a514a11c).

GPUMD 5.9 has been released under GPL. We plan to make this change in 5.9.1 after obtaining the necessary permissions.

Please reply in this issue with the following statement if you agree:

> I agree to license all of my contributions to GPUMD’s source code and header files under `src/` (including subdirectories), as present in the revision linked above, under the GNU Lesser General Public License, version 3 or any later version (LGPL-3.0-or-later). I confirm that I hold the relevant copyright or am authorized by its holder to grant this permission.

Please also let us know about any missing contributors or corrections to the table below.

Thank you for your contributions to GPUMD!

## Contributors

Contributors are listed alphabetically by surname, with representative examples of their work.

| Contributor | GitHub | Representative contributions | Consent |
| --- | --- | --- | --- |
| Hekai Bu（卜河凯） | @BBBuZHIDAO | Hybrid ILP potentials; spring forces; GPU parallelization of PIMD; NEP optimization, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5967902451) |
| Hongjian Chen（陈洪剑） | @hitergelei | Angular-dependent potential (ADP); HIP/PPPM FFT compatibility. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969793225) |
| Yixin Deng（邓奕鑫） | @YixinDeng | NEP and electronic-stopping fixes | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968739215) |
| Haikuan Dong（董海宽） | @hailan2005 | HIP synchronization and missing-header fixes. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968125519) |
| Zaixu Duan（段在旭） | @duanzaixu | Four-body NEP descriptor extensions (q123, q233, and q134). | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968414082) |
| Paul Erhart | @erhart1 | Energy-difference training loss; training restarts and checkpoint controls; NetCDF and XYZ output enhancements, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5967898845)  |
| Zheyong Fan（樊哲勇） | @brucefan1983 | Overall GPUMD design and development; core algorithms and GPU implementation; NEP methodology and development; thermal-transport methods, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5967862633) |
| Alexander J. Gabourie | @AlexGabourie | GKMA and HNEMA; Tersoff 1988; NetCDF output, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5988398008) |
| Ming-Yu Guo（郭铭禹） | @SchrodingersCattt | Deep potential interface enhancements | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968893763) |
| Tobias Hainer | @tobias-hainer | Liquid thermodynamic integration using the Uhlenbeck–Ford reference model. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5989853788) |
| Daniel Hedman | @Dankomaister | Per-atom uncertainty output for active learning. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5989562447) |
| Hongfu Huang（黄宏富） | @hfood02 | GNEP | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969400279) |
| Wenwu Jiang（姜文武） | @wenwu96 | SW/ILP model-name correction. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969249536) |
| Denan Li（李德楠） | @MoseyQAQ | NEP small-box buffer reuse; Born effective charge training-reference alignment; MTTK parameter handling. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5967915504) |
| Qing'an Li（李庆安） | @liqa1024 | NNAP interface enhancements | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968190696) |
| Yanlong Li（李延龙） | @DragonPara | Missing `<numeric>` include in NEP training source. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5983104943) |
| Yifan Li（李一帆） | @Yi-FanLi | Atomic virial, dipole, and polarizability fitting. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969878221) |
| Ting Liang（梁挺） | @Tingliangstu | NetCDF enhancements; multigroup SHC, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5967938870) |
| Zhixin Liang（梁智新） | @liangzhixin-202169 | HNEMDEC | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969892634) |
| Yingqin Lin（林应钦） | @cmdhwz | PPPM k=0 and per-atom virial corrections. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968859267) |
| Eric Lindgren | @elindgren | Multi-NEP ensemble predictions; active-learning uncertainty; dipole and polarizability prediction/output, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5990222928) |
| Jiahui Liu（柳佳晖） | @Jonsnow-willow | NEP–ZBL integration and extensions; electronic stopping; training-virial corrections, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968748543) |
| Wenhao Luo（罗文浩） | @luowh35 | REMD and PRD; quantum thermal bath; two-temperature model; spatially resolved compute_chunk, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5967965240) |
| Shuning Pan（潘书宁） | @psn417 | FIRE minimization; MTTK and shock ensembles; nonequilibrium thermodynamic integration and scaling methods; extrapolation grade, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969765755) |
| Vivekkumar Panneerselvam | @Astro093 | Hybrid Nosé–Hoover/Langevin thermostat with multiple reservoirs. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5976124001) |
| Jiuyang Shi（施九洋） | @XIX-YANG | Electron-temperature-dependent NEP and initialization fixes. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968724923) |
| Lucas Svensson | @lucassven | Fixes to separators between output fields. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5989488549) |
| Benrui Tang（唐本瑞） | @tang070205 | Atom deposition; ionic conductivity; cohesive energy and elastic constants; phonon calculation enhancements, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968101824) |
| Yong Wang（王勇） | @yongwangxx | Multi-GPU NEP training; RDF; qNEP charge-neutrality chain-rule correction. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5988365904) |
| Yongchao Wu（吴永超） | @mushroomfire | Variable-cell FIRE; angular distribution function; EAM/alloy; Steinhardt bond-orientational order, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5967912525) |
| Bin Xu（徐斌） | @Binxu-Stack | Angularly resolved radial distribution function. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5967967307) |
| Ke Xu（徐克） | @Kick-H | Deep Potential and NNAP interfaces; constant-power NEMD; multigroup SHC, etc. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969687417) |
| Nan Xu（许楠） | @tamaswells | Stress/virial consistency fixes for dipole and polarizability prediction. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968942948) |
| Penghua Ying（应鹏华） | @hityingph | NEP training/output, preprocessor, and header fixes. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969306395) |
| Qi You（由琪） | @YouQixiaowu | Increase to the supported EAM element-count limit. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968681987) |
| Qinghan Yu（于清涵） | @Ternity | PLUMED build/API compatibility fixes. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5969741657) |
| Zezhu Zeng（曾泽柱） | @ZengZezhu | Eco-PIMD. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5972494515) |
| Bohan Zhang（张博涵） | @BohanZhang2002 | HIP multi-GPU synchronization fix. | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5968299686) |
| Jintu Zhang（张锦途） | @initqp @jintuzhang | PLUMED interface | [Agreed](https://github.com/brucefan1983/GPUMD/issues/1808#issuecomment-5991840580) |
