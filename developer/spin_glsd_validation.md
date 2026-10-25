# 独立 GLSD：用法、物理约定与验证

## 怎么用

SIB 继续使用 `nve_sib / nvt_sib / npt_sib`，保持定模长。此前未发布的 `lsf` 开关已删除，包括 `lsf 0`；SIB 不接受 `mobility` 或 `damping`。

GLSD 独立使用以下命令，势函数和 `model.xyz` 的 `spin:R:3` 输入不变。**现用无量纲 `damping`。示例 `.01` 是试跑起点，不是材料标定值或程序默认值。原验证中的 mobility=10 是内部耗散系数，不能直接改名为 damping=10。**

```text
# 零温耗散、固定原子位置
ensemble nve_glsd damping .01 stemp -1
time_step 0.1
dump_xyz 100 spin.xyz precision double spin mforce
dump_restart 1000
run 1000

# 300 K 自旋热浴，固定晶格
ensemble nvt_glsd 300 300 100 lattice off damping .01 seed 731

# 原子 NHC 与自旋热浴耦合
ensemble nvt_glsd 300 300 100 damping .01 seed 731

# 原子、晶胞 MTTK 与自旋耦合
ensemble npt_glsd temp 300 300 iso 0 0 tperiod 100 pperiod 1000 damping .01 seed 731
```

`damping` 必填、有限且非负，定义为 `η=ħγ'_s`，无量纲，不是 Gilbert alpha，也不是论文定模长方程中的 γ_s。内部沿用 mobility `λ=η×gamma²×ħ`，单位 μB²/(eV ps)。`gamma=g_L/ħ` 默认约 3038.54；若显式修改 gamma，固定 damping 下的 λ 也会改变。使用既有常数 `ħ=.06465412×10.18051/1000 eV ps`。启动日志同时打印输入 damping 和换算后的 mobility。

GLSD 不再接受早期未发布的 `mobility`，也不接受 `alpha` 或 `lsf`；SIB 不接受 `damping`。`stemp -1` 关闭噪声，`stemp 0` 跟随系综温度，正值指定自旋浴温度。NVE 形式默认 `stemp -1` 且不允许 `stemp 0`。`seed` 默认 12345，必须为正整数。时间步输入单位 fs。

这里的 `nve_glsd` 表示固定晶格的驱动方式；有耗散时，自旋能量并不守恒。示例 damping 和时间步不是对 Fe 的拟合参数或通用稳定性保证。完整参数说明见 `doc/gpumd/input_parameters/ensemble_glsd.rst`。

## 用户如何选 damping

用户无需先扫描平衡磁矩或输入元素表；系数不依赖初始或瞬时模长。可以从 `damping .01` 开始试跑，并比较 `.005/.01/.02`，同时检查时间步偏差、相关时间和充分平衡后的统计。这是数值试跑建议，不是校准范围或置信区间；更大阻尼也不保证采样更好。

无量纲化只简化输入，没有新增物理项，也不会消除真实材料的耗散差异。势函数与 FDT 不会替用户确定动力学阻尼，因此没有默认材料值。以下讨论中的 mobility λ 是内部物理系数，旧输入转换为 `damping=λ/(gamma²ħ)`；默认 gamma 下是 `λħ/4`。原 λ=10 对应 damping≈.001645529788003，不是 .01。零阻尼同时关闭耗散和噪声。

- **只算平衡静态量**：对共同恒温热浴、可归一化且充分混合的系统，不同正的常数 mobility 对应相同的连续时间 Gibbs 平衡分布，但弛豫和相关时间不同。用短试跑检查能量、磁化及模长统计的平稳性与混合，缩小时间步并比较几组 mobility；平衡统计应在采样误差内一致。不同子系统浴温或外场驱动的非平衡状态不适用这个结论。
- **要预测真实时间过程**：需要相同材料、温度、磁态和热浴条件下的实验或独立动力学参考来标定，并报告参数来源和敏感性。不能因为某个值跑得稳定、温度正确，就认定退磁或弛豫时间正确；也不能简单增大它来加速真实动力学。
- **没有动力学参考**：可以开展平衡采样或参数敏感性研究，但不能仅凭 NEP 势给出有依据的绝对弛豫时间。内部系数换算值 `235.2` 也不是另一套 Fe 势的通用答案。

一个可解释的初估来自非零稳定磁矩附近、固定环境下的局部径向模态。若 `E(r)≈E(r0)+κ(r-r0)²/2`，则由当前方程推得 `dδr/dt=-λκδr`，因此：

\[
\lambda=\frac{1}{\kappa\tau_\parallel}.
\]

曲率 κ 的单位 eV/μB²，目标弛豫时间 τ 的单位 ps。**曲率可以从势得到，物理弛豫时间仍需外部依据。** 假设 κ=.4、τ=.5，则 mobility=5；这是算术示例，不是 Fe 数据，也不是耦合有限温度系统的通用拟合公式。若 τ 只是人为选择的采样时间，就应标注为数值设置。

已知 Gilbert alpha 也不能唯一确定完整变模长动力学。一个各向同性 mobility 同时限制纵横向弛豫；若两种实测弛豫率不能同时匹配，这是当前模型的适用限制，不能靠重命名或设置默认值消除。每个候选值都需单独做时间步收敛；径向谐振例中的 `λκΔt<2` 只给出显式 Heun 的局部稳定条件，准确性、进动和其它耦合模态可能要求更小时间步。

## 论文依据与符号映射

直接依据 [Ma and Dudarev, Phys. Rev. B 86, 054416 (2012)](https://doi.org/10.1103/PhysRevB.86.054416)，不是依据 LAMMPS 源码：

- 式 (18)–(19)：进动、完整矢量耗散、加性噪声。
- 式 (20)–(22)：横向投影与双叉乘的等价关系。
- 式 (33)：涨落耗散关系。

论文用角动量自旋 `s`，定义磁矩 `M = -g_L μB s`；GPUMD 存储数值磁矩 `S = M/μB = -g_L s`，磁力 `H = -∂E/∂S`。因此 `H_s = -g_L H`，转换后：

\[
 d\mathbf S=[-\gamma\mathbf S\times\mathbf H+\lambda\mathbf H]dt
 +\sqrt{2\lambda k_BT}\,d\mathbf W,
 \quad\gamma=g_L/\hbar,\quad\lambda=g_L^2\gamma'_s.
\]

时间以 ps 计，默认 `g_L=2`，新输入 `η=ħγ'_s`。对于有量纲的磁矩，耗散系数为 `(g_L μB)^2 γ'_s`。

**论文数值存在归一化歧义，不能把 235.2 当作确定的 Fe 标定值。** III.C 节打印的 `γ'_s=5.88×10^13 eV^-1 s^-1` 按上述坐标换算确实得到 `mobility 235.2`；但按式 (28) `γ'_s=s0²γ_s/ħ`，结合文中的 `M0=2.2 μB`、`s0=M0/(2 μB)=1.1` 和 `γ_s=.008`，得到的是 `γ'_s=1.47065×10^13`，对应 `mobility 58.826`。二者相差 4 倍，目前未由作者代码或勘误消解。原先“不能直接照抄 58.8”的表述不完整，已在这里更正：应明确采用哪一条推导，而不能只引用最后打印的数值。下方 235.2 测试只验证指定系数的坐标与时间单位转换，不验证它的实验标定。

`.008` 来自 [Ma、Dudarev、Woo，PRB 85, 184301 (2012)，III.B 节](https://doi.org/10.1103/PhysRevB.85.184301) 的 Fe 薄膜激光退磁拟合，其中同时处理电子热浴、晶格耦合和吸收脉冲能量。若明确把该值作为外部先验迁移，参考磁矩 m0 下的横向阻尼匹配给出 `λ=m0²×.008/ħ`、`η=(m0/g_L)²×.008`。这不是 NEP 势独立决定的耗散参数；得到的 λ 应保持常数，不能在运行中随瞬时模长更新，否则将改变噪声模型。针对当前 Fe 势的静态估计另见 `fe_glsd_mobility_estimate.md`。

当 `r=|S|>0`：

\[
 \lambda\mathbf H_\perp=-\frac{\lambda}{r^2}\mathbf S\times(\mathbf S\times\mathbf H),
 \qquad \dot r=\lambda\mathbf e\cdot\mathbf H.
\]

横向满足叉乘关系；一个各向同性 mobility 同时控制横纵向耗散。代码直接计算 `λH`，不除以模长。零磁矩可在外场或噪声作用下启动；零场零温保持为零。固定原子、静态保守势下，连续方程满足 `dE/dt = -λ Σ|H|² ≤ 0`；不要求有限时间步每一步严格降能。

噪声为 Cartesian 加性噪声，Ito 与 Stratonovich 一致，协方差为 `2λkBT dt I`。对足够约束磁矩的光滑势能，目标平衡分布为 `exp(-E/kBT) ∏d³S`；径向测度含 `r²dr`，不能换成平坦 `dr`。不引入额外径向熵力。

论文的示例 Heisenberg–Landau 哈密顿量和 Fe 系数没有硬编码；现有 Spin 势负责能量与导数。论文式 (34) 需要能量 Hessian 的自旋温度估计未实现，`thermo` 温度仍是原子动能温度。

## 数值方法和生命周期

独立 `Spin_GLSD` 使用 stochastic Heun。先保存起点和漂移，以同一个高斯增量预测；重算磁力后用起点、预测点漂移的平均值校正；随后重算终点的磁力、机械力、能量与 virial。初始求力后每步调用势函数两次。外场走原有 `Force` 接口，有状态的机械外力不重复抽样。

Heun 是本实现的离散选择，不声称复现论文的数值代码。确定性部分二阶，加性噪声强阶一；显式、条件稳定。纯进动会有有限步长模长漂移，精确定模长应用应选 SIB。谐振势径向稳定性条件 `λkΔt<2` 仅针对该模型，精度仍需时间步收敛。

SIB 的 Cayley predictor/corrector、阻尼、FDT 和零磁矩行为未改。两者共用从原 SIB 原样抽出的计数式 Gaussian helper；不复制磁力或势函数算法。公共 parser 只复用 `gamma/stemp/seed/lattice` 的解析与验证；GLSD 和 SIB 的耗散参数互斥。

GLSD 无独立惯性 `spin_vel`；进入时清零并取消初始化标记，请求 `spin_velocity` 输出会报错。restart 写入全精度磁矩及 `spin_glsd_step/seed`。保持原子顺序、种子、参数和势，可续接随机流；连续 GLSD run 块保持计数；选择其它积分器清零计数。

当前 `npt_glsd` 保存 MTTK 内部状态和温压 ramp 相位，详见 `mttk_restart_validation.md`；`nvt_glsd` 在启用原子运动时也保存独立 NHC 的链状态、温度 ramp 相位和全局时间，详见 `nhc_restart_validation.md`。下文早期验证记录中的 NVT/NPT 限制属于当时的基线版本。

## 原 mobility 接口的 Sai 构建与证据

以下原始数据对应作业 1892500，保留作为换算前基线，不将其改称新 damping 接口的运行结果。新版复验见文末。

2026-10-21，工作区 `abb9/GPUMD-MagTheoryLab`，分支 `spin3-refactor`，基于 HEAD `63e41d8e2c63ff9a5a4847e68031bf0dd6cd97e1` 加本轮未提交修改。未 stage、commit、push，也未修改 FIRE 文件。

复用 hpc-platform 的 Sai profile 与原 workspace plan，派生独立 `glsd-independent-v1` 运行目录。作业 **1892500**，最终 `COMPLETED`、退出码 `0:0`、运行 2 分 11 秒。

- 节点 `16v100n04`，分区 `16V100`，QoS `rush-gpu`。
- 分配 4 × Tesla V100-SXM2-16GB；单卡/4 卡测试在同一 allocation 内依次运行。
- CPU：AMD Ryzen Threadripper PRO 5995WX，64 核；OMP/OpenBLAS 线程均为 1。
- CUDA 12.9.1，driver 550.163.01；`make clean` 后 `make -j8 gpumd CUDA_ARCH=-arch=sm_70`。
- 重建前核验 **542** 个源码/测试/fixture 文件的 SHA-256，全部匹配；运行后再次核对本地源码/测试未变。
- 4 卡 Fe 日志明确打印 `spatial decomposition over 4 GPUs`，不是只暴露设备而实际跑单卡。
- 这是 correctness 验证，不做吞吐、负载隔离或性能改善结论。

SHA-256：

```text
当前 gpumd  9f244ff6bf48ac63df64cb24793dcea0e033e2aa7efcb026075db6b0b44169a9
旧版 6619b10 730401898400beb6e97866ca70505919686aa2adeade115eeb63274b9ac06889
Fe 势        38dabe33fd101432c54a453629c8fbbb8d8d7edbf437477180a09431915fb711
```

原 Fe 势 `$FE_MODEL` 保持只读，末尾重新核对哈希一致。测试使用其副本，旧验证结果保留。当前可执行文件已在本次 run 目录单独保存。

证据位置：

```text
远端根目录：$REMOTE_WORK_ROOT/gpumd-lsf-20261021/gpumd-spin3-spin3-refactor/glsd-independent-v1
  source.sha256 / source-verify.log / build.log / slurm-1892500.out
  gpumd / spin_glsd_analytic / analytic.json / runtime/report.json

本地小证据集：$VALIDATION_ROOT/experiments/gpumd-lsf-20261021/glsd-independent
  analytic.json / runtime.json / source.sha256 / source-verify.log
  slurm-1892500.out / build.log / collection_manifest.json

plan：$VALIDATION_ROOT/plans/sai-gpumd-lsf-20261021-final-v5-glsd-independent-v1.json
```

## 验证结果

独立解析/统计测试 **17** 项、全程序 **44** 个报告记录全部通过；后者包含 **27** 项非法输入拒绝检查。另运行原 `validate_spin_sib_runtime.py`，其解析 oracle、FDT、种子、12 项非法输入与 NPT smoke 全通过。

### 原 SIB 回归

与 `6619b10` 比较相同 Fe 模型、初态、种子和硬件配置：

| 模式 | 单卡最大磁矩分量差 / μB | 4 卡旧/新差 / μB | 4 卡旧/旧重复差 / μB |
|---|---:|---:|---:|
| 零温阻尼 | 0 | 2.15322e-9 | 2.15200e-9 |
| 300 K | 0 | 9.30037e-10 | 9.30037e-10 |

4 卡旧/新差处于旧版自身重复误差范围。原 SIB 单步 oracle 误差 4.962e-10、模长误差 5.865e-10、热噪声 FDT 误差 2.458e-9。`lsf 0/1` 均拒绝，SIB 不再提供变模长接口。

### 独立解析解、耗散与时间步

谐振哈密顿量 `E=0.4|S|²/2-0.2Sz`，`gamma=3`、`mobility=2`，初态 `(1,.2,-.3)`；直接用闭式指数衰减/旋转解验收，没有复制 Heun 更新公式作为 oracle。

| 检查 | 结果 |
|---|---|
| 5 ps 有阻尼外场，dt=.1/.05/.025 ps | 最大分量误差 1.54826e-4 / 3.79534e-5 / 9.38731e-6 μB |
| 同一测试能量 | 初始 .286 eV → -.0498873 eV，平衡极小值 -.05 eV |
| 无阻尼进动，三档步长 | 1.83547e-3 / 4.58648e-4 / 1.14613e-4 μB |
| 径向恢复 20 ps | 误差 1.93233e-11 μB，能量 2.86308e-15 eV |
| 零磁矩零场零温 | 严格保持零 |
| 全程序零磁矩 + 外场，1/4 卡 | 解析误差均为 1.90820e-17 μB |
| 指定系数的磁矩/时间单位映射，默认 gamma、mobility=235.2，dt=1/.5/.25 fs | 1.71272e-5 / 4.28750e-6 / 1.08630e-6 μB |
| 全程序横向叉乘外场解，dt=50/25/12.5 fs | 2.12616e-3 / 5.26087e-4 / 1.30781e-4 μB |
| 实际 ANN 径向响应，dt=20/10/5 fs，对独立高精度标量 RK4 | 2.93288e-6 / 7.27148e-7 / 1.78725e-7 μB |

误差随步长减半约缩小 4 倍。论文映射测试独立采用 `ℏ=6.582119569e-4 eV ps` 与 Zeeman 单位转换，保留仓库历史常数的微小舍入差。能量测试检验净耗散和趋近平衡，不把每个有限步长严格降能当作普遍条件。

### 有限温度与单卡/4 卡统计

谐振势每个温度/步长使用 131072 个独立样本，40 ps，检验 Cartesian 均值、方差 `kBT/k`、四阶矩和 Maxwell 径向 CDF。

- 100/400 K，dt=.8/.4/.05 ps：三分量平均方差相对解析值偏差分别为 **-12.9908% / -2.72009% / -0.193424%**。粗步长的明显偏差随缩小步长消失；这些数值同时含采样误差，不能全部解释为纯离散偏差。
- 400 K 的方差是 100 K 的 4 倍，符合温度依赖。
- 300 K + 外场：平衡 `Sz` 均值约 .500676 μB，解析值 .5；平均方差偏差 -0.203613%。
- 从零磁矩启动的有限时间 OU 方差也通过闭式解检查。
- 全程序自由扩散：三个种子 × 768 原子，共 2304 样本；1/4 卡的统计量相同。解析方差 .0206816232 μB²，实测三分量 .0201110740 / .0210062262 / .0208777042 μB²，满足预设采样置信门。零势扩散不是平衡分布测试。

随机数在 GPU 0 按原始原子编号、步数、分量和种子产生；势函数在多卡空间分解后按原子顺序回收结果，不给每张卡单独抽噪声。差异可来自邻居排序和浮点归约，因此不要求一般随机长轨迹逐位一致。

### 真实 Fe 势与耦合

768 原子，dt=.1 fs，mobility=10，seed=731，外场 `(7,-3,4) T`，默认 gamma；有限温度测试使用 300 K。下表是同一输入单卡/4 卡的最大分量差：

| 测试 | 时长 | 磁矩差 / μB | mforce 差 / (eV/μB) | 原子力差 / (eV/Å) |
|---|---:|---:|---:|---:|
| 零温固定晶格 | 20 fs | 9.00938e-7 | 4.45701e-7 | 8.60298e-7 |
| 300 K 固定晶格 | 20 fs | 1.16552e-6 | 4.80949e-7 | 8.65041e-7 |
| NVT 原子耦合 | 10 fs | 1.16418e-7 | 5.04252e-7 | 1.01284e-6 |
| NPT 原子/晶胞耦合 | 2 fs | 5.58797e-8 | 4.34786e-7 | 9.71421e-7 |

四类测试均在独立重新求力后核对终点能量、磁力、原子力和 virial，排除了输出仍使用 predictor 状态的问题。零温磁矩范围约 2.18028–2.63963 μB；300 K 约 1.98416–2.87578 μB。报告中的 wrapped position difference 包含周期性坐标回卷，不能作为真实位移或不稳定性的证据。

### Restart 与接口

- 固定晶格 toy 势，零温和 300 K，单卡和 4 卡：40 步连续运行对 20+20 restart，磁矩差均为 **0**；连续两段 run 与完整运行差也均为 **0**。
- GLSD → SIB → GLSD：旧计数被清除，新种子 913 可用，新计数为 1；单卡/4 卡均通过。
- 真实 Fe NVT/NPT restart 后各继续 20 步，单卡/4 卡均成功且输出有限；这是续跑可用性验收，不是 NHC/MTTK 内部状态精确恢复验收。
- 非法接口、缺失/重复 mobility、非有限值、alpha 冲突、lsf 旧开关、零/负/NaN 时间步、restart 缺失/不匹配/溢出及 spin_velocity 输出共 27 项，均明确报输入错误。

## 复现与适用边界

在分配的 GPU 节点可靠重建后：

```sh
nvcc -std=c++14 -O3 -arch=sm_70 -Isrc tests_pytest/spin_glsd_analytic.cu \
  src/integrate/spin_glsd.cu -o /path/to/spin_glsd_analytic
python3 tests_pytest/validate_spin_glsd_analytic.py /path/to/spin_glsd_analytic
GPUMD_COMMAND=/path/to/gpumd python3 tests_pytest/validate_spin_sib_runtime.py
python3 tests_pytest/validate_spin_glsd_runtime.py \
  --binary /path/to/gpumd --reference /path/to/gpumd-6619b10 \
  --mobility-reference /path/to/gpumd-before-damping \
  --model /path/to/readonly-Fe-nep.txt --output /path/to/new-result-directory
```

当前全程序脚本额外要求保存的旧 mobility 接口二进制，通过 `--mobility-reference` 提供；不能用新版替代该独立基线。

全程序 GLSD 脚本要求 allocation 含四张 GPU，会自行选择 1/4 张卡；输出目录必须不存在。实际远程提交继续经 hpc-platform plan/cycle，不能把示例当作绕过调度器的授权。

已验证的是上述方程、数值收敛、统计分布和生命周期。未声称材料特定纵横向弛豫率、量子统计、长期真实 Fe 热平衡或居里温度。有限 tanh 网络、单纯 Zeeman 能量不一定在无界磁矩空间给出可归一化 Gibbs 分布；真实 Fe 短跑不能替代约束势的理论模型测试。显式 Heun 的稳定性和训练范围仍是使用边界。本节初版的耦合 restart 限制随后对 MTTK NPT 作了补齐，见文末更新；独立 NHC 后续也已补齐，见 `nhc_restart_validation.md`。

## 无量纲 damping 接口复验

2026-10-21，将用户输入改为 `damping=η=ħγ'_s`，内部仍使用同一个 λ。换算采用当前 `gamma=g_L/ħ`，即 `λ=η×gamma²×ħ`；在所有参数解析结束后计算，因此 damping/gamma 顺序不影响结果。拒绝缺失、重复、负值、非有限值以及换算溢出/下溢；旧 mobility 关键字明确报错。零 damping 同时关闭耗散与噪声。

与上面已验证的 mobility 版本相比，源码/测试只有 `src/integrate/integrate.cu`、`src/integrate/integrate.cuh` 和 `tests_pytest/validate_spin_glsd_runtime.py` 改变。没有改积分器内核、磁力后端、随机数、SIB 或 restart 序列化。低层解析测试继续直接接收内部 λ，避免把输入换算混入其物理 oracle；全程序测试验证新输入及独立保存的旧二进制。

复用 Sai profile 和 workspace plan，独立运行目录 `glsd-damping-v1`。作业 **1892669**，节点 **16v100n05**，16V100 / rush-gpu，4 张 V100-SXM2-16GB，CUDA 12.9.1，OMP/OpenBLAS 线程 1。在同一 allocation 中核验 542 个源码/测试/fixture 哈希，`make clean` 后重新构建并验证；最终 COMPLETED、0:0、用时 3 分 58 秒。源码快照与运行后本地源码一致。原势和旧结果保持不变。

```text
新版 gpumd   51fa21e0a25e49529e5bb01ceb939170da08ee67525b496cd21310fe2df22e17
旧 mobility  9f244ff6bf48ac63df64cb24793dcea0e033e2aa7efcb026075db6b0b44169a9
SIB 基线     730401898400beb6e97866ca70505919686aa2adeade115eeb63274b9ac06889
Fe 势        38dabe33fd101432c54a453629c8fbbb8d8d7edbf437477180a09431915fb711
```

17 个解析案例、60 条全程序验收记录（其中一条汇总 35 项非法输入检查）及原有 SIB 独立脚本全部通过。旧/新对照使用相同 Fe 势、768 原子初态、种子 731、dt=.1 fs、外场 `(7,-3,4) T`、同一节点与 GPU 设置。覆盖零 damping、默认 gamma 下 damping=.01 的零温/300 K、自定义 gamma=3、两种关键字顺序、NVT 原子耦合和 NPT 晶胞耦合。旧程序输入等价 λ，新程序输入 η。

| 最大分量差 | 单卡旧/新 | 4 卡旧/新 | 4 卡旧/旧重复 |
|---|---:|---:|---:|
| spin (μB) | 1.66533e-16 | 3.26571e-7 | 3.23470e-7 |
| mforce (eV/μB) | 4.16334e-17 | 4.52126e-7 | 4.30266e-7 |
| 原子力 (eV/Å) | 6.66134e-16 | 5.58959e-7 | 6.81286e-7 |
| 每原子势能 (eV) | 0 | 1.66890e-6 | 1.43056e-6 |

4 卡差异与旧版重复运行的邻居排序/浮点归约差异同量级，不要求随机长轨迹逐位相同。本次短轨迹、匹配随机流的结果支持输入换算等价，不是通用材料动力学标定。

把同一个旧 checkpoint 分别交给旧、新程序继续运行，最大自旋差：单卡 4.44089e-16 μB，4 卡 1.46627e-7 μB；覆盖固定晶格、NVT、NPT。固定晶格的零温/300 K 全程与分段 restart 在 1/4 卡上的误差仍为 0。这个换算验证阶段尚未序列化 NHC/MTTK 内部状态，因此上述数据不声称耦合整条轨迹精确续接；后续 MTTK 更新见下文。

SIB 对 6619b10 的最大自旋差：单卡 8.67362e-19 μB，4 卡 2.31539e-9 μB。保留旧物理参数 λ=10 的 Fe 单卡/4 卡测试，最大自旋差分别为：零温 9.09790e-7、300 K 1.12950e-6、NVT 耦合 1.14323e-7、NPT 6.80926e-8 μB，均通过原有验收限。

证据保存在 `$VALIDATION_ROOT/experiments/gpumd-lsf-20261021/glsd-damping/`，包含 `runtime.json`、`analytic.json`、`summary.json`、源码哈希、构建日志、作业结果和精确收集清单。Plan 为 `$VALIDATION_ROOT/plans/sai-gpumd-lsf-20261021-final-v5-glsd-independent-v1-glsd-damping-v1.json`。远端目录位于原 software root 下的 `glsd-damping-v1`，保存独立新版二进制及全部案例输入/输出。未 stage、commit 或 push。

## 后续 MTTK NPT restart 验收（2026-10-21）

`npt_glsd` 现已保存并恢复 MTTK 恒温链、恒压器链、晶胞速率与质量、非静水参考晶胞、温压 ramp 相位和全局时间。Sai 作业 1892853 的 19 组连续／续跑对照及 10 项非法输入检查全部通过；真实 Fe 的 4 卡续跑最大位置差 3.32e-9 Å、自旋差 7.09e-8 μB。普通 MTTK NVT/NPH 共享入口、跨第 1000 步参考晶胞重置及缺失状态负对照也通过。完整方法、误差表、复现命令和证据见 [MTTK restart 验证](mttk_restart_validation.md)。上述历史报告中的 MTTK 限制由本次结果更新；这一 MTTK 验证阶段未包含独立 NHC；后续 NVT 实现与验收见 `nhc_restart_validation.md`。

## 后续 NVT NHC restart 验收（2026-10-21）

`nvt_glsd` 在启用原子运动时现已保存独立 NHC 的链质量、位置和共轭动量，以及温度 ramp 进度和全局时间。Sai 作业 1892883 的 12 组连续／续跑、6 组单卡／4 卡对照、13 项非法输入和 2 项 lattice-off 切换检查全部通过；真实 Fe 的 4 卡续跑最大位置差 1.27e-10 Å、自旋差 7.21e-8 μB。原有理论模型、SIB/GLSD 和 MTTK 回归也通过。使用说明、适用边界、构建来源和完整数据见 [NHC restart 验证](nhc_restart_validation.md)。此前 NHC 未保存状态的描述仅适用于历史二进制。

路径中的 `$VALIDATION_ROOT`、`$REMOTE_WORK_ROOT` 和 `$FE_MODEL` 是验证环境占位符；历史数值对应文中指定的源码、模型哈希和任务，不代表当前分支重测结果。可移植回归入口见 [Spin validation](spin_validation.md)。
