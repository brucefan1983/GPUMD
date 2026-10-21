# MTTK NPT 的完整积分器状态续跑

## 怎么继续运行

把新版 `restart.xyz` 复制到新目录并命名为 `model.xyz`，使用相同势、原子顺序、时间步、外场、ensemble 参数和 GLSD seed，不重新生成速度。恒温恒压例子：

```text
potential nep.txt
ensemble npt_glsd temp 300 300 iso 0 0 tperiod 100 pperiod 1000 damping .01 seed 731
time_step .1
dump_restart 100
run 600
```

如果原来是温压 ramp，则重复原始的起点和终点，不把起点改为 checkpoint 的温度或压力。checkpoint 保存原 `run` 的长度与已完成步数。例如原协议 `temp 300 500 iso 0 1`、`run 1000` 在第 400 步中断，续跑仍写相同的 `temp 300 500 iso 0 1`，再 `run 600`。跨过原协议终点后保持最终目标值。没有加载 checkpoint 时，多段 `run` 的原有重初始化行为不变。

旧 restart 没有 MTTK 状态，仍能读取，但只能重新初始化热浴和恒压器。改变为非 MTTK ensemble 时会打印提示并丢弃 MTTK 状态。若有意启动新协议，可删除头部 `mttk_state` 和 `mttk_time` 两个字段；这不再是连续轨迹续接。不要编辑 payload 内部数值。

## 保存了什么

- 原子恒温链：Q、eta_dot（包括末端哨兵）和 eta_dotdot。
- 恒压器热浴链：Q_p、eta_p_dot（包括哨兵）和 eta_p_dotdot。
- 晶胞速率 omega_dot、晶胞质量 omega_mass、非静水参考逆晶胞 h_ref_inv 及 vol_ref。
- 当前/目标温度、恒压器参考温度、温度缓存标记、速度缩放因子、压力张量、静水目标及 sigma。
- 原始 ramp 长度、完成步数、全局时间。非静水参考晶胞每 1000 步重置的相位随之恢复。
- GLSD 原有的自旋向量、随机计数器及 seed。

MTTK 当前实现不演化独立的链位置坐标，因此没有另造一组未使用的坐标。晶胞本身保存在 XYZ 的 Lattice；矩阵临时工作区、频率及子步长度由原配置重建。恢复之前校验版本、数组长度、有限值、整数计数、时间步、原子数、链长度、温压端点、周期和耦合方式。配置不匹配时明确报错。

状态在完整积分步结束时写出，使用版本 1 的数值 CSV 元数据 `mttk_state` 和内部时间单位的 `mttk_time`；浮点值采用 17 位有效数字。MTTK 初始化先重建必要存储，再恢复状态；GLSD 的自旋浴温度使用恢复后的协议相位。磁力、势函数、MTTK 推进公式和噪声公式未改。

## 验证设计

`tests_pytest/validate_mttk_restart.py` 比较独立进程的连续运行与重启续跑，检查重启后**每一步**的原子位置、速度、自旋、晶胞、时间、能量、机械力、磁力和 virial。测试固定模型、初态、种子和硬件，单卡与 4 卡分别验收。

真实 Fe 势覆盖零温自旋耗散、300 K、温压 ramp、tri 晶胞和非静水压力。共享入口 `npt_mttk`、`nvt_mttk` 和 `nph_mttk` 使用独立 LJ fixture（ε = 0.01 eV，σ = 2.2 Å，截断 4.5 Å）检查机械积分兼容性；它不是 Fe 的物理模型，且 LJ 后端仍在单卡执行，4 卡可见性不代表 LJ 多卡加速。零势 fixture 检查低噪声数值续接；跨第 1000 步的非静水案例检查参考晶胞重置相位。保存的旧二进制用于未中断轨迹回归，4 卡还测旧版自身重复运行误差。

负对照只移除 MTTK 元数据而保留 GLSD 随机计数器，检查重新初始化热浴/恒压器确实使轨迹明显偏离。另有参数不匹配、未知版本、截断、非有限数、重复字段、缺失时间及尾随逗号等拒绝测试。

## 适用边界

本改动补齐 MTTK 积分器状态，重点验收 `npt_glsd`，不是整个 GPUMD 所有 action 的通用 checkpoint。独立 NHC 驱动的 `nvt_glsd`、SIB 的随机流、随机机械外力、周期性速度重置、温度依赖 NEP 自己的协议、测量历史和 unwrapped image counters 不在本保证内。使用固定时间步，并保持势、原子顺序、质量、边界、外场和自旋参数一致。

重新建邻居列表及多 GPU 归约存在浮点差异，因此真实 NEP 的续跑按数值容差验收，不要求长期混沌或随机轨迹逐位相同。解析/零势测试用于区分状态遗漏与后端浮点误差。

## Sai 验收结果（2026-10-21）

源码位于 `spin3-refactor`，基底 HEAD 为 `63e41d8e2c63ff9a5a4847e68031bf0dd6cd97e1`，包含未提交的 GLSD/damping 和本次 restart 修改。Sai 使用既有 profile/plan，在 `16V100` 分区、`rush-gpu` QOS 的 `16v100n11` 上运行；4 张 Tesla V100-SXM2-16GB，CUDA 12.9.1，驱动 550.163.01，CPU 为 AMD Ryzen Threadripper PRO 5995WX。`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`，CUDA 目标为 sm_70。这里验收数值一致性，不作性能结论。

作业 `1892767` 完成干净重建及原有回归：17 个 GLSD 理论模型案例、60 项运行检查（含 35 项非法输入）和 SIB runtime 验证均通过。该作业随后在新测试把 NEP_Spin 误用于普通 MTTK 入口时退出；保持生产代码的限制，改为上述 LJ fixture 后重跑 restart 专项。作业 `1892853` 复用同一已核验二进制，最终 `COMPLETED / 0:0`，19 组连续／续跑对照及 10 项非法输入检查全部通过。中间一次脚本隔离目录漏带 fixture 的启动失败也保留在 `mttk-restart-v3`，未覆盖旧结果。

普通案例采用 768 原子、初态生成种子 813、GLSD seed 731、`damping .01`、时间步 0.1 fs；运行 80 步，第 43 步的 checkpoint 用独立进程继续 37 步。对比重启后每一步，而非只看终点。真实 Fe 势的五类案例（iso0、isoT、ramp、tri、deviatoric）最大绝对差如下：

| 量 | 单卡连续／续跑 | 4 卡连续／续跑 | 验收阈值 |
|---|---:|---:|---:|
| 位置 / Å | 2.13e-13 | 3.32e-9 | 2e-7 |
| 速度 / (Å/fs) | 6.40e-16 | 5.38e-11 | 2e-8 |
| 自旋 / μB | 3.99e-11 | 7.09e-8 | 3e-5 |
| 晶胞矩阵 / Å | 6.54e-15 | 3.27e-9 | 2e-7 |
| 机械力 / (eV/Å) | 2.88e-11 | 9.70e-7 | 5e-5 |
| 磁力 / (eV/μB) | 1.16e-10 | 6.27e-7 | 5e-6 |
| 单原子能量 / eV | 4.55e-13 | 2.15e-6 | 5e-6 |
| 单原子 virial / eV | 4.09e-11 | 1.78e-6 | 5e-5 |
| 输出时间 / fs | 0 | 0 | 1e-7 |

- 零势 fixture（仍有指定外磁场）在单卡和 4 卡的最大位置差均为 1.78e-14 Å，自旋及晶胞差为 0；非静水案例在第 997 步重启至 1007 步，跨参考晶胞重置点后最大位置差 9.24e-14 Å，自旋及晶胞差为 0。所有量满足独立的 1e-9 严格门限。
- LJ 的 `npt_mttk`、`nvt_mttk`、`nph_mttk` 续跑最大位置差 1.57e-13 Å，速度差 8.68e-19 Å/fs，力、能量、virial 和晶胞差为 0。普通 NPT 的旧版／新版未中断轨迹完全一致。
- Fe 恒温恒压及 ramp 的单卡旧版／新版自旋差至多 4.45e-16 μB。4 卡自旋差至多 6.76e-8 μB；旧版自身重复运行差至多 1.18e-7 μB。续跑误差与该归约误差量级相当。
- 同一新版二进制的原有 NPT 单卡／4 卡对照最大位置差 1.36e-9 Å、自旋差 4.98e-8 μB；这与每个卡数下的连续／续跑验收是两项不同检查。
- 负对照只删除 MTTK 元数据，保留 GLSD seed/counter：最大位置差 0.07336 Å、速度差 3.67e-5 Å/fs、自旋差 0.00754 μB，明显越过验收阈值。完整恢复的对应单卡案例位置差仅 4.44e-16 Å。
- 时间步、温控周期、压力端点、耦合方式不匹配，以及错误版本、截断、NaN、重复时间、缺失时间、尾随逗号均明确拒绝。

这些是 8 fs 的真实势短轨迹续接和 100.7 fs 的零势相位测试，不代表长期轨迹逐位复现或新的热力学精度结论。所有案例的全局时间、完成步数、原协议长度均恢复；GLSD 案例还检查随机计数器。

### 复现与证据

在有 4 张分配 GPU、NumPy 和已构建二进制的节点上，从仓库使用：

```bash
python3 tests_pytest/validate_mttk_restart.py \
  --binary /path/to/new/gpumd \
  --reference /path/to/pre-restart/gpumd \
  --model /path/to/Fe/nep.txt \
  --output /path/to/new/restart-check
```

`--output` 必须是新目录。`--reference` 使用本次 MTTK 改动前、已支持无量纲 damping 的二进制。

SHA-256：

| 对象 | SHA-256 |
|---|---|
| 新版 gpumd | `ddb861ca001ca05e285382c7e720515238241a95a8d67679427392345bc241fa` |
| 旧版 damping gpumd | `51fa21e0a25e49529e5bb01ceb939170da08ee67525b496cd21310fe2df22e17` |
| 只读 Fe 势 | `38dabe33fd101432c54a453629c8fbbb8d8d7edbf437477180a09431915fb711` |

远程根目录为 `/org/caep-xuben/chenchengbing/hpc-work/gpumd-lsf-20261021/gpumd-spin3-spin3-refactor`。`mttk-restart-v2` 保存构建日志、543 文件源码校验记录、原有回归和新二进制；`mttk-restart-v4/restart-check/report.json` 是最终专项报告。该次验收结束时本地 468 个 `src/` 文件与构建快照全部匹配，原 Fe 势 SHA-256 未变。

本地最小证据集在 `/Users/superbing/hpc-platforms/experiments/gpumd-lsf-20261021/mttk-restart-v2` 与 `mttk-restart-v4`，各含 `collection_manifest.json`。后者含最终 plan、job 脚本、测试脚本、Slurm 日志和报告；完整输入与轨迹保留在各远程隔离目录。未 commit 或 push。

后续 NVT NHC 的状态续跑已独立实现，见 [NHC restart 验证](nhc_restart_validation.md)。本报告的二进制和数据仍指上述 MTTK 验证快照。
