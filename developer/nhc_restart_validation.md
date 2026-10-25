# NVT GLSD 的 NHC 状态续跑

## 怎么用

将新版 `restart.xyz` 复制到新的运行目录，改名为 `model.xyz`，保持势、原子顺序、质量、边界、时间步、外场、ensemble 参数和 seed 一致。不重新生成速度，不需要新开关。默认启用原子运动的例子：

```text
potential nep.txt
ensemble nvt_glsd 300 500 100 damping .01 seed 731
time_step .1
dump_restart 100
run 600
```

若这是原先 `run 1000` 在第 400 步写出的 checkpoint，续跑写剩余的 600 步，温度仍填写原来的 `300 500`；不要改成 checkpoint 时刻的目标温度。原协议结束后继续运行，目标温度保持终值。默认自旋浴温度跟随恢复后的 ramp，正的显式 `stemp` 仍是独立常温热浴。恒温运行可以直接延长。连续重启保留原协议长度与累计步数。

普通 `nvt_nhc` 复用相同能力。`lattice off` 没有原子 NHC，不写链状态；从带 NHC 的 checkpoint 改用 `lattice off` 或其它驱动时，明确提示并丢弃链状态。旧 restart 缺少 NHC 字段时仍可读取，但重新初始化恒温器。若有意启动新的热浴协议，删除 `nhc_state` 和 `nhc_time` 两个完整字段；不要修改 payload 数值。配置不匹配时程序报错，不静默重置。

## 保存及恢复约定

- `nhc_state` 为版本 1 的数值 CSV，目前有 28 个数：版本、累计步数、原协议长度；13 个配置值；4 节链各自的质量、位置、共轭动量。
- `vel_nhc1` 实际储存共轭动量。原 `nhc()` 以 `dt * vel / mass` 推进链位置，因此这里保存原始值，不作速度解释或单位换算。与 MTTK 不同，本 NHC 确实演化链位置，也完整保存。
- 原 NHC 的质量在运行初始化时根据起始温度和 `time_step * temperature_coupling` 建立，此后在 ramp 中保持不变。本次保留该约定，恢复保存的质量，不按重启处温度重算。
- `nhc_time` 为 GPUMD 内部时间；XYZ 轨迹仍按原约定输出 fs。链变量、原子状态均用 17 位有效数字保存；GLSD 沿用原有的磁矩、随机 seed 和计数器。
- 配置检查包括链长度、原子数、固定时间步、温度端点、耦合周期、固定／移动分组设置与移动速度；状态校验版本、长度、有限数、整数步数及正的链质量。一个 XYZ 不能同时携带 NHC 与 MTTK 状态。
- 状态写于完整步末端，先初始化必要存储再恢复链；积分前用累计步数恢复原子和自旋的目标温度。原 NHC 分裂公式、GLSD 更新、磁力后端及噪声生成均未修改。

`has_nhc_driver()` 区分真正的 NHC 对象和借用 `NVT_NHC` 类型标记的固定晶格驱动，避免 `lattice off` 下的错误类型转换。没有加载 checkpoint 时，多段 `run` 命令仍按原行为重新初始化原子恒温器。

## 验证方法与边界

`tests_pytest/validate_nhc_restart.py` 在单卡及 4 卡分别比较连续运行与独立进程续跑的每个时间步，覆盖真实 Fe 的零温自旋、300 K、250→450 K ramp、独立 200 K 自旋浴；另外使用零势加外场检查严格数值续接，使用 LJ fixture 检查普通 `nvt_nhc` 兼容性。LJ 是机械测试模型而非 Fe 势；4 卡可见时 LJ 仍在单卡执行。

检查位置、速度、自旋、力、磁力、能量、virial、晶胞与时间；额外比较恒温链质量、位置和动量。零势案例覆盖二次重启及超出原 ramp 长度的分段延长。旧版／新版未中断轨迹回归与旧版自身 4 卡重复误差分别验收；也直接比较新版的单卡／4 卡结果。移除 NHC 元数据但保留 GLSD 计数器是缺失状态负对照。非法配置／损坏状态必须明确拒绝。

这是 NVT NHC 状态续跑能力，不是所有 GPUMD action 的通用 checkpoint。`heat_nhc`、`heat_nhc_power`、SIB 随机流、随机机械外力、周期性速度重置、温度依赖势的独立协议、分析历史及 unwrapped image counters 不在本保证内；固定／移动分组的复杂组合也没有本轮实测覆盖。`lattice off` 不携带 NHC 温度协议，不能借此功能保证其温度 ramp 续接。保持模型与原子分组内容一致，使用固定时间步。真实 NEP 多卡归约与重建邻居顺序产生浮点差异，验收为短轨迹数值一致，不保证长期混沌轨迹逐位相同。

## 复现

在分配了 4 张 GPU、有 NumPy 的目标节点上运行，输出目录必须不存在：

```bash
python3 tests_pytest/validate_nhc_restart.py \
  --binary /path/to/new/gpumd \
  --reference /path/to/pre-nhc-restart/gpumd \
  --model /path/to/Fe/nep.txt \
  --output /path/to/new/nhc-check
```

## Sai 验收结果（2026-10-21）

作业 **1892883** 在 `16V100 / rush-gpu` 的 `16v100n06` 上完成，`COMPLETED / 0:0`，用时 6 分 59 秒（含构建与全部回归）。配置为 4 张 Tesla V100-SXM2-16GB、CUDA 12.9.1、驱动 550.163.01、sm_70；CPU 为 AMD Ryzen Threadripper PRO 5995WX，OMP/OpenBLAS 线程数均为 1。单卡和 4 卡案例在同一 allocation 中执行，GPU UUID 记录在日志；本次不作性能比较。

构建前校验 544 个源码／测试／fixture 文件，`make clean` 后重建。当前源码位于 `spin3-refactor`，HEAD 为 `63e41d8e2c63ff9a5a4847e68031bf0dd6cd97e1`，加此前 GLSD/MTTK 与本次 NHC 的未提交修改。收尾时 544 个文件仍与构建清单逐个匹配；原 Fe 势保持只读、哈希未变。没有 stage、commit 或 push。

**NHC 专项全部通过：12 组连续／续跑对照、6 组单卡／4 卡对照、13 项非法输入和 2 项 lattice-off 切换检查。** 另含两个卡数下的二次重启、ramp 延长、恒温链状态核对及缺失状态负对照。

初态为 768 原子，生成种子 813；GLSD seed 731、`damping .01`，时间步 0.1 fs。常规案例连续运行 80 步，在第 43 步的 checkpoint 后另起进程续跑 37 步，比较重启后的每一步。真实 Fe 的四类 NVT 案例最大绝对差：

| 量 | 单卡连续／续跑 | 4 卡连续／续跑 | 验收阈值 |
|---|---:|---:|---:|
| 位置 / Å | 1.21e-13 | 1.27e-10 | 2e-7 |
| 速度 / (Å/fs) | 1.63e-14 | 5.84e-11 | 2e-8 |
| 自旋 / μB | 1.11e-16 | 7.21e-8 | 3e-5 |
| 机械力 / (eV/Å) | 1.32e-10 | 9.66e-7 | 5e-5 |
| 磁力 / (eV/μB) | 2.78e-17 | 5.89e-7 | 5e-6 |
| 单原子能量 / eV | 0 | 2.15e-6 | 5e-6 |
| 单原子 virial / eV | 1.88e-10 | 1.60e-6 | 5e-5 |
| 晶胞 / Å | 0 | 0 | 2e-7 |
| 输出时间 / fs | 0 | 0 | 1e-7 |

- **恒温链本身**：300 K Fe 的质量／位置／动量原始存储量最大数值差，单卡为 1.78e-15、4 卡为 3.65e-10；零势为 2.66e-14。这里仅作序列化数值核对，不把不同量纲合并成物理误差。版本、配置、完成步数、原协议长度精确一致。
- **零势与多次重启**：单卡、4 卡的自旋差均为 0，位置差不超过 1.21e-13 Å。第 43 步继续 20 步后再次保存，再续跑 17 步，与连续轨迹一致；跨原 80 步协议终点延长至 90 步的不同分段结果也一致，最大位置差 7.82e-14 Å。满足所有分量 1e-9 的独立严格门限。
- **普通 NVT NHC**：LJ fixture 的续跑位置差至多 1.14e-13 Å、速度差 1.74e-18 Å/fs；力、能量、virial 差为 0。普通 NVT 的旧版／新版未中断轨迹一致。
- **旧版／新版前向回归**：Fe 单卡最大自旋差 4.45e-16 μB，4 卡 1.12e-7 μB；4 卡 ramp 的旧版自身重复差为 9.85e-8 μB。300 K 的那一次旧版重复接近机器精度，因此不把某一次重复误差视作所有配置的误差上限。全部按预先固定的 NEP 容差验收。
- **单卡／4 卡**：真实 Fe 全程最大位置差 6.50e-10 Å、自旋差 1.42e-7 μB、能量差 3.10e-6 eV/atom；晶胞和时间差为 0。GLSD 随机计数规则不变，浮点归约差异仍存在，不要求逐位一致。
- **缺失状态负对照**：只删除 NHC 字段，保留 GLSD 计数器和 seed，最大位置差 4.03e-5 Å、速度差 5.51e-5 Å/fs、自旋差 3.31e-5 μB。位置与速度明显超过验收阈值，说明测试能检出恒温链重新初始化。负对照也丢失了时钟，但判据只用动力学量。
- **非法输入**：时间步、耦合周期、温度端点不匹配；错误版本、截断、NaN、非正质量、错误原子数、非整数步数、重复时间、缺失时间、尾随逗号及 NHC/MTTK 状态冲突，共 13 项均明确拒绝。

同一新版二进制还通过 17 个 GLSD 理论模型案例、60 项原有运行检查（含 35 项非法输入）、SIB runtime 回归，以及 MTTK 的 19 组连续／续跑对照和 10 项非法输入检查。这些真实势 NHC 测试是 8 fs 的短轨迹验收，延长测试到 9 fs；不将其解释为长期热平衡或长期轨迹逐位复现。

### 构建与证据

| 对象 | SHA-256 |
|---|---|
| 新版 gpumd | `7ddb9d0739b92ee216729999e3684c0a8b19b82429a2dacaf79fde0312729326` |
| NHC 改动前 gpumd（已有 MTTK restart） | `ddb861ca001ca05e285382c7e720515238241a95a8d67679427392345bc241fa` |
| 只读 Fe 势 | `38dabe33fd101432c54a453629c8fbbb8d8d7edbf437477180a09431915fb711` |

远程隔离运行目录：`$REMOTE_WORK_ROOT/gpumd-lsf-20261021/gpumd-spin3-spin3-refactor/nhc-restart-v1`。该目录保留独立二进制、构建日志、输入、轨迹及所有报告；旧验证目录未覆盖。

本地证据：`$VALIDATION_ROOT/experiments/gpumd-lsf-20261021/nhc-restart-v1`。其中 `nhc-check/report.json` 是本次专项结果，`restart-check/report.json` 是 MTTK 回归；还保存 `plan.json`、`job_result.json`、构建／运行／Slurm 日志、`source.sha256`、`source-snapshot.tar.gz` 和带哈希的 `collection_manifest.json`。源码归档只包含清单中的源码／测试及有意 fixture，不包含真实 Fe 原始数据或运行轨迹。

路径中的 `$VALIDATION_ROOT`、`$REMOTE_WORK_ROOT` 和 `$FE_MODEL` 是验证环境占位符；历史数值对应文中指定的源码、模型哈希和任务，不代表当前分支重测结果。可移植回归入口见 [Spin validation](spin_validation.md)。
