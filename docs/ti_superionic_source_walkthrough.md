# GPUMD 超离子两步热力学积分新增源码逐段解读

> 代码范围：  
> `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cuh`  
> `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cu`  
> `GPUMD-dev-v4.9.1/src/integrate/uf_reference.cuh`
>
> 目标：解释新增 `ti_superionic_stage1` 和 `ti_superionic_stage2` 系综如何用 Einstein 晶体与 UF 流体参考态，对任意按元素指定的超离子体系计算自由能积分。

---

## 1. 物理模型总览

### 1.1 任意超离子体系的角色划分

新增代码不把体系写死为 H2O 或 AlOOH，而是通过命令中的元素符号决定参考态：

```text
spring <element> <k> ...
uf <element_i> <element_j> <p> <sigma>
```

被 `spring` 指定的元素进入 Einstein 晶体参考态；被 `uf` 指定的元素对进入 Uhlenbeck-Ford, UF, 参考势。对于 AlOOH：

- H 扩散：可令 `Al` 和 `O` 为 `spring` 物种，`H-H` 为 UF 自项，`Al-H`、`O-H` 为 UF cross 项。
- H+Al 扩散：可令 `O` 为 `spring` 物种，`H-H`、`Al-Al` 为 UF 自项，`O-H`、`O-Al`、`Al-H` 可作为 UF cross 项参与积分。

代码实际只关心元素符号与原子类型的映射，因此同一套实现可用于更一般的固体亚晶格 + 扩散子晶格体系。

### 1.2 参考态、辅助态、目标态

记总原子数为 $N$。动能项在同一体系的热力学积分路径中不依赖 $\lambda$，因此下面重点写势能项。

Einstein 势能为

$$
U_\mathrm{E}
= \sum_{i \in S_\mathrm{spring}}
\frac{1}{2} k_i
\left|\mathbf r_i - \mathbf r_i^0\right|^2 ,
\tag{1}
$$

其中 $\mathbf r_i^0$ 是进入系综时保存的参考位置，$k_i$ 是该原子的弹簧常数。位移使用最小镜像约定。

UF 势能对每一对指定元素使用

$$
u_\mathrm{UF}(r_{ij};p,\sigma)
= -p k_\mathrm B T
\ln\left(1-\exp\left[-\frac{r_{ij}^2}{\sigma^2}\right]\right).
\tag{2}
$$

代码把 UF 对分为两类：

- `uf_self`：`element_i == element_j`，属于参考态自由能的一部分。
- `uf_cross`：`element_i != element_j`，只作为辅助排斥势参与两阶段积分，不贡献解析 $F_\mathrm{ref}$。

于是参考态为

$$
H_\mathrm{ref}
= K + U_\mathrm{E} + U_\mathrm{UF}^{\mathrm{self}} ,
\tag{3}
$$

辅助态为

$$
H_\mathrm{aux}
= H_\mathrm{ref} + U_\mathrm{UF}^{\mathrm{cross}}
= K + U_\mathrm{E} + U_\mathrm{UF}^{\mathrm{self}}
+ U_\mathrm{UF}^{\mathrm{cross}} ,
\tag{4}
$$

目标态为

$$
H_\mathrm{target}=K+U_\mathrm{target},
\tag{5}
$$

其中 $U_\mathrm{target}$ 由 GPUMD 正常的主势函数计算，例如 NEP 势。

### 1.3 两步热力学积分路径

Stage 1 对应参考态到辅助态：

$$
H_1(\lambda)
= H_\mathrm{ref} + \lambda U_\mathrm{UF}^{\mathrm{cross}},
\qquad
\lambda:0\to1 .
\tag{6}
$$

因此

$$
\frac{\partial H_1}{\partial \lambda}
= U_\mathrm{UF}^{\mathrm{cross}},
\tag{7}
$$

动力学中使用的力为

$$
\mathbf F_1(\lambda)
= \mathbf F_\mathrm{E}
+ \mathbf F_\mathrm{UF}^{\mathrm{self}}
+ \lambda \mathbf F_\mathrm{UF}^{\mathrm{cross}} .
\tag{8}
$$

Stage 2 对应辅助态到目标态：

$$
H_2(\lambda)
= (1-\lambda)H_\mathrm{aux}
+ \lambda H_\mathrm{target},
\qquad
\lambda:0\to1 .
\tag{9}
$$

因此

$$
\frac{\partial H_2}{\partial \lambda}
= U_\mathrm{target} - U_\mathrm{aux},
\qquad
U_\mathrm{aux}
= U_\mathrm{E}
+ U_\mathrm{UF}^{\mathrm{self}}
+ U_\mathrm{UF}^{\mathrm{cross}},
\tag{10}
$$

动力学中使用的力为

$$
\mathbf F_2(\lambda)
= (1-\lambda)\mathbf F_\mathrm{aux}
+ \lambda \mathbf F_\mathrm{target}.
\tag{11}
$$

最终 Helmholtz 自由能由两步积分叠加：

$$
F_\mathrm{target}
= F_\mathrm{ref}
+ \Delta F_1
+ \Delta F_2 .
\tag{12}
$$

在这三个新增源码文件中，每个 stage 的 YAML 只输出本阶段积分值和 $F_\mathrm{ref}$。最终的 $F_\mathrm{target}$ 与 $G_\mathrm{target}=F_\mathrm{target}+PV$ 由后处理脚本 `tools/si_free_energy_sum.py` 合并两个 stage YAML 后给出。

---

## 2. `ensemble_ti_superionic.cuh` 逐段解读

### 2.1 文件头与 include 区域，行 1-27

这一段包含 GPUMD 的 GPL 文件头以及本类需要的依赖：

- `ensemble_lan.cuh`：使 `Ensemble_TI_Superionic` 继承 Langevin NVT 积分框架。
- `force/force.cuh`：声明 `Force` 类型，因为 `compute3()` 与 `find_reference_forces()` 都需要访问主势和近邻表。
- `langevin_utilities.cuh`：用于初始化 Langevin 恒温器所需的 `curand_states`。
- `model/box.cuh`：提供盒子、周期边界和最小镜像相关能力。
- `utilities/common.cuh`：提供 $K_\mathrm B$、$\hbar$、单位换算等常量。
- `utilities/error.cuh`：提供 `PRINT_INPUT_ERROR` 等错误处理。
- `utilities/gpu_vector.cuh`：提供 GPUMD 封装的 GPU 数组。
- `utilities/read_file.cuh`：提供输入解析辅助函数。

这里 `.cuh` 已经包含 `force/force.cuh`，因此 `.cu` 文件中即使不重复 include `force/force.cuh` 也能使用 `Force` 的完整定义。`ti_liquid` 的 `.cu` 和 `.cuh` 都 include 了它，是更显式但冗余的写法。

### 2.2 `SuperionicStage`，行 29

```cpp
enum class SuperionicStage { stage1 = 1, stage2 = 2 };
```

这个枚举把同一个 C++ 类区分成两个公开命令：

- `stage1`：参考态 $\to$ 辅助态。
- `stage2`：辅助态 $\to$ 目标态。

代码没有写两个几乎重复的类，而是用一个 `stage` 标志决定公式中的 $H_1$ 或 $H_2$。

### 2.3 `SuperionicUFPair`，行 31-37

```cpp
struct SuperionicUFPair
{
  std::string element_i;
  std::string element_j;
  double p = 0.0;
  double sigma = 0.0;
};
```

这个结构保存一条用户输入的 UF 元素对。它不直接保存原子类型编号，而保存元素符号。原因是命令面向用户时应写 `uf H H 25 1.0` 或 `uf O H 10 1.0`，而不是依赖 GPUMD 内部 type id。真正的元素符号到 type id 转换在 `prepare_reference_state()` 中完成。

### 2.4 类声明与继承，行 39-43

```cpp
class Ensemble_TI_Superionic : public Ensemble_LAN
```

该类继承 `Ensemble_LAN`，说明它使用 GPUMD 已有的 Langevin NVT 积分框架。超离子自由能功能只改变力和用于积分的势能组合，不重新实现温控器和速度 Verlet 流程。

构造函数接收：

- `params` 与 `num_params`：`ensemble` 命令行分词后的参数。
- `input_stage`：由 `integrate.cu` 在识别 `ti_superionic_stage1/2` 时传入。

析构函数负责收尾：关闭 CSV，并写出本阶段 YAML。

### 2.5 `compute1()` 与 `compute3()`，行 45-58

GPUMD 的积分流程被拆成多个阶段。这里重载：

- `compute1()`：本 step 前半段积分，主要委托给 `Ensemble_LAN::compute1()`。
- `compute3()`：主势力已经计算后执行，适合读取 $U_\mathrm{target}$、访问 `Force` 对象、重写或混合力。

新增功能必须使用 `compute3()`，因为 Stage 2 需要主势给出的目标力 $\mathbf F_\mathrm{target}$ 和目标势能 $U_\mathrm{target}$，Stage 1 也需要主势对象中已经准备好的径向近邻表来计算 UF 势。

### 2.6 公开辅助函数，行 60-70

这些函数构成运行时主流程：

- `init()`：首次进入系综时初始化输出文件、随机数状态和 GPU 缓冲区。
- `find_lambda()`：根据当前 step 决定 $\lambda$、$\Delta\lambda$ 和是否处于积分区间。
- `find_thermo()`：从 GPUMD thermo 数组中读目标势能 `pe`。
- `find_reference_forces(Force&)`：计算 Einstein、UF self、UF cross 的能量和力。
- `apply_stage_forces()`：按 Stage 1 或 Stage 2 公式改写原子力。
- `get_sum()`：对每原子能量数组做 GPU 归约。
- `accumulate_msd_for_auto_k()` 与 `finalize_auto_k()`：支持 `spring auto`。
- `accumulate_work()`：累加非平衡功。
- `switch_func()` 与 `dswitch_func()`：给出光滑切换函数及其每步导数。

### 2.7 自由能辅助函数，行 72-75

```cpp
double get_uf_fe_for_pair(const SuperionicUFPair& pair, int count);
void compute_reference_free_energy();
```

这两个函数只用于解析参考自由能：

$$
F_\mathrm{ref}
= F_\mathrm{Einstein}
+ F_\mathrm{UF}^{\mathrm{self}} .
\tag{13}
$$

cross UF 项不进入 $F_\mathrm{ref}$，因为其作用是从参考态引入辅助排斥势，对应的自由能差由 Stage 1 的积分给出。

### 2.8 stage、输出、lambda 和功变量，行 76-91

主要变量含义如下：

- `stage`：当前类实例是 Stage 1 还是 Stage 2。
- `output_file`：当前 stage 的 CSV 文件。
- `lambda`：当前耦合参数。
- `dlambda`：当前 step 的 $\Delta\lambda$，也就是切换函数对 step 的导数。
- `t_equil`：平衡步数。
- `t_switch`：正向或反向切换步数。
- `target_pressure`：用于 $G=F+PV$ 后处理，不影响模拟动力学。
- `V`：每原子体积 $V_\mathrm{box}/N$。
- `W_forward`：正向切换功，单位 eV/atom。
- `W_backward`：反向切换功，代码中按真实 $\Delta\lambda<0$ 累加，因此通常为负值。
- `delta_F`：本 stage 的自由能差。
- `beta`：$\beta=1/(k_\mathrm BT)$。
- `pe`：主势计算出的目标势能 $U_\mathrm{target}$。
- `U_einstein`、`U_uf_self`、`U_uf_cross`、`U_aux`：当前构型的各参考势能分量。
- `dHdlambda`：当前 stage 的 $\partial H/\partial\lambda$。
- `F_Einstein`、`F_UF_self`、`F_ref`：解析参考自由能分量，均按总原子数归一化为 eV/atom。

功的累积公式在代码中是

$$
W \leftarrow W
+ \frac{1}{N}\frac{\partial H}{\partial\lambda}\Delta\lambda .
\tag{14}
$$

正反向组合为

$$
\Delta F
= \frac{1}{2}\left(W_\mathrm{forward}-W_\mathrm{backward}\right).
\tag{15}
$$

因为反向过程中 $\Delta\lambda<0$，`W_backward` 本身带负号，所以这里是减去 `W_backward`。

### 2.9 解析状态和 CPU/GPU 缓冲区，行 92-117

这些成员把用户输入转化为 GPU kernel 可直接使用的数组。

CPU 侧：

- `spring_map`：显式弹簧常数，键为元素符号，值为 $k$。
- `auto_spring_species`：`spring auto` 模式下需要自动估算 $k$ 的元素列表。
- `uf_pairs`：用户输入的 UF 对。
- `cpu_k`：每个原子的弹簧常数。
- `cpu_spring_mask`：每个原子是否属于 Einstein 参考。
- `cpu_uf_p`：按 type-pair 排列的 UF 参数 $p$。
- `cpu_uf_sigma_sqrd`：按 type-pair 排列的 $\sigma^2$。
- `cpu_uf_kind`：按 type-pair 标记 0/1/2，分别表示无 UF、自 UF、交叉 UF。

GPU 侧：

- `gpu_k`、`gpu_spring_mask`：每原子弹簧数据。
- `gpu_uf_p`、`gpu_uf_sigma_sqrd`、`gpu_uf_kind`：每类型对 UF 数据。
- `gpu_einstein`、`gpu_uf_self`、`gpu_uf_cross`：每原子势能分量。
- `gpu_msd`：`spring auto` 平衡阶段累计 MSD。
- `gpu_aux_fx/fy/fz`：辅助态力中的 Einstein + UF self 分量，之后可加入 cross。
- `gpu_cross_fx/fy/fz`：UF cross 力。
- `position_0`：进入系综时的参考位置，长度为 $3N$，布局为 x/y/z 三段。

### 2.10 私有准备和校验函数，行 119-123

- `prepare_reference_state()`：把元素符号解析成 per-atom 和 per-type-pair 数组，并分配 GPU 缓冲区。
- `validate_species()`：检查用户输入中的元素是否存在、UF 参数是否合法、self UF 是否可解析。
- `is_supported_self_p()`：判断 self UF 的 $p$ 是否有解析自由能表。
- `find_type_for_symbol()`：通过元素符号找 GPUMD type id。
- `write_yaml_pair_list()`：把 UF pair 写入 YAML。

---

## 3. `ensemble_ti_superionic.cu` 逐段解读

### 3.1 include 与匿名命名空间，行 1-24

`.cu` 文件首先 include：

- `ensemble_ti_superionic.cuh`：类声明和大部分依赖。
- `uf_reference.cuh`：解析 UF 自由能表。
- `utilities/gpu_macro.cuh`：`GPU_CHECK_KERNEL` 等 CUDA 错误检查宏。
- C/C++ 标准库：`cstdlib`、`cstring`、`math.h`、`set`。

之后进入匿名命名空间 `namespace { ... }`。这里定义的工具函数和 CUDA kernel 只在本 `.cu` 文件内部可见，避免污染全局符号空间。这与 GPUMD 其它 integrate 文件的风格一致。

### 3.2 输入关键字识别，行 27-33

`is_superionic_keyword()` 判断当前 token 是否是新命令的关键字：

```text
temp, tperiod, tequil, tswitch, press, spring, uf
```

这个函数用于解析可变长度的 `spring` 和 `uf` 参数。例如显式弹簧可以写成：

```text
spring Al 12.0 O 12.0 uf H H 25 1.0
```

解析器遇到下一个关键字时就知道当前 `spring` 参数段结束。

### 3.3 stage 与输出文件名，行 35-50

`stage_number()` 把 `SuperionicStage` 转成整数 1 或 2，用于打印和 YAML。

`csv_filename()` 和 `yaml_filename()` 决定输出文件：

```text
ti_superionic_stage1.csv
ti_superionic_stage1.yaml
ti_superionic_stage2.csv
ti_superionic_stage2.yaml
```

这保证两个 stage 可以独立运行，并由后处理工具合并。

### 3.4 NEP 小盒子检查，行 52-63

`is_nep_small_box()` 用于判断主势为 NEP 且盒子尺寸太小的情况。新增 UF 计算不手动指定 cutoff，而是复用主势的径向近邻表：

$$
U_\mathrm{UF}
= \sum_{(i,j)\in \mathrm{NL}_\mathrm{radial}}
u_\mathrm{UF}(r_{ij}).
\tag{16}
$$

对于某些小盒子 NEP 路径，主势可能不暴露普通径向近邻表。此时 UF 无法可靠遍历邻居，所以代码直接报错，提示使用更大的周期盒子。这个检查体现了之前确定的方案：不让用户手动指定 UF cutoff，而依赖主势近邻表。

### 3.5 `gpu_zero_superionic_arrays()`，行 65-89

这个 GPU kernel 每个线程处理一个原子，把以下数组清零：

- 每原子能量：`einstein`、`uf_self`、`uf_cross`。
- 辅助力：`aux_fx/fy/fz`。
- cross 力：`cross_fx/fy/fz`。

每个 step 都重新计算参考势和力，因此必须先清零。否则上一 step 的力和能量会残留。

数学上它做的是初始化：

$$
U_{\mathrm E,i}=0,\quad
U^{\mathrm{self}}_{\mathrm{UF},i}=0,\quad
U^{\mathrm{cross}}_{\mathrm{UF},i}=0,
\tag{17}
$$

$$
\mathbf F_{\mathrm{aux},i}=0,\quad
\mathbf F_{\mathrm{cross},i}=0 .
\tag{18}
$$

### 3.6 `gpu_find_superionic_spring()`，行 91-118

该 kernel 计算 Einstein 弹簧势能和力。

对原子 $i$，代码先计算相对参考位置的位移：

$$
\Delta\mathbf r_i
= \mathbf r_i-\mathbf r_i^0 ,
\tag{19}
$$

并用 `apply_mic()` 施加最小镜像约定，避免周期边界下位移跳变。

势能：

$$
U_{\mathrm E,i}
= \frac{1}{2}k_i|\Delta\mathbf r_i|^2 .
\tag{20}
$$

力：

$$
\mathbf F_{\mathrm E,i}
= -k_i\Delta\mathbf r_i .
\tag{21}
$$

如果某个原子不是 spring 物种，则 `k[i]=0`，势能和力自然为零。计算出的 Einstein 力被加到 `aux_f*` 中，因为 $U_\mathrm{E}$ 是参考态和辅助态的一部分。

### 3.7 `gpu_add_superionic_msd()`，行 120-140

这个 kernel 只服务于 `spring auto`。在平衡阶段，它对 spring 物种累计均方位移：

$$
\mathrm{MSD}_i
\leftarrow \mathrm{MSD}_i
+ |\mathbf r_i-\mathbf r_i^0|^2 .
\tag{22}
$$

只有 `spring_mask[i] > 0.5` 的原子才参与。平衡结束后，代码用

$$
k
= \frac{3k_\mathrm BT}
{\left\langle |\mathbf r_i-\mathbf r_i^0|^2\right\rangle}
\tag{23}
$$

估计该元素的 Einstein 弹簧常数。这是由三维谐振子的能量均分关系得到的：

$$
\frac{1}{2}k\langle |\Delta\mathbf r|^2\rangle
= \frac{3}{2}k_\mathrm BT .
\tag{24}
$$

### 3.8 `gpu_find_superionic_uf()`，行 142-224

这是 UF 势能和力的核心 kernel。每个线程处理一个中心原子 $i$，遍历主势径向近邻表 `NN/NL` 中的邻居 $j$。

#### 3.8.1 type-pair 查询

代码用

```cpp
pair = type_i * num_types + type_j
kind = uf_kind[pair]
```

查找当前元素对是否有 UF 相互作用：

- `kind == 0`：未定义 UF，跳过。
- `kind == 1`：self UF。
- `kind == 2`：cross UF。

这种矩阵化写法把字符串解析提前到 CPU 准备阶段，GPU kernel 中只做整数索引，效率更高。

#### 3.8.2 UF 势能

对有 UF 的 pair，代码计算

$$
r_{ij}^2
= |\mathbf r_j-\mathbf r_i|^2
\tag{25}
$$

和

$$
u_{ij}
= -\frac{p}{\beta}
\ln\left(1-\exp\left[-\frac{r_{ij}^2}{\sigma^2}\right]\right)
= -p k_\mathrm BT
\ln\left(1-\exp\left[-\frac{r_{ij}^2}{\sigma^2}\right]\right).
\tag{26}
$$

代码中 `beta = 1/(K_B*T)`，所以 `p / beta = p k_B T`。

#### 3.8.3 UF 力

由式 (26) 对 $\mathbf r_i$ 求负梯度，得到原子 $i$ 上由 $j$ 施加的力：

$$
\mathbf F_{i\leftarrow j}^{\mathrm{UF}}
= -\frac{2p k_\mathrm BT}
{\sigma^2\left[\exp(r_{ij}^2/\sigma^2)-1\right]}
\left(\mathbf r_j-\mathbf r_i\right).
\tag{27}
$$

代码中的

```cpp
factor = -2.0 * p / (beta * sigma2 * (exp(r2 / sigma2) - 1.0));
fx = dx * factor;
```

正是这个公式。由于 `factor` 为负，若 $j$ 在 $i$ 的正 x 方向，$i$ 受到负 x 方向力，表现为排斥。

#### 3.8.4 self 与 cross 的分流

如果 `kind == 1`：

$$
U_{\mathrm{UF},i}^{\mathrm{self}}
\leftarrow U_{\mathrm{UF},i}^{\mathrm{self}}
+ \frac{1}{2}u_{ij},
\tag{28}
$$

并把力加到 `aux_f*`。self UF 是参考态的一部分，所以属于 auxiliary force。

如果 `kind == 2`：

$$
U_{\mathrm{UF},i}^{\mathrm{cross}}
\leftarrow U_{\mathrm{UF},i}^{\mathrm{cross}}
+ \frac{1}{2}u_{ij},
\tag{29}
$$

并把力加到 `cross_f*`。cross UF 在 Stage 1 中以 $\lambda$ 缩放，在 Stage 2 中作为辅助态的一部分。

势能加上 `0.5` 是为了避免双计数。GPUMD 径向近邻表按每个中心原子遍历邻居时，同一个 pair 会在 $i$ 和 $j$ 两个中心上各出现一次；力不乘 `0.5`，因为每个中心原子的受力都需要完整贡献。

### 3.9 `gpu_apply_superionic_stage1()`，行 226-245

Stage 1 的 Hamiltonian 是

$$
H_1(\lambda)=H_\mathrm{ref}+\lambda U_\mathrm{UF}^{\mathrm{cross}} .
\tag{30}
$$

因此力为

$$
\mathbf F_i
= \mathbf F_{\mathrm{aux,no-cross},i}
+ \lambda \mathbf F_{\mathrm{cross},i},
\tag{31}
$$

其中 `aux_f*` 此时只包含 Einstein + UF self。代码直接覆盖 `atom->force_per_atom`：

```cpp
fx[i] = aux_fx[i] + lambda * cross_fx[i];
```

这意味着 Stage 1 中主势已经算出的目标力不会参与动力学。主势仍然需要存在，是因为 GPUMD 的框架和 UF 近邻表需要它。

### 3.10 `gpu_add_cross_to_aux()`，行 247-262

Stage 2 的辅助态包含 cross UF：

$$
\mathbf F_\mathrm{aux}
= \mathbf F_\mathrm{E}
+ \mathbf F_\mathrm{UF}^{\mathrm{self}}
+ \mathbf F_\mathrm{UF}^{\mathrm{cross}} .
\tag{32}
$$

而 `gpu_find_superionic_uf()` 结束后，`aux_f*` 还只是 Einstein + self UF，cross 单独在 `cross_f*`。这个 kernel 就是把 cross 力加进 aux 力中，为 Stage 2 混合做准备。

### 3.11 `gpu_apply_superionic_stage2()`，行 264-280

Stage 2 的 Hamiltonian 是

$$
H_2(\lambda)
= (1-\lambda)H_\mathrm{aux}
+ \lambda H_\mathrm{target}.
\tag{33}
$$

力也按同样线性组合：

$$
\mathbf F_i
= (1-\lambda)\mathbf F_{\mathrm{aux},i}
+ \lambda\mathbf F_{\mathrm{target},i}.
\tag{34}
$$

进入该 kernel 前，`atom->force_per_atom` 中已经是主势计算出的 $\mathbf F_\mathrm{target}$。kernel 用 `aux_f*` 与当前 `fx/fy/fz` 混合，再写回 `fx/fy/fz`。

### 3.12 `gpu_sum_array()`，行 282-307

这个 kernel 用一个 block 和 1024 个线程归约一个长度为 $N$ 的数组。每个线程跨多个 patch 累加，然后在 shared memory 中规约，最终把总和写入 `data[0]`。

它用于

$$
U_\mathrm{E}=\sum_i U_{\mathrm E,i},
\quad
U_\mathrm{UF}^{\mathrm{self}}
=\sum_i U_{\mathrm{UF},i}^{\mathrm{self}},
\quad
U_\mathrm{UF}^{\mathrm{cross}}
=\sum_i U_{\mathrm{UF},i}^{\mathrm{cross}}.
\tag{35}
$$

注意这个归约会覆盖 `data[0]`。这在当前流程中没有问题，因为每个 step 开始都会重新清零并重新计算 per-atom 能量数组。

### 3.13 构造函数：指针初始化与默认参数，行 311-325

构造函数先保存 `stage`，并把 GPUMD ensemble 基类会用到的指针设为空。随后设置默认温度耦合时间：

```cpp
temperature_coupling = 100;
```

解析从 `i = 2` 开始，因为 `params[0]` 是 `ensemble`，`params[1]` 是 `ti_superionic_stage1` 或 `ti_superionic_stage2`。

### 3.14 构造函数：`temp`、`tperiod`、`tequil`、`tswitch`、`press`，行 326-348

这些关键字含义是：

- `temp <T>`：目标温度，必须指定且 $T>0$。
- `tperiod <tau>`：Langevin 温控耦合时间，默认 100 step，必须 $\ge 1$。
- `tequil <steps>`：初始平衡步数，以及正向和反向之间在 $\lambda=1$ 的平衡步数。
- `tswitch <steps>`：每个方向切换的步数，必须 $>0$。
- `press <P>`：输入压力，只用于后处理 $G=F+PV$，不改变动力学。

这里需要特别注意 `tequil` 的含义。构造函数本身只负责把输入数值存入 `t_equil`，真正决定它如何使用的是后面的 `find_lambda()`。在该实现中，同一个 `t_equil` 会被使用两次：

1. 运行开始后的前 `t_equil` 步，体系停留在 $\lambda=0$，作为初始端点平衡。
2. 正向切换结束后，体系停留在 $\lambda=1$，再次平衡 `t_equil` 步，然后才开始反向切换。

所以一次完整的 stage 运行总步数是

$$
N_\mathrm{steps}
= t_\mathrm{equil}
+t_\mathrm{switch}
+t_\mathrm{equil}
+t_\mathrm{switch}
=2(t_\mathrm{equil}+t_\mathrm{switch}).
\tag{36}
$$

这也是 `init()` 中打印

```cpp
printf("The number of steps should be set to %d!\n", 2 * (t_equil + t_switch));
```

的原因。换言之，用户输入的 `tequil` 不是“总平衡步数”，而是“每个端点的平衡步数”：先用于 $\lambda=0$ 端点，再用于 $\lambda=1$ 端点。

压力被转换成 GPUMD 内部能量/体积单位：

$$
P_\mathrm{internal}
= \frac{P_\mathrm{input}}
{\texttt{PRESSURE\_UNIT\_CONVERSION}} .
\tag{37}
$$

代码没有强制要求 `press` 必须出现。如果不指定，`target_pressure` 保持 0，后处理得到的 $G$ 就等于 $F$。

### 3.15 构造函数：`spring` 解析，行 349-379

`spring` 有两种形式。

显式弹簧：

```text
spring Al 12.0 O 12.0
```

会写入

```cpp
spring_map["Al"] = 12.0;
spring_map["O"] = 12.0;
```

后续每个 Al/O 原子使用对应 $k$。

自动弹簧：

```text
spring auto Al O
```

会把元素符号加入 `auto_spring_species`，并设置 `auto_k = true`。平衡阶段根据 MSD 估算弹簧常数。代码禁止混用 `spring auto` 和显式 `spring <element> <k>`。

细节：

- 显式 $k$ 必须为正。
- `spring auto` 会检查重复元素并报错。
- 显式 `spring` 用 `std::map` 保存；如果同一元素重复输入，后一次会覆盖前一次，代码没有单独报错。

### 3.16 构造函数：`uf` 解析，行 380-389

每个 UF 项格式为：

```text
uf <element_i> <element_j> <p> <sigma>
```

代码保存元素符号、$p$、$\sigma$，但此处还不判断 self/cross，也不查找 type id。真正的合法性检查在 `validate_species()` 中完成。

### 3.17 构造函数：全局输入检查与 Langevin 常数，行 395-427

构造函数最后检查：

- 必须指定 `tequil` 和 `tswitch`。
- 必须指定 `temp`，且温度大于 0。
- `tperiod >= 1`。
- `tswitch > 0`。
- `spring auto` 时 `tequil > 0`，因为需要平衡阶段采样 MSD。
- 至少有一个 spring 物种。
- 至少有一个 UF pair。

随后设置

$$
\beta = \frac{1}{k_\mathrm BT}
\tag{38}
$$

并初始化 Langevin 系数：

$$
c_1=\exp\left[-\frac{1}{2\tau_T}\right],
\qquad
c_2=\sqrt{(1-c_1^2)k_\mathrm BT}.
\tag{39}
$$

这里 $\tau_T$ 对应 `temperature_coupling`。`type = 3` 表示该 ensemble 需要走带 `Force&` 参数的 `compute3()` 路径。

### 3.18 `is_supported_self_p()`，行 429-432

这个函数转发到 `uf_reference::supports_p(p)`。self UF 的解析自由能只支持表中已有的 $p$：

$$
p \in \{1,25,50,75,100\}.
\tag{40}
$$

cross UF 不需要解析自由能，因此只要求 $p>0$、$\sigma>0$。

### 3.19 `find_type_for_symbol()`，行 434-441

该函数遍历所有原子，找到第一个元素符号匹配的原子，然后返回它的 `cpu_type`。如果找不到，返回 -1。

这意味着用户命令以元素符号为接口，而 GPU kernel 最终以 type id 为接口。一般 GPUMD 输入中一个元素对应一个 type；如果人为让同一元素符号对应多个 type，该函数只会返回第一个 type，这种情况不适合当前实现。

### 3.20 `validate_species()`，行 443-481

这个函数负责在真正分配 GPU 数组前拦截错误输入。

对 spring 物种：

- 检查每个元素符号是否存在于结构中。

对 UF pair：

- 检查两个元素都存在。
- 检查 $p>0$、$\sigma>0$。
- 若是 self pair，检查 $p$ 是否属于式 (39) 的支持集合。
- 用无序 type-pair key 检查重复 pair，因此 `uf H O ...` 与 `uf O H ...` 被视为同一对。
- 要求至少存在一个 self UF pair。

要求 self UF pair 的原因是参考态中的流体自由能来自解析 UF 自由能。如果完全没有 self UF，则 $F_\mathrm{ref}$ 不完整。

### 3.21 `prepare_reference_state()`：数组初始化，行 483-493

该函数先调用 `validate_species()`，然后分配 CPU 数组：

$$
\texttt{cpu\_k}[i] \leftrightarrow k_i,
\quad
\texttt{cpu\_spring\_mask}[i]\in\{0,1\},
\tag{41}
$$

以及按 type-pair 排列的 UF 矩阵：

$$
\texttt{pair index}
= t_i N_\mathrm{type}+t_j .
\tag{42}
$$

`cpu_uf_kind` 的值为：

$$
0:\text{无 UF},\quad
1:\text{self UF},\quad
2:\text{cross UF}.
\tag{43}
$$

### 3.22 `prepare_reference_state()`：spring 原子标记，行 494-508

对每个原子读取 `atom->cpu_atom_symbol[i]`。

在 `spring auto` 模式下，如果原子元素在 `auto_spring_species` 中，则

$$
\texttt{spring\_mask}_i=1 .
\tag{44}
$$

此时 $k_i$ 暂时为 0，等平衡阶段结束后根据 MSD 填入。

在显式 spring 模式下，如果元素在 `spring_map` 中，则

$$
\texttt{spring\_mask}_i=1,\qquad
k_i=k(\mathrm{element}_i).
\tag{45}
$$

非 spring 原子的 $k_i=0$，不会产生 Einstein 势能或力。

### 3.23 `prepare_reference_state()`：UF pair 矩阵，行 510-522

对每条用户输入的 UF pair，代码查找两个元素的 type id，并同时填充 $ij$ 和 $ji$ 两个方向：

$$
p_{ij}=p_{ji}=p,
\qquad
\sigma^2_{ij}=\sigma^2_{ji}=\sigma^2,
\qquad
\mathrm{kind}_{ij}=\mathrm{kind}_{ji}.
\tag{46}
$$

这让 GPU kernel 在遍历 $i\to j$ 或 $j\to i$ 时都能找到相同的 UF 参数。

### 3.24 `prepare_reference_state()`：GPU 分配与参考位置，行 524-550

CPU 数组准备好后复制到 GPU：

- `gpu_k`
- `gpu_spring_mask`
- `gpu_uf_p`
- `gpu_uf_sigma_sqrd`
- `gpu_uf_kind`

随后分配所有每原子能量、力、MSD 数组。

最后复制当前原子位置到 `position_0`：

$$
\mathbf r_i^0 \leftarrow \mathbf r_i(t=0).
\tag{47}
$$

这就是 Einstein 晶体的参考晶格位置。对于需要高质量自由能计算的生产任务，运行该系综前应确保初始结构就是希望束缚的参考晶格。

### 3.25 `write_yaml_pair_list()`，行 552-579

该函数把 `uf_pairs` 按 self 或 cross 筛选后写到 YAML，例如：

```yaml
uf_self_pairs:
  - {element_i: "H", element_j: "H", p: 25, sigma: 1}
uf_cross_pairs:
  - {element_i: "O", element_j: "H", p: 10, sigma: 1}
```

如果某类 pair 不存在，则写成空列表：

```yaml
uf_cross_pairs: []
```

这种元数据用于后处理阶段确认 Stage 1 和 Stage 2 使用了完全相同的参考态定义。

### 3.26 `get_uf_fe_for_pair()`：UF 自由能变量，行 581-606

该函数计算某个 self UF 物种的解析自由能贡献。设该物种原子数为 $N_s$，每个该物种原子平均占据体积

$$
V_s=\frac{V_\mathrm{box}}{N_s}.
\tag{48}
$$

代码定义

$$
x_\mathrm{UF}
= \frac{(\pi\sigma^2)^{3/2}}{2V_s}.
\tag{49}
$$

随后按 $x_\mathrm{UF}$ 所在区间选择样条表 index：

- $x<0.1$：细步长 0.0025。
- $0.1\le x<1$：步长 0.025。
- $1\le x<4$：步长 0.1。
- $x\ge4$：使用表尾值。

`uf_reference::get_data(pair.p)` 返回该 $p$ 对应的累计积分表和分段系数表。`uf_reference::fe()` 返回无量纲 excess 自由能函数值 $f_\mathrm{UF}(x,p)$。代码得到

$$
F_\mathrm{UF}^{\mathrm{excess}}
= N_s k_\mathrm BT f_\mathrm{UF}(x_\mathrm{UF},p).
\tag{50}
$$

### 3.27 `get_uf_fe_for_pair()`：理想气体项，行 607-619

代码查找该物种质量 $m$，然后定义热 de Broglie 波长因子：

$$
\Lambda
= \hbar\sqrt{\frac{2\pi}{m k_\mathrm BT}} .
\tag{51}
$$

理想气体 Helmholtz 自由能为

$$
F_\mathrm{IG}
= N_s k_\mathrm BT
\left[
\ln\left(\frac{1}{V_s}\right)-1
+3\ln\Lambda
\right].
\tag{52}
$$

也就是

$$
F_\mathrm{IG}
= N_s k_\mathrm BT
\left[
\ln\left(\rho_s\Lambda^3\right)-1
\right],
\qquad
\rho_s=\frac{N_s}{V_\mathrm{box}}.
\tag{53}
$$

函数最终返回

$$
\frac{
F_\mathrm{UF}^{\mathrm{excess}}+F_\mathrm{IG}
}{N},
\tag{54}
$$

也就是按总原子数 $N$ 归一化的 eV/atom。这与 CSV、YAML 中其它能量单位保持一致。

### 3.28 `compute_reference_free_energy()`，行 622-654

该函数计算

$$
F_\mathrm{ref}
=F_\mathrm{Einstein}
+F_\mathrm{UF}^{\mathrm{self}}.
\tag{55}
$$

Einstein 部分逐个 spring 原子累加：

$$
\omega_i
= \sqrt{\frac{k_i}{m_i}},
\tag{56}
$$

$$
F_\mathrm{Einstein}
= \frac{3k_\mathrm BT}{N}
\sum_{i\in S_\mathrm{spring}}
\ln\left(
\frac{\hbar\omega_i}{k_\mathrm BT}
\right).
\tag{57}
$$

这里代码把每个 spring 原子视为独立三维谐振子，没有额外的质心约束修正项。

UF self 部分遍历 `uf_pairs`，只对 `element_i == element_j` 的 pair 调用 `get_uf_fe_for_pair()`。cross pair 不在这里出现，因为其自由能贡献由 Stage 1 数值积分给出。

### 3.29 `init()`，行 656-683

`init()` 在第一个 step 被调用，主要做四件事。

第一，打印建议总步数：

$$
N_\mathrm{steps}
=2(t_\mathrm{equil}+t_\mathrm{switch}).
\tag{58}
$$

这对应：

1. $\lambda=0$ 平衡 `t_equil`。
2. 正向切换 `t_switch`。
3. $\lambda=1$ 平衡 `t_equil`。
4. 反向切换 `t_switch`。

第二，打开当前 stage 的 CSV 文件并写 header。

Stage 1：

```text
lambda,dlambda,U_einstein,U_uf_self,U_uf_cross,dHdlambda
```

Stage 2：

```text
lambda,dlambda,U_target,U_einstein,U_uf_self,U_uf_cross,U_aux,dHdlambda
```

第三，初始化 Langevin 随机数状态。

第四，调用 `prepare_reference_state()`，把用户输入变成 GPU 可用的数值数组。

### 3.30 析构函数，行 685-731

析构函数在 ensemble 生命周期结束时运行。它先在可能的情况下更新每原子体积：

$$
V=\frac{V_\mathrm{box}}{N},
\tag{59}
$$

并调用 `compute_reference_free_energy()` 计算 $F_\mathrm{Einstein}$、$F_\mathrm{UF}^{\mathrm{self}}$、$F_\mathrm{ref}$。

然后写 YAML：

```yaml
stage: 1 or 2
T: ...
V: ...
P: ...
N_total: ...
spring_species: [...]
spring_constants: [...]
uf_self_pairs: [...]
uf_cross_pairs: [...]
W_forward: ...
W_backward: ...
delta_F: ...
F_Einstein: ...
F_UF_self: ...
F_ref: ...
```

这里的 `P` 是内部单位，`V` 是每原子体积，因此后处理中的 $PV$ 直接给出 eV/atom。

析构函数最后关闭 CSV 和 YAML 文件。

### 3.31 `compute1()`，行 733-743

`compute1()` 在当前 step 前半段被 GPUMD 调用。它只做一件额外工作：

```cpp
if (*current_step == 0 && !initialized)
  init();
```

也就是保证所有输出文件和 GPU 缓冲区在第一次积分前准备好。随后调用

```cpp
Ensemble_LAN::compute1(...)
```

继续执行 GPUMD 原有 Langevin 积分的前半步。

### 3.32 `find_thermo()`，行 745-758

该函数调用 `Ensemble::find_thermo()` 计算 thermo 数组，并把 thermo 从 GPU 复制到 CPU。代码取

```cpp
pe = thermo_cpu[1];
```

作为主势目标势能：

$$
pe = U_\mathrm{target}.
\tag{60}
$$

这里的关键点是：`find_thermo()` 只负责取得主势目标体系的势能 `pe`，但它并不直接决定当前 stage 的 $dH/d\lambda$。真正设置 $dH/d\lambda$ 的位置在后面的 `apply_stage_forces()`。

在 Stage 2，`pe` 会进入

$$
\frac{\partial H_2}{\partial\lambda}
= U_\mathrm{target}-U_\mathrm{aux}.
\tag{61}
$$

对应代码是：

```cpp
dHdlambda = pe - U_aux;
```

而在 Stage 1，当然也要计算 $dH_1/d\lambda$，只是它不使用 `pe`。Stage 1 的路径为

$$
H_1(\lambda)
=H_\mathrm{ref}
+\lambda U_\mathrm{UF}^{\mathrm{cross}},
\tag{62}
$$

因此

$$
\frac{\partial H_1}{\partial\lambda}
=U_\mathrm{UF}^{\mathrm{cross}}.
\tag{63}
$$

对应代码是：

```cpp
dHdlambda = U_uf_cross;
```

所以更准确的说法是：Stage 1 会计算 $dH_1/d\lambda$，但其值来自 `find_reference_forces()` 得到的 `U_uf_cross`，而不是 `find_thermo()` 得到的主势能 `pe`。`find_thermo()` 在 Stage 1 中仍被统一调用，是因为 `compute3()` 采用同一套执行流程；但 Stage 1 随后会用参考态和 cross UF 力覆盖主势力，且 `pe` 不参与 Stage 1 的功积分。

### 3.33 `accumulate_msd_for_auto_k()`，行 760-778

如果没有 `spring auto`，或者当前 step 已经达到 `t_equil`，函数直接返回。

否则调用 `gpu_add_superionic_msd()`，在初始平衡阶段累计 spring 物种的 MSD：

$$
\mathrm{MSD}_i^\mathrm{sum}
=\sum_{n=0}^{t_\mathrm{equil}-1}
|\mathbf r_i(n)-\mathbf r_i^0|^2 .
\tag{64}
$$

注意这段采样发生在切换开始前，即 $\lambda=0$ 的平衡阶段。

### 3.34 `finalize_auto_k()`，行 780-820

该函数只在

```cpp
auto_k && *current_step == t_equil - 1
```

时执行，也就是初始平衡阶段最后一个 step。

对每个自动 spring 元素，计算平均 MSD：

$$
\left\langle |\Delta\mathbf r|^2 \right\rangle
= \frac{1}{N_s t_\mathrm{equil}}
\sum_{i\in s}\mathrm{MSD}_i^\mathrm{sum}.
\tag{65}
$$

然后用式 (23) 得到

$$
k_s
=\frac{3k_\mathrm BT}
{\left\langle |\Delta\mathbf r|^2 \right\rangle}.
\tag{66}
$$

该 $k_s$ 写入 `spring_map`，并填回每个 spring 原子的 `cpu_k[i]` 和 `gpu_k`。因此自动弹簧最终与显式弹簧走同一套后续计算流程，YAML 中也会输出 `spring_constants`。

### 3.35 `get_sum()`，行 822-829

这是 `gpu_sum_array()` 的 C++ 包装：

1. 在 GPU 上规约数组。
2. 把 `data[0]` 复制回 CPU。
3. 返回总和。

用于得到总势能分量，而 CSV 输出时再除以 $N$ 得 eV/atom。

### 3.36 `find_reference_forces()`：清零与 spring，行 831-864

这个函数每个 step 重新计算参考/辅助势能和力。

第一步，调用 `gpu_zero_superionic_arrays()` 清空所有参考能量和力。

第二步，调用 `gpu_find_superionic_spring()` 计算：

$$
U_\mathrm{E},\quad
\mathbf F_\mathrm{E}.
\tag{67}
$$

此时 `gpu_aux_f*` 中已有 Einstein 力。

### 3.37 `find_reference_forces()`：近邻表检查，行 865-873

代码先对 NEP 小盒子做特殊检查。随后从主势对象取得径向近邻表：

```cpp
const GPU_Vector<int>& NN = force.potentials[0]->get_NN_radial_ptr();
const GPU_Vector<int>& NL = force.potentials[0]->get_NL_radial_ptr();
```

如果近邻表为空，则报错：

```text
The main potential must provide a radial neighbor list for ti_superionic.
```

这再次体现了当前实现的 cutoff 策略：不在 `ensemble` 命令中手动指定 UF cutoff，而使用主势已经构建的 radial neighbor list。

### 3.38 `find_reference_forces()`：UF 与能量汇总，行 875-903

调用 `gpu_find_superionic_uf()` 后，代码得到：

$$
U_\mathrm{UF}^{\mathrm{self}},
\quad
U_\mathrm{UF}^{\mathrm{cross}},
\quad
\mathbf F_\mathrm{UF}^{\mathrm{self}},
\quad
\mathbf F_\mathrm{UF}^{\mathrm{cross}}.
\tag{68}
$$

随后分别归约：

$$
U_\mathrm{E}=\sum_i U_{\mathrm E,i},
\quad
U_\mathrm{UF}^{\mathrm{self}}
=\sum_i U_{\mathrm{UF},i}^{\mathrm{self}},
\quad
U_\mathrm{UF}^{\mathrm{cross}}
=\sum_i U_{\mathrm{UF},i}^{\mathrm{cross}}.
\tag{69}
$$

并得到

$$
U_\mathrm{aux}
=U_\mathrm{E}
+U_\mathrm{UF}^{\mathrm{self}}
+U_\mathrm{UF}^{\mathrm{cross}}.
\tag{70}
$$

### 3.39 `apply_stage_forces()`：Stage 1，行 905-924

如果当前为 Stage 1：

```cpp
dHdlambda = U_uf_cross;
```

对应式 (7)：

$$
\frac{\partial H_1}{\partial\lambda}
=U_\mathrm{UF}^{\mathrm{cross}}.
\tag{71}
$$

随后调用 `gpu_apply_superionic_stage1()`，把真实用于动力学的力改成式 (31)。Stage 1 不使用主势目标力。

### 3.40 `apply_stage_forces()`：Stage 2，行 925-947

如果当前为 Stage 2：

```cpp
dHdlambda = pe - U_aux;
```

对应式 (10)：

$$
\frac{\partial H_2}{\partial\lambda}
=U_\mathrm{target}-U_\mathrm{aux}.
\tag{72}
$$

随后先把 cross 力加入 auxiliary force：

$$
\mathbf F_\mathrm{aux}
\leftarrow
\mathbf F_\mathrm{E}
+\mathbf F_\mathrm{UF}^{\mathrm{self}}
+\mathbf F_\mathrm{UF}^{\mathrm{cross}},
\tag{73}
$$

再调用 `gpu_apply_superionic_stage2()` 完成

$$
\mathbf F
=(1-\lambda)\mathbf F_\mathrm{aux}
+\lambda \mathbf F_\mathrm{target}.
\tag{74}
$$

### 3.41 `accumulate_work()`，行 950-959

每个处于切换区间的 step，代码累加

$$
\Delta W
=\frac{1}{N}
\frac{\partial H}{\partial\lambda}\Delta\lambda.
\tag{75}
$$

若 `dlambda > 0`，进入正向功：

$$
W_\mathrm{forward}
\leftarrow W_\mathrm{forward}+\Delta W.
\tag{76}
$$

若 `dlambda < 0`，进入反向功：

$$
W_\mathrm{backward}
\leftarrow W_\mathrm{backward}+\Delta W.
\tag{77}
$$

最后实时更新

$$
\Delta F
=\frac{1}{2}
\left(W_\mathrm{forward}-W_\mathrm{backward}\right).
\tag{78}
$$

这里没有显式乘以时间步长，是因为 `dlambda` 已经是每个 MD step 的 $\Delta\lambda$，即 `dswitch_func()` 返回的是对 step 的导数。

### 3.42 `find_lambda()`，行 961-985

该函数先把

$$
\lambda=0,\quad
\Delta\lambda=0,\quad
\texttt{lambda\_active=false}
\tag{79}
$$

作为默认值，并更新每原子体积。随后处理 `spring auto` 的 MSD 累积和弹簧常数最终确定。

令

$$
t = \texttt{current\_step}-t_\mathrm{equil},
\qquad
r_\mathrm{switch}=\frac{1}{t_\mathrm{switch}}.
\tag{80}
$$

代码中对应的核心逻辑是：

```cpp
const int t = *current_step - t_equil;
const double r_switch = 1.0 / t_switch;

if ((t >= 0) && (t <= t_switch)) {
  lambda = switch_func(t * r_switch);
  dlambda = dswitch_func(t * r_switch);
  lambda_active = true;
} else if ((t > t_switch) && (t < t_equil + t_switch)) {
  lambda = 1.0;
} else if ((t >= t_equil + t_switch) && (t <= (t_equil + 2 * t_switch))) {
  lambda = switch_func(1.0 - (t - t_switch - t_equil) * r_switch);
  dlambda = -dswitch_func(1.0 - (t - t_switch - t_equil) * r_switch);
  lambda_active = true;
}
```

这里使用的切换函数为

$$
S(\tau)
=126\tau^5
-420\tau^6
+540\tau^7
-315\tau^8
+70\tau^9,
\qquad 0\le\tau\le1.
\tag{81}
$$

它的导数为

$$
S'(\tau)
=630\tau^4(1-\tau)^4.
\tag{82}
$$

因此 `switch_func(x)` 对应 $S(x)$，`dswitch_func(x)` 对应 $S'(x)/t_\mathrm{switch}$。这样写的目的，是让 $\lambda$ 在切换起点和终点处变化率为零，从而减少非平衡切换带来的突变。

这段代码解释了为什么 `tequil` 会在物理流程中出现两次。因为 `t` 被定义为 `current_step - t_equil`，所以最开始的 `current_step < t_equil` 区间满足 $t<0$，不会进入任何切换分支。此时 `lambda` 保持函数开头设置的默认值 0，`dlambda` 也保持 0。

初始平衡阶段：

$$
0 \le \texttt{current\_step} < t_\mathrm{equil},
\qquad
t<0,
\qquad
\lambda=0,
\qquad
\Delta\lambda=0.
\tag{83}
$$

这一区间不输出 CSV，不累加功，因为 `lambda_active=false`。

正向切换中，定义归一化切换时间

$$
\tau_\mathrm{fwd}
=\frac{t}{t_\mathrm{switch}},
\qquad 0\le\tau_\mathrm{fwd}\le1.
\tag{84}
$$

于是

$$
0\le t\le t_\mathrm{switch},
\qquad
\lambda=S(\tau_\mathrm{fwd}),
\qquad
\Delta\lambda=\frac{S'(\tau_\mathrm{fwd})}{t_\mathrm{switch}}>0.
\tag{85}
$$

中间平衡：

$$
t_\mathrm{switch}<t<t_\mathrm{equil}+t_\mathrm{switch},
\qquad
\lambda=1,
\qquad
\Delta\lambda=0.
\tag{86}
$$

这就是第二次使用 `t_equil` 的地方。因为中间平衡分支的上界是 `t_equil + t_switch`，所以正向切换结束后，体系会在 $\lambda=1$ 的端点继续运行约 `t_equil` 步。这里 `dlambda=0`，`lambda_active=false`，因此不输出 CSV、不累加功，只用于让反向切换从已经平衡的 $\lambda=1$ 状态开始。

反向切换：

$$
t_\mathrm{equil}+t_\mathrm{switch}
\le t
\le t_\mathrm{equil}+2t_\mathrm{switch},
\tag{87}
$$

定义反向切换的归一化时间

$$
\tau_\mathrm{bwd}
=1-\frac{t-t_\mathrm{switch}-t_\mathrm{equil}}
{t_\mathrm{switch}},
\qquad 1\ge\tau_\mathrm{bwd}\ge0.
\tag{88}
$$

则

$$
\lambda=S(\tau_\mathrm{bwd}),
\qquad
\Delta\lambda
=-\frac{1}{t_\mathrm{switch}}
S'(\tau_\mathrm{bwd})<0.
\tag{89}
$$

因此完整时间轴可以整理为：

| 区间 | 代码条件 | 物理含义 | 是否积分 |
|---|---|---|---|
| 初始平衡 | `current_step < t_equil` | $\lambda=0$ 端点平衡 | 否 |
| 正向切换 | `0 <= t <= t_switch` | $\lambda:0\to1$ | 是 |
| 中间平衡 | `t_switch < t < t_equil + t_switch` | $\lambda=1$ 端点平衡 | 否 |
| 反向切换 | `t_equil + t_switch <= t <= t_equil + 2*t_switch` | $\lambda:1\to0$ | 是 |

这正是前面把 `tequil` 描述为“初始平衡步数，以及正向和反向之间在 $\lambda=1$ 的平衡步数”的直接代码依据。

只有正向和反向切换时 `lambda_active = true`，才写 CSV 和累加功。

### 3.43 `compute3()`：总体时序，行 987-1042

这是整个新系综的运行核心。`compute3()` 的调用时机是在 GPUMD 主势已经完成本 step 的力和每原子势能计算之后，因此进入该函数时，`atom->force_per_atom` 中已经有目标体系主势力 $\mathbf F_\mathrm{target}$，`atom->potential_per_atom` 中也已经有目标势能分量。`ti_superionic` 在这里要做的事情，是根据当前 stage 把这些主势结果替换或混合成真正用于 Langevin 后半步积分的力。

源码主体如下：

```cpp
find_lambda();
if (auto_k && *current_step < t_equil) {
  Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
  if (*current_step == t_equil - 1) {
    find_thermo();
    find_reference_forces(force);
    apply_stage_forces();
  }
  return;
}

find_thermo();
find_reference_forces(force);
apply_stage_forces();

if (lambda_active) {
  accumulate_work();
  fprintf(...);
}

Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
```

每个 step 的顺序是：

第一步，调用 `find_lambda()`。它根据 `current_step` 决定当前的 $\lambda$、$\Delta\lambda$ 和 `lambda_active`。这一步不计算势能或力，只决定当前 step 应处于哪一个 Hamiltonian：

$$
H_\mathrm{stage}(\lambda)
=
\begin{cases}
H_\mathrm{ref}+\lambda U_\mathrm{UF}^{\mathrm{cross}}, & \text{Stage 1},\\
(1-\lambda)H_\mathrm{aux}+\lambda H_\mathrm{target}, & \text{Stage 2}.
\end{cases}
\tag{90}
$$

第二步，处理 `spring auto` 的初始平衡特殊分支。如果使用自动弹簧，并且仍满足

$$
\texttt{auto\_k}=\mathrm{true},
\qquad
\texttt{current\_step}<t_\mathrm{equil},
\tag{91}
$$

代码会先执行

```cpp
Ensemble_LAN::compute2(...)
```

然后直接 `return`。这意味着在自动弹簧的初始平衡阶段，体系先按正常主势动力学采样 MSD，用于估计

$$
k_s=\frac{3k_\mathrm BT}
{\left\langle |\Delta\mathbf r|^2\right\rangle_s}.
\tag{92}
$$

在初始平衡最后一个 step，即

$$
\texttt{current\_step}=t_\mathrm{equil}-1,
\tag{93}
$$

代码会在 `compute2()` 之后额外执行

```cpp
find_thermo();
find_reference_forces(force);
apply_stage_forces();
```

这一步的目的不是为了当前 step 再积分一次，而是为了预先把力切换到 stage Hamiltonian 对应的力。这样下一步进入 `compute1()` 时，速度 Verlet/Langevin 前半步使用的已经是正确的参考态或辅助态力。

第三步，在正常分支中调用 `find_thermo()`，从主势 thermo 中取出

$$
pe=U_\mathrm{target}.
\tag{94}
$$

这个值只在 Stage 2 的

$$
\frac{\partial H_2}{\partial\lambda}
=U_\mathrm{target}-U_\mathrm{aux}
\tag{95}
$$

中使用。Stage 1 的

$$
\frac{\partial H_1}{\partial\lambda}
=U_\mathrm{UF}^{\mathrm{cross}}
\tag{96}
$$

不使用 `pe`。

第四步，调用 `find_reference_forces(force)`。它重新计算当前构型下的参考态/辅助态势能分量：

$$
U_\mathrm{E},\quad
U_\mathrm{UF}^{\mathrm{self}},\quad
U_\mathrm{UF}^{\mathrm{cross}},
\tag{97}
$$

并得到

$$
U_\mathrm{aux}
=U_\mathrm{E}
+U_\mathrm{UF}^{\mathrm{self}}
+U_\mathrm{UF}^{\mathrm{cross}}.
\tag{98}
$$

同时它准备两组力：`gpu_aux_f*` 中的 Einstein + self UF 力，以及 `gpu_cross_f*` 中的 cross UF 力。

第五步，调用 `apply_stage_forces()`，按当前 stage 改写实际用于后续积分的 `atom->force_per_atom`。

Stage 1 中：

$$
\mathbf F
=\mathbf F_\mathrm{E}
+\mathbf F_\mathrm{UF}^{\mathrm{self}}
+\lambda\mathbf F_\mathrm{UF}^{\mathrm{cross}},
\tag{99}
$$

并设置

$$
\frac{\partial H}{\partial\lambda}=U_\mathrm{UF}^{\mathrm{cross}}.
\tag{100}
$$

Stage 2 中：

$$
\mathbf F
=(1-\lambda)\mathbf F_\mathrm{aux}
+\lambda\mathbf F_\mathrm{target},
\tag{101}
$$

并设置

$$
\frac{\partial H}{\partial\lambda}=U_\mathrm{target}-U_\mathrm{aux}.
\tag{102}
$$

第六步，如果 `lambda_active=true`，说明当前 step 在正向或反向切换段内，而不是平衡段。此时才调用 `accumulate_work()`：

$$
\Delta W
=\frac{1}{N}
\frac{\partial H}{\partial\lambda}\Delta\lambda.
\tag{103}
$$

正向切换时 $\Delta\lambda>0$，加入 `W_forward`；反向切换时 $\Delta\lambda<0$，加入 `W_backward`。随后把当前 step 的 $\lambda$、$\Delta\lambda$ 和势能分量写入 CSV。

第七步，调用

```cpp
Ensemble_LAN::compute2(...)
```

执行 Langevin 后半步。由于 `apply_stage_forces()` 已经改写了 `atom->force_per_atom`，所以这里的后半步使用的是 stage Hamiltonian 的力，而不是未经处理的主势力。

### 3.44 CSV 输出，行 1012-1039

CSV 只在 `lambda_active` 为真时输出，即只记录正向和反向切换段，不记录平衡段。

Stage 1 每行：

$$
\lambda,\quad
\Delta\lambda,\quad
\frac{U_\mathrm{E}}{N},\quad
\frac{U_\mathrm{UF}^{\mathrm{self}}}{N},\quad
\frac{U_\mathrm{UF}^{\mathrm{cross}}}{N},\quad
\frac{\partial H_1/\partial\lambda}{N}.
\tag{104}
$$

由于

$$
\frac{\partial H_1}{\partial\lambda}
=U_\mathrm{UF}^{\mathrm{cross}},
\tag{105}
$$

Stage 1 最后一列应与 `U_uf_cross` 列相同。

Stage 2 每行：

$$
\lambda,\quad
\Delta\lambda,\quad
\frac{U_\mathrm{target}}{N},\quad
\frac{U_\mathrm{E}}{N},\quad
\frac{U_\mathrm{UF}^{\mathrm{self}}}{N},\quad
\frac{U_\mathrm{UF}^{\mathrm{cross}}}{N},\quad
\frac{U_\mathrm{aux}}{N},\quad
\frac{\partial H_2/\partial\lambda}{N}.
\tag{106}
$$

其中

$$
\frac{\partial H_2}{\partial\lambda}
=U_\mathrm{target}-U_\mathrm{aux}.
\tag{107}
$$

所有能量列均为 eV/atom。

### 3.45 `switch_func()` 与 `dswitch_func()`，行 1044-1057

代码使用与 GPUMD 其它 TI 系综一致的九次光滑切换函数。设归一化时间 $\tau\in[0,1]$，则

$$
S(\tau)
=126\tau^5
-420\tau^6
+540\tau^7
-315\tau^8
+70\tau^9 .
\tag{108}
$$

代码写法是

$$
S(\tau)
=
\left(
70\tau^4
-315\tau^3
+540\tau^2
-420\tau
+126
\right)\tau^5 .
\tag{109}
$$

其导数为

$$
\frac{dS}{d\tau}
=630\tau^4(1-\tau)^4.
\tag{110}
$$

代码中的 `dswitch_func()` 返回每个 MD step 的 $\Delta\lambda$：

$$
\Delta\lambda
=\frac{1}{t_\mathrm{switch}}
\frac{dS}{d\tau}.
\tag{111}
$$

这个函数在 $\tau=0$ 和 $\tau=1$ 处导数为 0，使切换起止时不会突变。

---

## 4. `uf_reference.cuh` 逐段解读

### 4.1 文件用途，行 1-21

`uf_reference.cuh` 是从已有 `ti_liquid` 逻辑中抽出的共享 UF 解析自由能工具。它不计算 MD 中的 UF 力；MD 中的 UF 势能和力由 `ensemble_ti_superionic.cu` 的 GPU kernel 计算。

这个文件只负责 self UF 的解析自由能：

$$
F_\mathrm{UF}^{\mathrm{self}}
=F_\mathrm{IG}+F_\mathrm{excess}.
\tag{112}
$$

它被 `ti_liquid` 和 `ti_superionic` 共同使用，避免两套代码维护两份相同的 UF 表。

### 4.2 `UFReferenceData`，行 24-28

```cpp
struct UFReferenceData
{
  const std::vector<double>& sum_spline;
  const std::vector<std::vector<double>>& spline;
};
```

该结构只是把两张表打包返回：

- `sum_spline`：预累计的分段积分值。
- `spline`：每段的四个系数。

二者共同用于计算无量纲 UF excess 自由能函数 $f_\mathrm{UF}(x,p)$。

### 4.3 `sum_spline*()` 表，行 30-193

文件提供五组累计积分表：

```cpp
sum_spline1()
sum_spline25()
sum_spline50()
sum_spline75()
sum_spline100()
```

它们分别对应

$$
p=1,\ 25,\ 50,\ 75,\ 100.
\tag{113}
$$

这些表给出了 $x_\mathrm{UF}$ 网格上的累计积分值。`get_uf_fe_for_pair()` 根据式 (48) 算出 $x_\mathrm{UF}$，然后选择对应区间，用这些累计值作为分段积分的基准。

### 4.4 `spline*()` 表，行 195-753

文件同样提供五组分段系数表：

```cpp
spline1()
spline25()
spline50()
spline75()
spline100()
```

每一行有四个系数，可记为

$$
(a,b,c,d).
\tag{114}
$$

在 `fe()` 中，这些系数用于从区间左端 $x_0$ 积分到当前 $x$。代码对应的分段表达式为

$$
f(x)
= f(x_0)
+\frac{a}{2}(x^2-x_0^2)
+b(x-x_0)
+(c-1)\ln\frac{x}{x_0}
-d\left(\frac{1}{x}-\frac{1}{x_0}\right).
\tag{115}
$$

因此这些表本质上提供了 UF excess 自由能在不同 $p$ 和不同 $x$ 区间上的高精度解析插值。

### 4.5 `fe()`，行 755-789

`fe()` 输入：

- `x`：式 (48) 的 $x_\mathrm{UF}$。
- `coef[4]`：当前分段的 $(a,b,c,d)$。
- `sum_spline`：当前 $p$ 的累计积分表。
- `index`：当前分段位置。

对于极小 $x<0.0025$，代码直接用

$$
f(x)
=\frac{a}{2}x^2+bx .
\tag{116}
$$

对于其它区间，代码先判断 $x$ 是否正好落在表格节点上。如果是，直接返回 `sum_spline[index-1]`，避免重复插值误差。否则用式 (98) 从左端点 $x_0$ 积分到 $x$。

若 $x\ge4$，代码返回表尾值 `sum_spline[index]`。这相当于使用已有表的最大范围近似。

### 4.6 `supports_p()`，行 791-794

```cpp
return p == 1 || p == 25 || p == 50 || p == 75 || p == 100;
```

这是 self UF 解析自由能的限制。只有这些 $p$ 有表。注意这不限制 cross UF，因为 cross UF 不需要解析自由能，只通过 Stage 1 和 Stage 2 数值积分处理。

### 4.7 `get_data()`，行 796-811

`get_data(p)` 根据 $p$ 返回对应的两张表：

$$
p=25
\Rightarrow
\{\texttt{sum\_spline25()},\texttt{spline25()}\}.
\tag{117}
$$

如果传入不支持的 $p$，直接报输入错误：

```text
Self UF p must be 1, 25, 50, 75, or 100.
```

---

## 5. 运行时完整数据流

### 5.1 用户命令到 GPU 数组

用户输入：

```text
ensemble ti_superionic_stage1 temp T tperiod tau tequil Neq tswitch Nsw press P \
  spring O kO Al kAl \
  uf H H 25 1.0 \
  uf O H 10 1.0 \
  uf Al H 10 1.0
```

会被转换成：

$$
k_i =
\begin{cases}
k_\mathrm O, & i\in O,\\
k_\mathrm{Al}, & i\in Al,\\
0, & i\in H,
\end{cases}
\tag{118}
$$

$$
\mathrm{kind}_{HH}=1,\quad
\mathrm{kind}_{OH}=2,\quad
\mathrm{kind}_{AlH}=2.
\tag{119}
$$

GPU kernel 后续只看到 per-atom $k_i$、per-type-pair $p,\sigma^2,\mathrm{kind}$，不再处理字符串。

### 5.2 每个 MD step 的势能和力

主势首先正常计算：

$$
U_\mathrm{target},\quad
\mathbf F_\mathrm{target}.
\tag{120}
$$

然后 `find_reference_forces()` 计算：

$$
U_\mathrm{E},\quad
U_\mathrm{UF}^{\mathrm{self}},\quad
U_\mathrm{UF}^{\mathrm{cross}},
\tag{121}
$$

$$
\mathbf F_\mathrm{E},\quad
\mathbf F_\mathrm{UF}^{\mathrm{self}},\quad
\mathbf F_\mathrm{UF}^{\mathrm{cross}}.
\tag{122}
$$

Stage 1 覆盖力：

$$
\mathbf F
=\mathbf F_\mathrm{E}
+\mathbf F_\mathrm{UF}^{\mathrm{self}}
+\lambda\mathbf F_\mathrm{UF}^{\mathrm{cross}}.
\tag{123}
$$

Stage 2 混合力：

$$
\mathbf F
=(1-\lambda)
\left(
\mathbf F_\mathrm{E}
+\mathbf F_\mathrm{UF}^{\mathrm{self}}
+\mathbf F_\mathrm{UF}^{\mathrm{cross}}
\right)
+\lambda\mathbf F_\mathrm{target}.
\tag{124}
$$

### 5.3 每个 stage 的积分量

Stage 1：

$$
\Delta F_1
=\int_0^1
\left\langle
U_\mathrm{UF}^{\mathrm{cross}}
\right\rangle_\lambda d\lambda.
\tag{125}
$$

Stage 2：

$$
\Delta F_2
=\int_0^1
\left\langle
U_\mathrm{target}
-U_\mathrm{E}
-U_\mathrm{UF}^{\mathrm{self}}
-U_\mathrm{UF}^{\mathrm{cross}}
\right\rangle_\lambda d\lambda.
\tag{126}
$$

非平衡正反向估计为：

$$
W_\mathrm{forward}
=\sum_{\mathrm{forward}}
\frac{1}{N}
\frac{\partial H}{\partial\lambda}\Delta\lambda,
\tag{127}
$$

$$
W_\mathrm{backward}
=\sum_{\mathrm{backward}}
\frac{1}{N}
\frac{\partial H}{\partial\lambda}\Delta\lambda,
\quad \Delta\lambda<0,
\tag{128}
$$

$$
\Delta F
=\frac{1}{2}
\left(
W_\mathrm{forward}
-W_\mathrm{backward}
\right).
\tag{129}
$$

### 5.4 参考自由能

参考自由能为

$$
F_\mathrm{ref}
=F_\mathrm{Einstein}
+\sum_s F_{\mathrm{UF},s}^{\mathrm{self}}.
\tag{130}
$$

Einstein 部分：

$$
F_\mathrm{Einstein}
=\frac{3k_\mathrm BT}{N}
\sum_{i\in S_\mathrm{spring}}
\ln\left(
\frac{\hbar}{k_\mathrm BT}
\sqrt{\frac{k_i}{m_i}}
\right).
\tag{131}
$$

每个 self UF 物种：

$$
F_{\mathrm{UF},s}^{\mathrm{self}}
=\frac{1}{N}
\left[
N_s k_\mathrm BT f_\mathrm{UF}(x_s,p_s)
+N_s k_\mathrm BT
\left(\ln(\rho_s\Lambda_s^3)-1\right)
\right].
\tag{132}
$$

其中

$$
x_s=\frac{(\pi\sigma_s^2)^{3/2}}{2V_s},
\quad
V_s=\frac{V_\mathrm{box}}{N_s},
\quad
\rho_s=\frac{1}{V_s},
\quad
\Lambda_s=\hbar\sqrt{\frac{2\pi}{m_s k_\mathrm BT}}.
\tag{133}
$$

最终

$$
F_\mathrm{target}
=F_\mathrm{ref}+\Delta F_1+\Delta F_2.
\tag{134}
$$

若需要 Gibbs 自由能，后处理使用

$$
G_\mathrm{target}
=F_\mathrm{target}+PV,
\tag{135}
$$

这里 $P$ 是 YAML 中的内部单位压力，$V$ 是每原子体积，所以 $PV$ 为 eV/atom。

---

## 6. 输出文件含义

### 6.1 Stage 1 CSV

`ti_superionic_stage1.csv`：

```text
lambda,dlambda,U_einstein,U_uf_self,U_uf_cross,dHdlambda
```

对应：

$$
\frac{\partial H}{\partial\lambda}=U_\mathrm{UF}^{\mathrm{cross}}.
\tag{136}
$$

该文件用于检查参考态到辅助态过程中 cross UF 排斥势的积分行为。

### 6.2 Stage 2 CSV

`ti_superionic_stage2.csv`：

```text
lambda,dlambda,U_target,U_einstein,U_uf_self,U_uf_cross,U_aux,dHdlambda
```

对应：

$$
U_\mathrm{aux}
=U_\mathrm{E}+U_\mathrm{UF}^{\mathrm{self}}
+U_\mathrm{UF}^{\mathrm{cross}},
\tag{137}
$$

$$
\frac{\partial H}{\partial\lambda}=U_\mathrm{target}-U_\mathrm{aux}.
\tag{138}
$$

该文件用于检查辅助态到真实目标态过程中，目标势和辅助势之间的能量差。

### 6.3 Stage YAML

每个 stage 的 YAML 记录：

- stage 编号。
- $T,V,P,N$。
- spring 物种和最终弹簧常数。
- UF self/cross pair。
- 正向、反向、平均后的 stage 积分值。
- $F_\mathrm{Einstein}$、$F_\mathrm{UF}^{\mathrm{self}}$、$F_\mathrm{ref}$。

它不直接写最终 $F_\mathrm{target}$ 和 $G_\mathrm{target}$，因为单个 stage 不知道另一个 stage 的 $\Delta F$。最终汇总由 `tools/si_free_energy_sum.py` 读取两个 YAML 后完成。

---

## 7. 重要实现限制和使用注意

1. `press` 只影响后处理的 $G=F+PV$，不参与力、速度、盒子或温控。

2. UF 不手动指定 cutoff。实际参与 UF 计算的 pair 由主势径向近邻表决定。因此主势的近邻范围必须覆盖你希望 UF 相互作用生效的距离范围。

3. self UF 的解析自由能只支持 $p=1,25,50,75,100$。cross UF 的 $p$ 可以是任意正数，因为它不走解析自由能表。

4. 至少需要一个 self UF pair。否则代码会报错，因为参考态流体自由能无法定义。

5. cross UF pair 不贡献 $F_\mathrm{ref}$。它们只通过 Stage 1 的 $\Delta F_1$ 和 Stage 2 的 $\Delta F_2$ 影响最终自由能。

6. `spring auto` 必须有 `tequil > 0`。自动弹簧常数来自初始平衡阶段的 MSD。

7. 初始 `position_0` 是进入系综时的坐标。Einstein 晶体参考位置不会在模拟中更新。

8. 对 Stage 1，主势目标力被完全覆盖；对 Stage 2，主势目标力与辅助力线性混合。

9. 代码按总原子数 $N$ 归一化所有输出自由能和功，因此不同组分的解析贡献也都以 eV/atom of whole system 表示。

10. 当前实现没有在 stage YAML 中给统计误差。若要误差，需要多副本运行后在后处理层统计。

---

## 8. 三个新增文件之间的关系

```text
ensemble_ti_superionic.cuh
  声明 Ensemble_TI_Superionic 类、stage 标志、UF pair、CPU/GPU 缓冲区和接口。

ensemble_ti_superionic.cu
  实现命令解析、参考态准备、GPU 势能/力 kernel、lambda 切换、功累加、CSV/YAML 输出。

uf_reference.cuh
  提供 UF self 解析自由能的样条表和插值函数，被 ti_superionic 和 ti_liquid 共享。
```

从物理上看：

```text
uf_reference.cuh
  只回答：给定 self UF 物种，F_UF_self 是多少？

ensemble_ti_superionic.cu
  回答：每个 step 的 U_E、U_UF_self、U_UF_cross、U_target、力和 dH/dlambda 是多少？

ensemble_ti_superionic.cuh
  定义：这些数据在 GPUMD ensemble 生命周期中如何保存和调用？
```

这三者合起来实现了：

$$
F_\mathrm{target}
=
\left(
F_\mathrm{Einstein}
+F_\mathrm{UF}^{\mathrm{self}}
\right)
+\int_\mathrm{stage1}
\left\langle
U_\mathrm{UF}^{\mathrm{cross}}
\right\rangle d\lambda
+\int_\mathrm{stage2}
\left\langle
U_\mathrm{target}
-U_\mathrm{aux}
\right\rangle d\lambda .
\tag{139}
$$
