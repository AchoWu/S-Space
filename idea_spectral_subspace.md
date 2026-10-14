# Idea：把 S-Space 从"三根独立的轴"改为"谱分解得到的子空间"

> 状态：想法记录，尚未实现，也未经实验验证。文中的预期结果都是假设。

## 1. 动机

现在的做法（`sspace/core/extraction/execution.py`、`merge.py`）是：每个样本只对它所属轴的答案对比 z_g 求导，三个轴各自把反对称梯度 a = (∂z_g/∂h_q − ∂z_g/∂h_r)/2 取平均，再归一化，分别得到 v_H、v_V、v_D。

这里隐含了两个预设：

1. **子空间恰好是三维的**，而且三根轴各自独立。
2. **"平均梯度"就是最合适的估计**。但它不告诉我们样本之间方向是否一致，也不告诉我们是否还有其他方向在起作用。

可以从已发布的 artifact 中找到证据：在选中的层上，三根轴两两之间的余弦相似度如下（作为参照，2560 维空间里两个随机方向的余弦，标准差约为 0.02）：

| 模型 | 层 | cos(H,V) | cos(H,D) | cos(V,D) |
|---|---|---|---|---|
| Molmo2-ER | L22 | 0.048 | −0.057 | **0.140** |
| MolmoAct2-Pretrain | L18 | −0.016 | 0.011 | **0.264** |
| MolmoAct2 | L19 | 0.034 | 0.040 | **0.231** |
| Qwen3.5-4B | L15 | 0.027 | 0.128 | −0.102 |
| Qwen3.6-27B | L43 | −0.001 | −0.015 | 0.065 |

Molmo 系列中 V 与 D 明显耦合：画面越靠下，越被判为越近，这符合地平面透视先验。而且经过动作训练后，耦合比 Molmo2-ER 更强。把这些轴当作三根独立的轴分别估计，就掩盖了这种结构。

## 2. 方法

### 2.1 把逐样本的反对称梯度堆成矩阵

对每个样本 p、问法 t、方向 o（original/swapped）、问题轴 g，取第 l 层的反对称梯度：

```
a_{p,t,o,g} = ( ∂z_g/∂h_q − ∂z_g/∂h_r ) / 2   ∈ R^D
```

把所有 a 按行堆叠，得到 A ∈ R^{N×D}。以全量数据为例，N = 6000 个样本 × 5 种问法 × 2 个方向 = 60000。

### 2.2 用非中心化的 SVD 求子空间

```
A = U Σ Wᵀ,   S-Space_k = span(W[:, :k])
```

- **必须用非中心化的二阶矩**（AᵀA/N），不能做常规 PCA。我们要的信号恰好就是平均方向，中心化会把它减掉。
- 从奇异值谱 σ_1 ≥ σ_2 ≥ … 可以直接判断维度：如果 σ_3/σ_4 处出现明显断崖，"S-Space 是三维的"就由假设变成了结论。如果谱呈现 2+1 的结构（例如 D 明显更弱）或者其他形态，本身就是一个发现。
- 实现上，D = 5120，N = 60000，可以直接对 D×D 的矩阵 AᵀA 做特征分解，也可以用 randomized SVD，计算开销都不大。

> **修正（2026-10-14）**：上面的做法有一个缺陷。AᵀA/N = C_within + Σ_g π_g m_g m_gᵀ，其中样本间的波动 C_within 不会随 N 增大而减小。如果单个样本的梯度噪声（例如物体类别本身带来的方向）比平均方向更强，前三个奇异向量捕获的就是噪声，而不是 S-Space。在合成数据上已复现这个失败模式：均值能量只占 2% 时，投影后的轴与真实轴的余弦只有约 0.1。改进做法见 2.5 节。

### 2.5 改进：用 original × swapped 的交叉协方差找"角色一致"的方向

对每个样本 p，把 5 种问法平均，得到两个向量：P_orig[p]（原始顺序）和 P_swap[p]（交换顺序）。交换 query 与 reference 时：

- **空间信号**符号不变：两次求出的都是"把 query 往右推"的方向；
- **与角色绑定的成分**（例如物体类别）符号翻转：a = (g_q − g_r)/2，交换后 q 和 r 互换；
- **措辞噪声**两次相互独立。

因此交叉协方差 X = (P_origᵀP_swap + P_swapᵀP_orig)/2 只在角色一致的方向上为正，噪声方向期望为 0，角色绑定的方向为负。

- **维度检验（cvPCA）**：在一半样本上求 X 的特征向量，在另一半上计算 mean_p (P_orig[p]·v)(P_swap[p]·v)。真实维度得分为正，其余接近 0。这样不需要零假设基线，噪声底自然就是 0。
- **轴（cross_svd）**：取全量 X 的前三个特征向量张成子空间，再把 m_g 投影进去。

合成数据中植入了三个方向和很强的物体类别噪声。结果是：交叉协方差谱恰好有 3 个正值，之后约为 0；`cross_svd` 与真实轴的余弦约为 0.96，而 `raw_svd` 只有约 0.1。

### 2.3 在子空间内恢复有名字的轴

SVD 得到的奇异向量只确定到一个旋转，本身没有"水平 / 垂直 / 远近"的含义。有两种方式给它们重新赋予名字：

- **投影法（推荐，简单）**：把各轴原来的平均梯度 m_g 投影到子空间里再归一化，ṽ_g = unit(P_S m_g)。这样既保留了可解释的轴名，又借助子空间去掉了噪声。

  > **注意**：当 k=3 且子空间估计得好时，S 基本就等于 span(m_H, m_V, m_D)，ṽ_g ≈ v_g，读出准确率不会有区别。所以准确率上的提升只可能来自对偶基读出、白化，或者 k≠3 的情况。谱分解的主要价值在于给出维度证据。
- **Procrustes 对齐**：在 S 内寻找一个旋转，使其与 {m_H, m_V, m_D} 最接近。

如果轴之间不正交，读出坐标时应使用对偶基：c = (ṼᵀṼ)⁻¹ Ṽᵀ h。这和论文附录 C.3 中干预所用的 B = V(VᵀV)⁻¹ 是同一套几何。

### 2.4 变体：最大化信噪比（广义特征值问题）

把轴之间的差异当作信号，把样本之间的波动当作噪声：

```
M_signal = Σ_g m_g m_gᵀ,   C_noise = Cov(a)
求解  M_signal w = λ C_noise w
```

这相当于在梯度空间里做 LDA，能自动把残差流的各向异性（少数离群的大值维度）白化掉，也和"读出方向 vs 编码方向"的问题直接相关。

## 3. 需要配套的统计检验

- **折半稳定性**：把样本随机分成两半，分别求出子空间，计算两者之间的主角度（principal angles）。主角度越小，说明子空间越稳定、越可信。
- **方向一致性**：‖mean(a)‖ / mean(‖a‖)。如果所有样本都指向同一个方向，这个值接近 1；如果只是噪声平均后的残余，它约等于 1/√N。可以按层、按轴分别报告，也可以作为一种不需要标签的选层依据。
- **零假设基线**：~~给每个 a 随机翻转正负号后重复整个流程~~。**修正**：非中心化二阶矩 Σ a aᵀ 对每一行的正负号不敏感，翻转符号后谱完全不变，所以这个基线对谱无效。符号翻转只适用于依赖均值的统计量（例如方向一致性）。谱的维度检验改用 2.5 节的 cvPCA。

## 4. 实验计划

| 阶段 | 内容 | 成本 |
|---|---|---|
| Phase 0 | 直接用现有的 shard（`shards/rank_*/role_gradients.npy`，形状 `[J,2,L,D]`）。对单个轴 A_g 和合并后的 A 分别做 SVD，计算谱、方向一致性和折半稳定性 | 不需要 GPU，也不用重新抽取；只需要完整的训练输出（当前仓库里只有最终的 artifact） |
| Phase 1 | 用 ṽ_g 替换 v_g，在 COCO-1800 和各个 benchmark 上比较读出准确率，并比较干预的效果 | 只需前向 |
| Phase 2 | 对每对物体同时提问 H、V、D 三个问题，得到 3×D 的 Jacobian 块 J_p，在这个联合矩阵上做 SVD | 抽取成本约为原来的 3 倍 |
| Phase 3 | 跨模型比较：Molmo2-ER → MolmoAct2-Pretrain → MolmoAct2 三者共享同一个隐藏空间，可以比较它们子空间之间的主角度，以及 V–D 耦合的变化，从而量化"动作训练如何改变空间几何"。另外比较同一模型相邻各层之间的子空间主角度，观察子空间在层间的演化 | 只需分析 |

关于 Phase 2：构建梯度本来就不需要标签，所以任何一对物体都可以同时用来问三个问题，不受样本原本所属轴的限制。

## 5. 预期结论（假设）

- 谱在第 3 个奇异值之后出现断崖，为"三维空间工作区"提供由数据驱动的证据，而不再依赖构造上的假设。
- Molmo 系列中 V 与 D 共享一部分主方向，耦合程度随动作训练增强。这可以作为论文中 action supervision 一节的几何解释。
- 子空间投影去噪后，读出准确率与原方法持平或略有提升，跨提示词的稳定性更好。

## 6. 在本仓库中的实现要点

- **抽取**：现在的 `LayerwiseFixedContrastGradient.extract` 只对 `task_group` 求导。Phase 2 需要拿到三个对比各自的梯度，可以用 `retain_graph=True` 连续做三次 `autograd.grad`，也可以设置 `is_grads_batched=True`、并令 `grad_outputs = I_3`，一次性批量求出。对 27B 模型，前者更省显存。
- **提示词**：每个样本需要对三个轴分别调用一次 `render_prompt_pair`，提示词数量从 10 条变为 30 条。
- **分析模块**：建议新建 `sspace/experiments/analysis/spectral_subspace/`，并在 `sspace/experiment_steps.py` 的 `ENTRYPOINTS` 中注册为一个 step。
- **Artifact**：保持 `[L,3,D]` 的格式不变，存放 ṽ_g，这样下游代码不需要任何修改。新增的奇异值谱和诊断量写入 manifest。注意 `artifacts.py` 对 `method` 和 `construction` 字段做了严格校验，需要相应扩展。
- **流程约束**：改动任何 `.py` 文件，都会让续跑判断认为已有输出失效（启动指纹包含源码哈希）。新增 config 或 step 之后，要运行 `python -m sspace.registry --sync`。
- **存储**：Phase 0 需要逐样本的梯度。Qwen3.6-27B 的完整 shard 约 236 GB，可以只读取少数几个候选层（例如选中层前后各几层），以 memmap 方式按层切片读取。

## 7. 验证方案：怎样才算"优于"

**参赛的几组方法**（都由 `spectral_subspace/analyze.py` 生成）：

| 轴 | 含义 |
|---|---|
| `mean` | 当前方法，作为基线 |
| `raw_svd` | 逐提示词梯度的非中心化 SVD，取前 3 维子空间后投影（2.2 节原始想法） |
| `sample_svd` | 同上，但先把每个样本的 5 种问法平均 |
| `cross_svd` | 基于 orig × swap 交叉协方差的前 3 维子空间（2.5 节） |
| `whiten_a{α}` | (C + α·tr(C)/D·I)⁻¹ m_g，即无需标签的 LDA（2.4 节） |

每组轴各配两种读出方式：投影 vᵀΔh，以及对偶基 [(VVᵀ)⁻¹VΔh]_g。其中 `mean/dual` 单独检验"提升是否只来自读出几何"。

**评测指标**：
1. **COCO-1800 读出准确率**：
   - 选层准确率，沿用原规则：先比总体准确率，再比最差轴，再取较浅的层；
   - 在基线所选层上的准确率；
   - 交叉拟合准确率：在一半验证集上选层、在另一半上测试，消除选层带来的偏差；
   - 交叉拟合准确率相对基线的差值，用配对 bootstrap 给出 95% CI。
2. **样本效率**：每轴只用 n 个样本建轴时的准确率曲线。
3. **诊断**：
   - 方向一致性；
   - 均值能量占比；
   - 交叉协方差谱，即维度证据；
   - 折半稳定性；
   - 与 merged 轴、与发布轴的余弦，以及验证集 margin 的复现误差。这两项都是流程正确性检查。
4. **后续**（Phase 0 有正面信号后再做）：
   - 把胜出的轴打包成 artifact，用 `run_suite --artifact-dir` 跑 CV-Bench、EmbSpatial、SpatialTunnel 和提示词泛化；
   - 在 `spatial_causality` 上比较目标轴的翻转率与跨轴串扰。

**判定规则（预先写定）**：满足下面任意一条即判为"优于"：
- 交叉拟合准确率提升 ≥ 1 个百分点，95% CI 不含 0，且最差轴不下降超过 0.5 个百分点；
- 准确率持平，但只用不超过 25% 的样本就达到基线水平；
- 后续的干预实验中串扰显著降低。

否则，结论写为"均值方法已经足够好，谱分析作为维度证据放进附录"。

## 8. Qwen3.5-4B 执行步骤（Linux GPU 服务器）

新增的文件：
- `configs/sspace/train/pilot_qwen35_4b_spectral.json`：试跑配置。按 (轴, 端点) 分层，用 seed 42 抽样；训练集每端点 100 个样本，共 600 个；验证集每端点 50 个，共 300 个。这个目录不受 registry 管理，所以无需 `--sync`。
- `spectral_subspace/dump_validation_states.py`：把验证集上每层的 h_q − h_r 一次性存下来，之后各组轴都能离线评测。
- `spectral_subspace/analyze.py`：纯 NumPy 的 Phase 0 分析，输出 `report.json`、`summary.md` 和各组的 `axes_*.npz`。

两个脚本都放在 `sspace/` 之外，不会改变源码指纹。所以正式输出的续跑判断不受影响。

```bash
# 0. 环境：只装需要的两个
export TMPDIR=$PWD/.tmp UV_CACHE_DIR=$PWD/.cache/uv \
       UV_PYTHON_INSTALL_DIR=$PWD/.cache/uv/python HF_HOME=$PWD/.cache/huggingface
mkdir -p "$TMPDIR"
uv venv --python 3.10 .envs/spatialmqa
uv pip install --python .envs/spatialmqa/bin/python -r requirements-molmo2.txt
uv venv --python 3.11 .envs/qwen36
uv pip install --python .envs/qwen36/bin/python -r requirements-qwen.txt

# 1. 模型与数据：只需要 Qwen3.5-4B 和 COCO
./scripts/prepare_models.sh qwen35_4b
./sspace/data/coco/prepare.sh

# 2. 试跑（约 600 个样本）
./sspace/core/scripts/run_coco6000.sh qwen35_4b \
    configs/sspace/train/pilot_qwen35_4b_spectral.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
    .envs/qwen36/bin/python spectral_subspace/dump_validation_states.py \
    --config configs/sspace/train/pilot_qwen35_4b_spectral.json \
    --output outputs/spectral_pilot/qwen35_4b_validation_states
PYTHONPATH=. .envs/spatialmqa/bin/python spectral_subspace/analyze.py \
    --run-dir outputs/spectral_pilot/qwen35_4b_coco600_final_logit_axes \
    --validation-states outputs/spectral_pilot/qwen35_4b_validation_states \
    --released-artifact sspace/core/artifacts/pretrained/qwen35_4b_coco6000_final_logit_axes_v2 \
    --out outputs/spectral_pilot/qwen35_4b_analysis

# 3. 全量：试跑通过后再跑
./sspace/core/scripts/run_coco6000.sh qwen35_4b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
    .envs/qwen36/bin/python spectral_subspace/dump_validation_states.py \
    --config configs/sspace/train/train_qwen35_4b_coco6000.json \
    --output outputs/spectral/qwen35_4b_validation_states
PYTHONPATH=. .envs/spatialmqa/bin/python spectral_subspace/analyze.py \
    --run-dir outputs/release_training/qwen35_4b_coco6000_final_logit_axes_v2 \
    --validation-states outputs/spectral/qwen35_4b_validation_states \
    --released-artifact sspace/core/artifacts/pretrained/qwen35_4b_coco6000_final_logit_axes_v2 \
    --efficiency-sizes 10,25,50,100,250,500,1000 \
    --out outputs/spectral/qwen35_4b_analysis
```

- **多卡**：先复制一份配置，把 `world_size` 改成 GPU 数。merge 的结果与卡数无关。
- **磁盘**：试跑约 7 GB。全量约 65 GB，其中 role_gradients 约 20 GB、object_states 约 39 GB、验证集 dump 约 6 GB。

**试跑后先看这几项**（判断流程是否可信，不看准确率）：
1. 显存是否够用，单个 job 用时多少，据此估算全量的时间。全量是试跑的 10 倍。
2. `summary.md` 中的"复现误差"应远小于"margin 量级"。这说明 dump 出的表示与训练时验证阶段的结果一致。
3. L15 上"与发布轴余弦"应当较高，说明用 600 个样本建出的轴与发布的轴方向一致。
4. 试跑每轴只有 200 个样本，准确率对比不作结论，结论以全量结果为准。
