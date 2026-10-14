# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

**S-Space** 的研究代码。S-Space 是多模态模型中间层激活里的一个三维线性子空间，用水平、垂直、距离三条轴（H/V/D）编码物体位置。`sspace/` 包负责两件事：
1. 在 COCO 上为每一层学习 H/V/D 轴；
2. 运行各类评测与分析实验：空间问答、提示词泛化、多视角、干预、视角转换、CoT 演化等。

涉及五个模型：`molmo2_er`、`molmoact2`、`molmoact2_pretrain`、`qwen35_4b`、`qwen36_27b`。实验设置以 JSON 形式放在 `configs/` 下。

## 环境与命令

所有脚本都面向装有 `uv` 的 Linux CUDA 主机。`./scripts/install.sh` 会创建三个相互隔离的虚拟环境（不同模型家族需要互不兼容的 `transformers` 版本），并下载模型：

| 虚拟环境 | Python | 用途 |
|---|---|---|
| `.envs/spatialmqa` | 3.10 | Molmo2-ER、数据准备、调度/启动脚本、测试、绘图（`requirements-molmo2.txt`；开发依赖 `requirements-dev.txt` 额外包含 ruff） |
| `.envs/molmoact` | 3.11 | MolmoAct2 与 MolmoAct2-Pretrain（`requirements-molmoact2.txt`） |
| `.envs/qwen36` | 3.11 | Qwen3.5-4B 与 Qwen3.6-27B（`requirements-qwen.txt`） |

启动脚本会自己选对解释器（见 `sspace/experiments/launch.py` 中的 `DEFAULT_MODEL_PYTHONS`），不要依赖系统 Python。直接调用模块时，在仓库根目录下设置 `PYTHONPATH=.`，并使用 spatialmqa 环境的解释器。

```bash
./scripts/prepare_data.sh                       # 下载/准备数据集 -> .cache/assets/、data/processed/
./scripts/run_smoke.sh                          # CPU 夹具 + 全部 unittest + 注册表校验
./scripts/run_experiment.sh --list              # 列出 suite、workflow、step
./scripts/run_experiment.sh --experiment NAME [--dry-run]
./scripts/run_experiment.sh --suite NAME [--model M] [--limit N] [--output-root DIR] [--dry-run]
./scripts/run_experiment.sh configs/experiments/.../X.json [--limit N] [--dry-run]
./scripts/run_experiment.sh --step NAME [args]  # 单个分析/绘图阶段（sspace/experiment_steps.py）
./scripts/run_all_dry.sh                        # 打印完整发布命令图，不实际执行
./scripts/reproduce_small.sh CONFIG [LIMIT]     # 小规模快速运行，输出到 outputs/small/limit_N
```

测试用标准库 `unittest`，只需 CPU：

```bash
PYTHONPATH=. .envs/spatialmqa/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=. .envs/spatialmqa/bin/python -m unittest tests.test_registry.RegistryTests.test_explicit_formal_experiment_contract
```

代码检查用 `ruff`（行宽 88，目标 py310，配置在 `pyproject.toml`）。

## 架构

**三层启动方式**（入口都是 `scripts/run_experiment.sh`）：
1. **单个 Config** → `sspace/experiments/launch.py`。`configs/experiments/` 下每个可运行的 JSON 都声明 `runner`、`model_adapter`、`suites`、`output_dir` 四个字段：
   - `runner` 通过 `RUNNER_MODULES` 映射到对应的 CLI 模块。`direct_generation` 是特例，分两步执行：先跑分片 worker，再跑合并。
   - `model_adapter` 决定使用哪个虚拟环境。
2. **Suite** → `sspace/experiments/scripts/run_suite.py`。suite 没有集中定义，而是由所有 config 的 `suites` 标签合并而成。config 通过 `rglob("*.json")` 自动发现，会跳过 `*.example.json` 和不含 `runner` 的文件。已完成的输出只有同时满足以下三个条件才会被跳过：
   - `run_manifest.json` 的状态为 `complete`；
   - `checksums.json` 校验一致；
   - `resolved_config.json` 中的 `launch_fingerprint` 与当前输入一致。
3. **Workflow / experiment** → `sspace/experiments/workflows.py`。配方放在 `configs/experiments/workflows/*.json`，内容是有序的 `steps`。每个 step 要么是 `module` + `model`（`model` 决定虚拟环境），要么是 `script`。`--all` 的执行顺序是：先构建 S-Space，再跑全部 benchmark config，最后跑每个 workflow。

**注册表是受校验的契约。** `configs/registry.json` 记录：
- 每个可运行 config 和 workflow 配方的 sha256；
- 预期的 config、命令、workflow 数量；
- `sspace/experiment_steps.py` 中的 `ENTRYPOINTS` 列表。

`python -m sspace.registry` 负责校验它。`run_smoke.sh`、`run_all_dry.sh` 以及每个 workflow 启动前都会执行这项校验。**新增或修改任何 config、配方或 step 入口后，都必须运行 `PYTHONPATH=. .envs/spatialmqa/bin/python -m sspace.registry --sync`**，否则启动时会报 "digest changed"。

**续跑身份。** `sspace/experiments/identity.py` 生成启动指纹，由以下几部分组成：
- config 本身；
- config 中声明的数据指纹；
- 模型资产标记文件（不哈希权重）；
- `sspace/` 下*所有* `.py` 文件的哈希（`run_records.python_source_identity`）。

因此，任何源码改动都会让此前已完成的输出在续跑判断中失效。

**S-Space 构建**（`sspace/core/`）：调用链为 `scripts/run_sspace.sh` → `sspace/core/scripts/run_coco6000.sh MODEL` → 分布式 worker（`core/cli.py`，使用 torchrun 风格的 `RANK`/`WORLD_SIZE`）。流程如下：
1. 每个 worker 使用冻结的 COCO 提示词（`core/prompts/templates.py`，轴顺序为 H/V/D），计算固定答案对比在最终 logit 上对各 block 输出状态的 Jacobian。
2. worker 把结果写成可续跑的分片（`core/extraction/`）。
3. rank 0 按全局样本顺序合并分片，在 COCO-1800 上选择读出层（`experiments/validation/coco1800/`），并发布 artifact。

训练配置在 `configs/sspace/train/`。

**Artifact**（`sspace/core/artifacts/artifacts.py` 中的 `SSpaceArtifact`）：不可变目录，包含 `manifest.json`、`checksums.json` 和 `tensors.safetensors`，后者存放每层 `[3, D]` 的单位轴。加载时会严格校验，要求 JSON 为规范格式：UTF-8、LF 换行、恰好一个结尾换行。每个 artifact 绑定到一个 `ModelSpec`。

已发布的轴位于 `sspace/core/artifacts/pretrained/*_coco6000_final_logit_axes_v2`，评测默认直接使用，重新构建是可选的。可用 `run_suite --artifact-dir MODEL=DIR` 指定其他目录。

**模型运行时**（`sspace/core/models/`）：`runtime.py` 分派到 `molmo.py` 或 `qwen.py`，负责提示词批处理、直接生成、多图提示和答案 token ID。`RUNTIME_SPECS` 固定了模型 revision、dtype 和注意力后端。

**实验**（`sspace/experiments/`）：
- `common/pairwise`：投影读出与 margin 符号判定，被 `benchmarks/` 下的 CV-Bench、EmbSpatial、SpatialTunnel 适配器复用。
- `common/direct`：原生答案生成。
- `multi_view/hstar`：HStar 多视角。
- `prompt/{where,color}`：提示词泛化。
- `spinbench/{perspective_taking,evolving_cot}`：视角转换与 CoT 演化。
- `intervention/spatial_causality`：空间干预。
- `analysis/`：上下文案例、MMSI 方向耦合、物体坐标绘图。
- `action_supervision/instructpart`：详见该目录 README。其伪真值仍为 `review_status=pending`，结果属于暂定。

**数据**（`sspace/data/<dataset>/`）：每个数据集都有一个 `prepare.sh`，由 `scripts/prepare_data.sh` 统一调用。config 中固定了 `expected_dataset_fingerprint`，运行时会校验。

## 约定

- 强制确定性：子进程会设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（`sspace/determinism.py`），若已预设为其他值则直接报错。随机种子写在 config 中。
- config 中的路径必须是项目内的相对路径，不能跳出项目目录（`sspace/config_paths.py`）。
- 实验输出写到 `outputs/`，临时文件写到 `.tmp/`，下载内容和模型权重写到 `.cache/assets/`。这些目录都在 `.gitignore` 中。
- JSON 写入使用 `run_records.atomic_json`。每次运行都会记录 manifest、checksum 和失败标记，中断后可以安全续跑。
