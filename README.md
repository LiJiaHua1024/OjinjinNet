# OjinjinNet

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v1.4.0-brightgreen.svg)](../../releases)
[![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX%20INT8-green.svg)](https://onnxruntime.ai/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

`OjinjinNet` 是一个面向中文社区评论区的轻量级端侧 AI 评论检测模型家族。所有模型均导出为纯 CPU ONNX INT8 格式（体积覆盖 `1.22 MB – 49.49 MB`），不依赖 PyTorch 或移动端 GPU，重点解决去掉常见关键词与模板口癖后的隐蔽 AI 评论识别问题，并支持 `Nano → Pro` 两级级联分诊。

> **需要 30–44 KB 零依赖纯 JS 方案（适用于油猴脚本或浏览器扩展）？**
> 配套零依赖蒸馏产物 [**OjinjinRules v2.1.0**](https://github.com/LiJiaHua1024/OjinjinRules)（`29.1 – 44.1 KB` 纯 ES5 JS / `9.9 – 15.7 KB` gzip，无 WASM / 无 ONNX / 无联网，`0.086 – 0.101 ms / 20 条`）在封存卷 `Bench-v3 Real` 上纯文本 `PR-AUC` 达 **`0.9447`**（高于上代 49.5 MB `Pro v1.3` 的 `0.9394`），平衡档精准率 **`99.0%`**、召回率 **`94.3%`**、高难真人 `0 / 307` 误杀，严格档 `0 / 2,881` 误杀。

### 主要特性

- **四档轻量纯 CPU ONNX INT8 推理**：提供 `1.22 MB`（Nano，多字哈希卷积 Static QDQ INT8）、`7.05 MB`（Lite KD）、`16.86 MB`（Standard KD）、`49.49 MB`（Pro SmoothQuant INT8）四档模型。开启 `--parallel-buckets` 后，`Nano v1.4` 在 PC 4 线程端到端单条 `0.29 ms`、整页 20 条分桶 `1.89 ms`。
- **纯文本与端到端跨代提升（`v1.4.0`）**：在封存双卷 `Bench-v3 Real (2,987)` 与 `Hard (1,434)` 上，`1.22 MB Nano v1.4` 纯文本 `PR-AUC` 达 `0.8890`（Hard，`+9.0 pp`）与 `0.9507`（Real），双卷均高于上代 49.5 MB `Pro v1.3`；四档 ONNX 在 `Hard` 卷上的默认阈值误杀条数由 `16` 条降至 `6–9` 条（下降 `44%–62.5%`）。
- **开箱即用的 `Nano v1.4 → Pro v1.4` 两级级联（`OjinjinCascadeClassifier`）**：`Nano` 先在 `1.89 ms` 内分诊绝大多数明确评论，在 `Bench-v3 Real` 上 `L2D-FP 10%` 档仅将 **`3.95%`** 边界评论转交 `Pro` 复核（20 条整页估算均值约 `8.1 ms`），在 `Bench-v3 Hard` 端到端综合 `PR-AUC` 达到 **`0.9810`**（`R@P95% 93.5%`，仅 `7` 条误杀）。
- **极简运行时与长度分桶**：Python 推理包仅依赖 `onnxruntime` 与 `numpy`，内置单遍 WordPiece 分词、8K 词片缓存、自动长度分桶批处理（`bucketed=True`）及 1 KB 元信息融合权重，便于移植至 Kotlin、Swift、Rust 或 JavaScript。

---

## 封存盲测实绩一览（`OjinjinBench-v3 Real & Hard`）

![五代同卷历史演进（v1.0 至 v1.4 与 Rules v1 至 v2.1）](docs/charts/v14/ojinjinnet_v14_card_5_history.png)

![纯文本赛道专项对比（Bench-v3 Hard 与 Real 双帕累托）](docs/charts/v14/ojinjinnet_v14_card_2_text_only.png)

![端到端融合专项对比（耗时帕累托、Hard 误杀对比与 Real 五个视频年龄段）](docs/charts/v14/ojinjinnet_v14_card_3_fusion.png)

> 交互式评测看板：[`docs/index.html`](docs/index.html)（支持切换 `Bench-v3 Real / Hard`、`Bench-v2`、`Bench-v1` 及历史基线横向对比），完整评测协议与局部 ROC 数据见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)。

---

## 模型规格与实测性能（`v1.4.0` 与配套 `Rules v2.1`）

ONNX 模型权重通过 [GitHub Releases](../../releases) 分发，每档压缩包包含 `model_int8.onnx`（Nano 另含 `model_cascade_int8.onnx`）、`vocab.txt`、`fusion_weights.json`、`manifest.json` 与 `NOTICE`，并附带 `SHA256SUMS.txt` 校验文件。

| 规格 | 底座 / 量化方式 | 体积 | PC 4 线程 (`--parallel-buckets`)<br>单条 / 整页 20 条 (p50) | 适用场景 |
|---|---|---:|---:|---|
| *配套 Tier 0：[`OjinjinRules v2.1`](https://github.com/LiJiaHua1024/OjinjinRules)* | *纯 ES5 JS 静态蒸馏规则（零依赖）* | ***29.1 – 44.1 KB*** | ***4.3–5.1 µs / 0.086–0.101 ms (V8)*** | *油猴脚本、浏览器扩展、无 WASM 同步拦截* |
| **`OjinjinNet-Nano v1.4`** | 多字哈希卷积 (Bigram+Trigram+DWConv)，Static QDQ INT8 | **1.22 MB** | **0.29 ms / 1.89 ms** | 极低体积、移动端与弹幕流、两级级联第一级 |
| **`OjinjinNet-Lite v1.4`** | MiniRBT-h256 (4L-256H, Super-Teacher KD)，Dynamic INT8 | 7.05 MB | 1.74 ms / 24.3 ms | 单文件轻量部署，`Real` 纯文本 (`0.9559`) 与综合 (`0.9960`) PR 最高 |
| **`OjinjinNet-Standard v1.4`** | BGE-small-zh-v1.5 (4L-512H, Super-Teacher KD)，Dynamic INT8 | 16.86 MB | 3.32 ms / 52.8 ms | 单文件均衡推荐，`Real` 开箱 1 误杀 (`99.0%` 精)，`Hard R@P95% 94.4%` |
| **`OjinjinNet-Pro v1.4`** | MacBERT-6L 剪枝 (6L-768H)，SmoothQuant INT8 | 49.49 MB | 8.30 ms / 157.4 ms | 单模型语义上限，`Hard` 纯文本 PR `0.9211`、`Real` 0 硬核误杀 |
| **`Nano → Pro v1.4` 级联** | `Nano` (L2D-FP 10%) + `3.95%` 转交 `Pro` | 1.22 + 49.49 MB | — / 估算均值 **~8.1 ms** | 服务端/高精准综合推荐，`Hard` 综合 PR `0.9810`（仅 7 误杀） |

> **实测说明**：统一最大序列长度 `max_len=192`（Head 140 + Tail 50），整页 20 条分桶采用 `ojinjinnet-speedbench` 确定性混合长度分布（`12×32 + 5×64 + 3×128`），在本地 PC（AMD Ryzen 5 5600X，4 线程 ONNX Runtime，`--parallel-buckets`）实测，串行与并行输出完全一致（`max_logit_diff = 0`）。移动端骁龙 865（Android 16 ARM64，4 线程）同架构整页分桶参考：Lite `56.4 ms`、Standard `106.4 ms`、Pro `314.2 ms`。

---

## 快速开始

1. 从 [GitHub Releases](../../releases) 下载任意规格的 `v1.4.0` 模型包（如 `ojinjinnet-nano-int8-v1.4.0.zip` 与 `ojinjinnet-pro-int8-v1.4.0.zip`），解压至 `./models/nano/`、`./models/pro/`。
2. 使用 `uv` 安装推理依赖（仅 `onnxruntime` 与 `numpy`）：

```bash
uv sync
```

3. 命令行直接推理或开启两级级联：

```bash
# 单模型推理
uv run ojinjinnet --model-dir ./models/standard \
  "这波属实难绷，看完了直接笑出声" \
  "不得不说这个视角确实很独特，不是简单的情绪宣泄，而是从底层逻辑剖析了问题本质，值得深思。"

# Nano -> Pro 两级级联（默认 L2D-FP 10% 档，真实分布仅 ~3.95% 转交 Pro）
uv run ojinjinnet --model-dir ./models/nano --cascade-strong-dir ./models/pro --parallel-buckets \
  "这波属实难绷" "怕的不是工具变强，是判断力变懒——这才是隐形成本。"

# 本机线程与并行分桶速度校验
uv run ojinjinnet-speedbench --model-dir ./models/nano --threads 1,2,4
```

4. Python API 调用：

```python
from ojinjinnet import OjinjinCascadeClassifier, OjinjinClassifier

# 单模型（如 1.22 MB Nano 或 16.86 MB Standard）
clf = OjinjinClassifier("./models/standard", num_threads=4, parallel_buckets=True)
scores = clf.predict_proba([
    "哈哈哈太真实了，笑死我了",
    "平心而论，这个观点看似犀利实则忽略了现实中的复杂博弈，让人看完颇有触动。",
])

# 两级级联（L2D-FP 防误杀分诊头，真实分布约 96% 由 1.22 MB Nano 直接处理）
cascade = OjinjinCascadeClassifier(
    fast_model_dir="./models/nano",
    strong_model_dir="./models/pro",
    strategy="l2d",       # 可选 "l2d" 或 "naive"
    target_budget=10,     # 预设转交预算：10 / 15 / 20 (%)
    parallel_buckets=True,
)
print(cascade.predict_detailed(["确实难绷", "怕的不是工具变强，是判断力变懒——这才是隐形成本。"]))
```

---

## 评测结果（封存 `Bench-v3 Real & Hard` 与历史延续）

### 1. 封存全实景双卷：`OjinjinBench-v3 Real (2,987)` & `Bench-v3 Hard (1,434)`

两套封存考卷与所有训练、蒸馏、主动学习语料在视频 `bvid` 与脱敏账号 `mid_hash` 上严格零重叠：
- **`Bench-v3 Real`（2,987 条，116 个视频）**：按实证推荐流配比抽样（首页推荐 + 相关视频 + 中性搜索，横跨 `≤14 天` 到 `>1 年` 全部 5 个视频年龄段），纯文本 AI `92` 条、综合 AI `106` 条（`3.55%` 低正例），高难真人 `307` 条、常规真人 `2,574` 条。
- **`Bench-v3 Hard`（1,434 条，127 个视频）**：抗饱和困难压测卷（含 `455` 条 `>1 年` 老视频边界样本），纯文本 AI `155` 条、综合 AI `214` 条，隐蔽 AI `159` 条、高难真人 `394` 条。

| 方案（导出产物实测） | 体积 | 20 条耗时 | `Real` 纯文<br>PR / F0.5 | `Real` 综合<br>PR / F0.5 (R@P95) | `Real` 开箱默认<br>精 / 召 / 误杀 | `Hard` 纯文<br>PR / F0.5 | `Hard` 综合<br>PR / R@P95 / 误杀 |
|---|---:|---:|---|---|---|---|---|
| **`OjinjinNet-Pro v1.4`** | 49.49 MB | 157.4 ms | `0.9535` / `0.9536` | `0.9944` / **`0.9825`** (`97.2%`) | 98.1% / 95.3% / 2 FP (0 HH) | **`0.9211`** / **`0.9155`** | `0.9802` / `92.1%` / 8 FP |
| **`OjinjinNet-Standard v1.4`** | 16.86 MB | 52.8 ms | `0.9524` / **`0.9596`** | `0.9950` / `0.9804` (**`99.1%`**) | **99.0%** / 94.3% / **1 FP (0 HH)** | `0.9164` / `0.9105` | `0.9802` / **`94.4%`** / 8 FP |
| **`OjinjinNet-Lite v1.4`** (KD) | 7.05 MB | 24.3 ms | **`0.9559`** / `0.9505` | **`0.9960`** / `0.9796` (**`99.1%`**) | 98.0% / 93.4% / 2 FP | `0.9004` / `0.8909` | `0.9771` / `90.2%` / 9 FP |
| **`OjinjinNet-Nano v1.4`** | **1.22 MB** | **1.89 ms** | `0.9507` / `0.9407` | `0.9956` / `0.9774` (`98.1%`) | 97.1% / **96.2%** / 3 FP | `0.8890` / `0.8838` | `0.9783` / `93.0%` / **6 FP** |
| **`Nano → Pro v1.4`（L2D 10%）** | 1.2+49.5 MB | ~8.1 ms | `0.9506` / `0.9587` | `0.9940` / `0.9783` (`97.2%`) | 98.0% / 94.3% / 2 FP (`3.95%` 转交) | **`0.9217`** / **`0.9155`** | **`0.9810`** / `93.5%` / 7 FP |
| *配套：`OjinjinRules v2.1 Pro`* | *44.1 KB JS* | *0.101 ms* | *0.9447 / 0.9311* | *0.9936 / 0.9839 (97.2%)* | *99.0% / 94.3% / 1 FP (0 HH)*<br>*严格档 100% / 90.6% / 0 FP* | *0.8768 / 0.8881* | *0.9665 / 88.8% / 8 FP (3 HH)* |
| *配套：`OjinjinRules v2.1 Compact`* | *38.9 KB JS* | *0.090 ms* | *0.9101 / 0.8951* | *0.9873 / 0.9749 (97.2%)* | *97.1% / 95.3% / 3 FP* | *0.8243 / 0.8656* | *0.9599 / 85.5% / 8 FP (3 HH)* |
| *配套：`OjinjinRules v2.1 Lite` (0 树)* | *29.1 KB JS* | *0.086 ms* | *0.9157 / 0.8995* | *0.9882 / 0.9661 (94.3%)* | *96.2% / 94.3% / 4 FP* | *0.8250 / 0.8534* | *0.9677 / 86.9% / 6 FP (2 HH)* |
| *上代对照：`OjinjinNet-Pro v1.3`* | *49.49 MB* | *157.4 ms* | *0.9394 / 0.9375* | *0.9944 / 0.9717 (97.2%)* | *97.1% / 95.3% / 3 FP* | *0.8883 / 0.8756* | *0.9710 / 86.9% / 16 FP* |
| *上代对照：`OjinjinNet-Std v1.3`* | *16.86 MB* | *52.8 ms* | *0.9376 / 0.9439* | *0.9923 / 0.9674 (96.2%)* | *96.2% / 96.2% / 4 FP* | *0.8732 / 0.8569* | *0.9698 / 85.1% / 16 FP* |
| *上代对照：`OjinjinNet-Nano v1.3`* | *1.22 MB* | *1.89 ms* | *0.9130 / 0.9102* | *0.9901 / 0.9644 (98.1%)* | *95.4% / 98.1% / 5 FP* | *0.7993 / 0.8171* | *0.9621 / 85.5% / 16 FP* |
| *上代对照：`OjinjinRules v2.0 Pro`* | *43.8 KB JS* | *0.072 ms* | *0.9278 / 0.9245* | *0.9915 / 0.9861 (96.2%)* | *100% / 93.4% / 0 FP (thr=0.62)* | *0.8262 / 0.8522* | *0.9604 / 83.2% / 10 FP (6 HH)* |

### 2. 五代历史同卷演进（`v1.0 → v1.4` 全部归档权重统一在封存 `Bench-v3` 重跑）

| 规格 | `Bench-v3 Hard` 纯文本 PR-AUC (`v1.0 → v1.4`) | `Bench-v3 Real` 纯文本 PR-AUC (`v1.0 → v1.4`) | `Bench-v3 Hard` 开箱误杀条数 (`v1.0 → v1.4`) |
|---|---|---|---|
| **Pro (49.5 MB)** | `0.8455 → 0.8575 → 0.8673 → 0.8883 → 0.9211` | `0.9054 → 0.9118 → 0.9337 → 0.9394 → 0.9535` | `16 → 16 → 12 → 16 → 8` |
| **Standard (16.9 MB)** | `0.7537 → 0.7617 → 0.7070 → 0.8732 → 0.9164` | `0.9014 → 0.9068 → 0.9059 → 0.9376 → 0.9524` | `20 → 20 → 41 → 16 → 8` |
| **Lite (7.05 MB)** | `0.6816 → 0.7711 → 0.8132 → 0.8379 → 0.9004` | `0.8325 → 0.9079 → 0.9216 → 0.9233 → 0.9559` | `44 → 24 → 24 → 16 → 9` |
| **Nano (1.22 MB)** | `— → — → 0.7459 → 0.7993 → 0.8890` | `— → — → 0.8955 → 0.9130 → 0.9507` | `— → — → 20 → 16 → 6` |
| **Rules Pro (44 KB JS)** | `— → — → 0.7988 (v1) → 0.8262 (v2.0) → 0.8768 (v2.1)` | `— → — → 0.9009 → 0.9278 → 0.9447` | `— → — → 4 → 10 → 8` |

### 3. 历史考卷与全市场公开基线对照

在冻结的 `OjinjinBench-v2`（1,476 条）与 `OjinjinBench-v1`（1,310 条）上与 8 款 HuggingFace 开源中文/多语言 AI 文本检测器（`390 MB – 1.06 GB`）、冻结 `bge-small-zh-v1.5` 向量检索方案、2 款零样本端侧小大模型（`Qwen2.5-0.5B` / `Qwen3-0.6B`）及 12 款匿名社区规则插件 A–L 的完整明细见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md) 与交互看板 [`docs/index.html#arena`](docs/index.html#arena)。

---

## 技术实现要点

1. **防误杀负样本扩充与多教师蒸馏（`v1.4.0`）**：在自然推荐流与难负真人语料上进一步扩展高质量负例与边界样本，并结合 12 层教师与多架构集成软目标对 `Lite / Standard / Nano` 与 `OjinjinRules v2.1` 进行统一蒸馏，同时提升纯文本语义辨别力并显著降低高难真人误杀。
2. **SmoothQuant 修复 768 维中文 Encoder 的 INT8 精度损失**：在导出 `MacBERT-6L` 阶段插入等价通道缩放（`α=0.80`，权重增量约 `60 KB`），缓解 FFN 输出 `18×` 离群值导致的量化退化，`v1.4 Pro` INT8 与 FP32 相关系数达 `0.9989`。
3. **词表按频裁剪与 Head-Tail 截断**：词表裁剪至 `4,096 / 8,192 / 10,000`，超长评论保留前 140 与后 50 token，避免丢失开头观点或结尾套话。
4. **单遍分词与自动长度分桶并行推理**：内置 8K 词片 LRU 缓存与 `≤32 / ≤64 / ≤128 / ≤192` token 分组并行执行（`--parallel-buckets`），数值输出与串行零误差（`max_logit_diff = 0`）。

---

## 版本更新记录

### v1.4.0

全系四档 ONNX INT8、`Nano → Pro` 两级级联与配套纯 ES5 JS 引擎 [**OjinjinRules v2.1.0**](https://github.com/LiJiaHua1024/OjinjinRules) 同步升级，全部成绩在封存双卷 `OjinjinBench-v3 Real (2,987)` 与 `Hard (1,434)` 上验证：

- **纯文本跨代提升**：`Pro v1.4` 在 `Hard` 纯文本 `PR-AUC` 升至 `0.9211`（`+3.3 pp`）；`Standard v1.4` 升至 `0.9164`（`+4.3 pp`）；`Lite v1.4` 升至 `0.9004`（`+6.3 pp`），并在 `Real` 纯文本取得全系最高 `0.9559`；`1.22 MB Nano v1.4` 在 `Hard` 升至 `0.8890`（`+9.0 pp`）、`Real` 升至 `0.9507`（`+3.8 pp`），双卷均超过上代 49.5 MB `Pro v1.3`。
- **Hard 误杀减半与 Real 零硬核误杀**：四档 ONNX 在 `Bench-v3 Hard` 开箱默认阈值下的误杀数由 `v1.3` 的 `16` 条全部降至 `6–9` 条（`Nano 6`、`Std / Pro 8`、`Lite 9`）；`Standard v1.4` 在 `Bench-v3 Real` 开箱精准率 `99.01%`（仅 `1` 条误杀，近 1 年前 4 个视频年龄段全 0 误杀，`307` 条高难真人 `0` 误杀）；`Pro v1.4` 同样实现 `0` 硬核误杀。
- **两级级联与纯 JS 引擎同步增强**：`Nano → Pro v1.4 (L2D 10%)` 在 `Hard` 端到端综合 `PR-AUC` 达到 `0.9810`（`R@P95% 93.5%`，仅 `7` 误杀），`Real` 仅 `3.95%` 样本转交 `Pro`；配套 44.1 KB `OjinjinRules v2.1 Pro` 的 `Real` 纯文本 `PR-AUC` 升至 `0.9447`、`Hard` 纯文本 `PR-AUC` 升至 `0.8768`（`+5.1 pp`）、`Hard R@P99%` 由 `50.0%` 升至 `77.6%`。

### v1.3.0

- 完成 67,612 条实证推荐流抓取与第 3 轮委员会主动学习（`AL-R3`，新增 2,297 条高分歧金标、清理 369 条灰区），引入反稀释全量重训并在封存双卷 `OjinjinBench-v3 Real (2,987)` + `Hard (1,434)` 上验证；同步发布 `OjinjinRules v2.0`。

### v1.2.0

- 引入第 2 轮委员会主动学习（`AL-R2`，1,354 条）与 `Bench-v2`（1,476 条）时间外盲测；新增 **1.22 MB `OjinjinNet-Nano`**、**`OjinjinCascadeClassifier`** 两级级联与 `--parallel-buckets` 并行分桶加速。

### v1.1.0

- 引入第 1 轮委员会主动学习（840 条）与自动长度分桶批处理（`bucketed=True`），`Lite` `Hard-Only R@P95%` 提升 `+15.6%`。

### v1.0.0

- 首次发布 `Lite / Standard / Pro` 三档纯 CPU ONNX INT8 模型权重、零 PyTorch 推理包与 `OjinjinBench-v1` 评测规范。

---

## 数据隐私与合规说明

为尊重社区用户隐私、原始 UGC 权益与平台内容规范，本仓库**不公开发布原始抓取评论文本及含可识别用户身份信息的训练集与评测集**（详见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)）。仓库完整公开模型推理代码、评测指标脚本、方法论文档与 ONNX INT8 权重。

---

## 开源协议与致谢

- 本项目代码与衍生权重采用 [Apache License 2.0](LICENSE) 开源，允许个人、开源项目与商业应用集成与二次开发。
- 致谢以下中文开源预训练模型（详见 [`NOTICE`](NOTICE)）：
  - 哈工大讯飞联合实验室（HFL）：[MiniRBT](https://github.com/ymcui/MiniRBT)、[MacBERT](https://github.com/ymcui/MacBERT)、[Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA)（Apache-2.0）
  - 智源研究院（BAAI）：[BGE-small-zh-v1.5](https://github.com/FlagOpen/FlagEmbedding)（MIT）
