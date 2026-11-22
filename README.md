# OjinjinNet 🐋🕸️

> **面向中文社区评论区的超轻量端侧 AI 评论（「大肥鱼 / 哦鲸鲸」）检测模型家族与评测基准**  
> 纯 CPU ONNX INT8 推理 · 7 MB ~ 49 MB · 零 GPU 争抢 · 专抓去八股隐蔽 AI 评论

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX%20INT8-green.svg)](https://onnxruntime.ai/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

---

## 为什么做 OjinjinNet？

中文视频与内容社区评论区近年来涌现出大量自动化 AI 养号评论（俗称「大肥鱼 / 哦鲸鲸」）。它们早已从早期带明显口癖/关键词的模板，演化为**去掉一切关键词、甚至删掉标点符号，仅保留 AI 典型论证节奏与工整语体**的隐蔽变体——传统正则规则对这类评论的召回率直接归零（`0.0%`）。

另一方面，在手机或浏览器端部署 `0.5B–0.6B` 生成式小语言模型（量化后仍需 `350–614 MB`，且常需抢占移动端 GPU）代价过高。

**OjinjinNet** 证明：通过**双向 Encoder 架构 + 两级教师集成蒸馏 + 词表裁剪 + SmoothQuant 通道平滑无损 INT8 量化**，仅需 **7 MB ~ 49 MB 纯 CPU ONNX 模型**，就能在静默过滤最严苛的 `Precision ≥ 99%` 条件下实现高达 **`83.2%` 纯文本召回率**（端到端融合最高 **`96.8%` @ P95% / `93.8%` 隐蔽 AI 召回**），体积比 0.5B 端侧模型小 **10 ~ 50 倍**，可直接嵌入浏览器扩展、移动端 App、第三方客户端或后端风控服务。

---

## 模型矩阵（v1.0.0）

权重文件通过 **GitHub Releases** 分发（不塞入 Git 提交历史），每档压缩包均包含 `model_int8.onnx`、`vocab.txt`、`fusion_weights.json` 与 `manifest.json`，附带 `SHA256SUMS.txt` 校验：

| 规格 | 底座 / 工艺 | 裁剪后参数 | INT8 体积 | PC 4核 CPU<br>单条 / 20条 | 适用场景 |
|---|---|---:|---:|---:|---|
| **`OjinjinNet-Lite`** | MiniRBT-h256 (6L-256H) · Dynamic INT8 | 7.03M | **7.05 MB** | 2.5 ms / 35 ms | 低端机 / 浏览器极速省电模式（推荐配合轻量元信息融合） |
| **`OjinjinNet-Standard`** | BGE-small-zh-v1.5 (4L-512H) · Dynamic INT8 | 17.33M | **16.86 MB** | 4.4 ms / 75 ms | **默认主力推荐**：体量、速度、纯文本与融合精度最佳平衡点 |
| **`OjinjinNet-Pro`** | MacBERT-6L 剪枝蒸馏 (6L-768H) · **SmoothQuant INT8** | 51.19M | **49.49 MB** | 13.1 ms / 240 ms | **旗舰高精档**：不依赖任何账号元信息，纯文本裸考严控误杀仍达 `83.2%` |

> 注：耗时基准在 PC 4 线程 `CPUExecutionProvider`（`seq_len=64`）测得，动态序列轴支持按批次自适应长度（最大 `192` token），短评论无冗余计算。

---

## OjinjinBench-v1 跑分结果

评测集 `OjinjinBench-v1` 由 **1,310 条纯真实最新社区评论**构成（`150` 显性 AI + `160` 规则零分隐蔽 AI + `617` 难负真人科普/影评长文 + `383` 普通真人），**0% 合成数据，且与训练集在视频与账号维度 100% 严格隔离**。交互式跑分看板与指标详解见 [`docs/index.html`](docs/index.html) 与 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)。

### 赛道一：纯文本裸考（仅输入评论正文，无任何账号/上下文线索）

> 衡量模型的真实语义判别能力，也是跨模型（含 `0.5B–0.6B` 端侧模型）最公平的横向对比赛道。

| 模型 | 体积 | PR-AUC `↑` | Recall@P99% `↑` | Recall@P95% `↑` | Best F0.5 `↑` | 隐蔽 AI 召回 `↑` | 难负真人误杀 `↓` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`OjinjinNet-Pro` (INT8)** | **49.49 MB** | **0.9785** | **83.2%** | **92.3%** | **0.9647** | **90.0%** | 1.78% |
| `OjinjinNet-Standard` (INT8) | **16.86 MB** | 0.9721 | 64.2% | 87.4% | 0.9436 | 85.0% | **1.62%** |
| `OjinjinNet-Lite` (INT8) | **7.05 MB** | 0.9535 | 14.8% | 82.9% | 0.9245 | 81.2% | 1.62% |
| *参考：离线 12 层教师 (ELECTRA-180g-base FP32)* | *~390 MB* | *0.9853* | *72.9%* | *92.6%* | *0.9547* | *91.9%* | *2.27%* |

### 赛道二：端到端实战（ONNX INT8 + 1 KB 轻量元信息线性融合 vs 启发式规则基线）

> 结合账号等级、昵称/复读等轻量特征进行 1 KB 线性融合，模拟真实客户端静默过滤效果。

| 模型 | 体积 | PR-AUC `↑` | Recall@P99% `↑` | Recall@P95% `↑` | Best F0.5 `↑` | 隐蔽 AI 召回 `↑` | 难负真人误杀 `↓` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 启发式规则基线（含高阶账号保护） | 0 MB | 0.8821 | 32.3% | 42.6% | 0.8584 | 0.0% | 0.97% |
| `OjinjinNet-Lite` (+1KB 融合) | **7.05 MB** | 0.9896 | **85.5%** | 94.5% | 0.9631 | 90.6% | 2.11% |
| `OjinjinNet-Standard` (+1KB 融合) | **16.86 MB** | 0.9917 | 83.9% | 95.2% | **0.9680** | 90.6% | 2.11% |
| **`OjinjinNet-Pro` (+1KB 融合)** | **49.49 MB** | **0.9926** | 83.2% | **96.8%** | 0.9658 | **93.8%** | 2.27% |

---

## 核心技术亮点

1. **SmoothQuant 解决 768 维中文 BERT 的 INT8 离群值塌陷**：实测发现 768 维中文 Encoder 的 FFN 输出层存在高达 `24.7×` 的通道激活离群值，直接动态 INT8 量化会导致相关系数跌至 `0.984`。通过在导出期插入等价通道缩放（`α=0.80`，仅增 ~60 KB），将 INT8 与 FP32 相关系数拉回 `0.9990`，实现无损压缩。
2. **两级蒸馏 + 12 层教师剪枝继承**：使用 3 架构 12 层离线教师（`ELECTRA-180g-base`、`MacBERT-base`、`BGE-base`）在隔离未标注池上生成连续软标签与分歧权重；`OjinjinNet-Pro` 直接从精调后的 12 层 `MacBERT-base` 抽取 `[0, 2, 4, 6, 8, 11]` 层继承表征并蒸馏恢复，在 `Recall@P99%` 上以 1/8 体积反超 400 MB 教师模型。
3. **词表按频裁剪（21,128 → 8,192 / 10,000）+ Head-Tail 截断**：在保留全部常用中文与标点的同时砍掉约 60% 冗余嵌入参数；超长评论保留前 140 + 后 50 token，确保开头定调与结尾总结套话不丢失。
4. **零 PyTorch 极简推理**：运行时仅需 `onnxruntime` + `numpy`，内置百行纯 Python WordPiece 分词器，可直接平迁至 Kotlin / Swift / Rust / JavaScript。

---

## 快速上手（零 PyTorch 推理）

1. 从 [GitHub Releases](../../releases) 下载任意一档模型包（例如 `ojinjinnet-standard-int8-v1.0.0.zip`）并解压到 `./models/standard/`。
2. 使用 `uv` 安装轻量依赖（仅 `onnxruntime` + `numpy`）：

```bash
uv sync
```

3. 命令行直接推理：

```bash
uv run ojinjinnet --model-dir ./models/standard \
  "这波属实难绷，看完了直接笑出声" \
  "不得不说这个视角确实很独特，不是简单的情绪宣泄，而是从底层逻辑剖析了问题本质，值得深思。"
```

4. Python API 调用：

```python
from ojinjinnet import OjinjinClassifier

clf = OjinjinClassifier("./models/standard", num_threads=4)
scores = clf.predict_proba([
    "哈哈哈太真实了，笑死我了",
    "平心而论，这个观点看似犀利实则忽略了现实中的复杂博弈，让人看完颇有触动。",
])
print(scores)
```

---

## 数据隐私与合规说明

为尊重社区用户隐私与平台内容规范，本仓库**不公开发布原始抓取评论文本与含可识别用户身份信息的训练集**（详见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)）。仓库完整公开模型推理代码、标准化评测框架、方法论说明及无损 ONNX INT8 模型权重。

---

## 开源协议与致谢

- 本项目代码与衍生权重采用 **[Apache License 2.0](LICENSE)** 开源，允许个人、开源项目与商业应用自由集成与二次开发。
- 致谢以下优秀中文开源预训练模型（详见 [`NOTICE`](NOTICE)）：
  - 哈工大讯飞联合实验室（HFL）：[MiniRBT](https://github.com/ymcui/MiniRBT)、[MacBERT](https://github.com/ymcui/MacBERT)、[Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA)（Apache-2.0）
  - 智源研究院（BAAI）：[BGE-small-zh-v1.5](https://github.com/FlagOpen/FlagEmbedding)（MIT）
