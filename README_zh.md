# MemCode 使用说明

**MemCode: Discrete Semantic Representations for Long-Term Agent Memory** 已被 **NeurIPS 2026** 录用（Poster）。

[论文](https://arxiv.org/abs/2610.19472) | [PDF](https://arxiv.org/pdf/2610.19472) | [English](README.md)

本项目提供 LoCoMo / LongMemEval、GPT / Qwen、未演化 / 演化共 8 个运行入口。
参数直接定义在 `settings.py`，个人密钥和本地路径通过 `local_config.py` 覆盖。

## 环境和配置

1. 使用 Python 3.10/3.11，安装 `requirements.txt`。
2. 执行 `python -m nltk.downloader punkt punkt_tab`。
3. 按 `data/README.md` 准备数据，按 `models/README.md` 配置本地模型。
4. 将 `local_config.example.py` 复制为 `local_config.py`，填写 API、设备和路径。
   `local_config.py` 不纳入 Git 或源码压缩包。
5. 初次设置 `MAX_ROWS = 1`；全量运行设置为 `None`。
   每次独立实验使用新的 `RUN_LABEL`。
6. 执行 `python check_environment.py`，再启动所需入口。

## 运行入口

| 数据集 | GPT 未演化 | GPT 演化 | Qwen 未演化 | Qwen 演化 |
|---|---|---|---|---|
| LoCoMo | `run_locomo_gpt_baseline.py` | `run_locomo_gpt_consolidated.py` | `run_locomo_qwen_baseline.py` | `run_locomo_qwen_consolidated.py` |
| LongMemEval | `run_longmemeval_gpt_baseline.py` | `run_longmemeval_gpt_consolidated.py` | `run_longmemeval_qwen_baseline.py` | `run_longmemeval_qwen_consolidated.py` |

例如：`python run_locomo_gpt_baseline.py`。
`baseline` 指不执行演化的 MemCode。构建、演化、答题使用所选工作模型；
评估模型通过 `JUDGE_MODEL` 配置。

结果保存在 `runs/<profile>/<RUN_LABEL>/`，包含逐题 QA 记录、汇总结果、
性能计数和记忆数据库。LoCoMo 使用新的输出目录；LongMemEval 可在配置一致时续跑。
LoCoMo 运行器排除 category 5。

## 数据处理流程

完整消息 → 分句与历史背景 → LLM 抽取 AMU → BGE 向量与 V10 离散码入库
→ 可选演化 → 双路检索 → 原始消息回填 → 答题与评估。

LoCoMo 的写入背景最多 10 个句子单元，LongMemEval 最多 5 个。
普通检索最多选择 20 个向量 AMU 和 20 个去重后的新码本 AMU；
LongMemEval 未演化版本最多回填 10 条完整原始消息。
句子单元通过 `parent_bmu_id` 指向完整消息，背景窗口、AMU 候选数和消息回填数是不同层级。

## 复用已有记忆

设置 `EXISTING_MEMORY_SOURCE_DIR` 和 `EXISTING_MEMORY_PROFILE`，执行
`python run_longmemeval_from_existing_memory.py`。

源目录下每个 `row_<index>_data/` 必须包含：

- `metadata.db`
- `metadata.db.stats.json`
- `vector_stores/atomic_memory.faiss`
- `vector_stores/atomic_memory_map.json`

源记忆使用的数据顺序、嵌入模型、权重和工作模型应与当前配置一致。
程序先复制到独立输出目录，再执行演化，不修改源记忆。

## 训练和测试

按 `data/README.md` 准备本地 DailyDialog JSON，安装 `requirements-analysis.txt`，
执行 `python -m vqvae_training.train_vqvae`。权重保存到 `models/T5/`。

执行 `python -m unittest discover -s tests -v` 运行离线测试，
执行 `python tools/validate_release.py` 检查源码。
