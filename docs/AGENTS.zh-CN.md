[English](../AGENTS.md) | **中文**

# 大模型助手驱动本工具的说明

本文告诉 AI 助手（Claude Code、Cursor、Copilot Chat、Codex、任何带终端能力的本地模型）如何代替用户
操作这个仿真替代模型 Agent。**不需要 API key，也不需要联网**：Agent 是一个本地 Python 命令行程序，
对接本机安装的 CST。助手提供判断与汇报，Agent 提供确定的、可审计的流程。

如果你就是这样的助手：动手之前请把本文从头读完。

---

## 1. 你在驱动什么

`agent.py` 实现的闭环是：导入全波数据 → 训练替代模型 → **在替代模型上优化** → 冻结候选 →
用 CST 验证 → 回填 → 再循环。每一步都写入任务目录，那里是唯一的事实来源。你不需要自己记状态：
读 `config.json`、`state.json`、`data.json`、`events.jsonl`、`报告.md` 和 `batches/*/batch.json` 即可。

命令速查见 [`../README.zh-CN.md`](../README.zh-CN.md)，详细手册见 [`使用说明.md`](使用说明.md)。

## 2. 不可违反的规则

1. **未经用户对该次运行的明确批准（包含允许用多少次求解），绝不启动 CST 仿真。**
   `validate`、`resume`、`run`、`materialize` 都会启动求解器；准备类命令（`init`、`import-*`、
   `train`、`propose`、`report`、`bounds`、`inspect`）不会，可以放心执行。
2. **替代模型负责优化，CST 只负责验证。** 不要手工编造候选，也不要去调 CST 自带的优化器。
   如果你觉得内置的挑点方式太保守或太激进，先说明、提方案、等用户同意。
3. **按 Agent 自己的流程走。** 觉得缺了某个环节，就提议把它加进 Agent，而不是在外面写脚本绕过去。
   绕过 `propose`/`validate` 的临时脚本会丢掉"预测先冻结"的审计链——那正是这个工具存在的意义。
4. **绝不把预测值当成已实现的结果汇报。** 预测和实测必须分列，并给出差值。
5. **不要手工改任务目录里的文件**，除非是在处理已记录在案的故障；真要改，必须明确说出来。

## 3. 开始一个新结构

```bat
REM 1. 找齐同一结构的所有工程文件，不要只用最新的那个
REM 2. 比较它们的优化器范围
.venv\Scripts\python.exe agent.py bounds --project A.cst --project B.cst
```

然后写配置（复制 `examples/config_template.json`），**参数范围**要这样定：

* 取你将要导入的所有版本的优化器范围的并集，**并且**
* 再并上这些工程里已保存样本的实际取值范围——工程的优化器范围常在跑完一轮后被收窄，
  早期的好样本会落在范围之外，导入时被静默拒绝，数据就悄悄变少了。

先把 `"simulation_budget"` 设为 0，这样绝不会误启动求解。

```bat
.venv\Scripts\python.exe agent.py init --task tasks\my_task --config my_config.json
.venv\Scripts\python.exe agent.py import-cst --task tasks\my_task --project A.cst --project B.cst
.venv\Scripts\python.exe agent.py train --task tasks\my_task
.venv\Scripts\python.exe agent.py propose --task tasks\my_task --count 1
```

检查 `import_summary.json`（各工程的读入/入库/重复/拒绝数量）和 `bounds_advice.json`
（导入数据或优化器范围超出任务范围时才会写）。然后汇报数据量、各指标达标比例、当前最优设计、
模型留出误差和第一个候选，再请用户给仿真预算。

### 带频带的指标

指标用 `"band": [f1, f2]` 时，两个端点必须正好是结果曲线的采样频点，否则提取会失败。建任务前先核实。

## 4. 跑闭环

```bat
.venv\Scripts\python.exe agent.py run --task tasks\my_task --rounds N --count 1 --budget N
```

启动前告诉用户：预计耗时、期间源工程必须保持关闭、想中途停就在任务目录下建一个空的 `PAUSE` 文件
（当前这次求解结束后停）。**强制关闭 CST 会连带杀掉 Agent 的后台求解，那一次预算就浪费了。**

运行期间不要轮询等待；跑完后汇总：

* 每轮一行：选点方式、各指标的预测与实测、适应度、库内最优走势；
* 各指标的预测误差（MAE），以及哪些指标系统性偏乐观；
* 有多少候选压在参数边界上——持续触边说明范围限制了优化；
* 有没有出现全部达标的设计。

## 5. 如实解读结果

* `fitness` 是各约束超标量的加权和，0 表示全部满足。
* 某项差距小于 Agent 对该指标自身的预测误差时，只能说"接近"，不能说"达标"。
* 隔离副本与源工程中重解同一设计，差异通常在 0.05–0.3 dB。余量比这还小的，要说明是"压线达标"。
* 如果最优的几组分成了各自满足不同约束子集的两类，那就是真实的取舍。先量化它
  （把一项指标分区间，报告另一项在各区间能达到的最好值），再建议要不要继续加轮次。

## 6. 引导手段

| 现象 | 手段 |
|---|---|
| 搜索总在牺牲用户在意的某项指标 | 给该指标加 `"penalty_weight": 2`–`3`，重新训练与生成候选 |
| 最优设计持续压同一个边界 | 建议放宽该范围；**范围变化必须新建任务** |
| 候选离数据太远、预测过于乐观 | 调大 `conservative_sigma`，或把 `search_domain` 从 `full` 改回 `trust_region` |
| 轮次波动大、种群横跨互不相干的区域 | 补密数据；或减小 `sb_sadea.lambda`（偏离论文设定，需用户同意）|

改目标、参数范围或挑点方法会改变配置签名，必须重新训练并重新生成候选；只改预算则不会。

## 7. 把好结果存给用户看

```bat
.venv\Scripts\python.exe agent.py materialize --task tasks\my_task [--design 候选ID]
```

它会先完整备份源工程，写入参数，用工程原有的求解器设置求解并保存，这样用户就能在 CST 里直接看曲线。
`run` 闭环中每当库内最优被刷新也会自动执行一次（`materialize.on`，默认 `improvement`，
单次 `run` 最多 `max` 次，默认 3）——要提前告诉用户会发生这件事，每次约等于一次求解的时间。

## 8. 求解失败时

Agent 会停下并保留现场：候选 `state: running`、`job.json` 里有退出码、`status.txt` 里没有 `DONE`。
它**不会自动重试**，`discard` 和 `resume` 也会故意拒绝处理这种候选。先诊断：读 `job.json`、
`status.txt` 和 `project/Result/Model.log`——重建失败指向这组参数本身；而划网或求解中途崩溃通常是
外部原因（用户关了 CST、许可证冲突）。然后问用户：是重试（把崩溃的副本移走，让 `run` 重新准备一份干净的，
消耗 1 次预算），还是放弃这个候选。

## 9. 汇报方式

先给决策相关的结论：有没有全部达标的设计；如果没有，差的是哪一项、差多少。然后才是证据。
预测值和实测值分开列，每个数字都能说出它来自哪个文件或哪个 Run，没验证过的事情要明确说没验证。
