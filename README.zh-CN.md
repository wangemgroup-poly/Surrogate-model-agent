[English](README.md) | **中文**

# CST 仿真替代模型 Agent

*面向任意 CST 参数化频域模型的替代模型优化。*

一个完全离线的本地 Agent：用 CST Studio Suite 的全波结果训练替代模型，**在替代模型上做优化**，
把候选设计连同预测一起冻结，再由 **CST 裁决**它们是否真的达标。它不限定器件类型：只要目标能从一维结果算出来——滤波器、耦合器、匹配网络、波导器件、
频率选择表面、天线——配置方式完全一样，只是结果树路径不同。它是一个普通的命令行程序——
**不需要 API key、不联网、不依赖云服务**。任何能执行命令的大模型助手（Claude Code、Cursor、
Copilot Chat……）读 [`AGENTS.md`](AGENTS.md) 就能驱动它，你也可以完全手工使用。

在线优化部分参考 Liu et al., *An Efficient Method for Antenna Design Based on a Self-Adaptive
Bayesian Neural Network Assisted Global Optimization Technique*, IEEE TAP 2022
（DOI 10.1109/TAP.2022.3211732），下称 **SB-SADEA**。

---

## 它解决什么问题

在 CST 里直接做参数优化，每试一个设计就要跑一次全波求解。用已有结果训练的替代模型可以在几秒内
对上万个候选打分，于是每一次宝贵的求解都花在"模型认为最有希望"的设计上。本工具把这个闭环自动化，
同时守住一条界线：

* 替代模型**只负责提议**——候选参数和预测值在任何求解启动之前就写入磁盘冻结；
* **CST 才有裁决权**——只有实测的全波结果能判定达标；
* 报告里每个数字都能追溯到文件哈希、Run ID 和批次。

该闭环在波导缝隙阵上实测过：起点设计的带内最差 S11 离目标还差 3 dB 以上，用 34 次求解找到了
完全达标的设计。流程本身与这个具体问题无关。

---

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows（CST 结果库与 VBA 宏执行器仅支持 Windows）|
| CST | CST Studio Suite 2025，**频域**求解器，参数化工程 |
| Python | 3.12——CST 2025 的 Python 结果库要求 |
| 目标指标 | 任何能从已保存的一维结果算出的量（S 参数、传输系数、增益表、群时延、效率……）；另有一个可选类型读远场 φ 切面算天线旁瓣 |
| 依赖 | numpy、scipy、scikit-learn、joblib、threadpoolctl（见 `requirements.txt`）|
| 可选 | PyTorch，仅用于变分推断 BNN 代理（`sb_sadea.surrogate = "bnn"`）|

没有 API key、没有账号、没有任何外部连接。全部在你自己的机器上、对着你自己的 CST 运行。

## 安装

```bat
setup.cmd                                  REM 用 Python 3.12 创建 .venv 并安装依赖
setup.cmd "C:\完整的python3.12路径\python.exe"   REM py -3.12 不在 PATH 时
```

`setup.cmd` 还会安装 CUDA 12.6 版 PyTorch（无 GPU 自动回落 CPU），只在使用可选的 BNN 代理时需要；
不需要就删掉那一行。

验证安装：

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

应通过 21 项测试。测试是自包含的：不需要 CST，也不需要 GPU。

## 不装 CST 也能试

```bat
.venv\Scripts\python.exe examples\demo_no_cst.py
```

演示脚本构造一个三参数的"虚拟结构"，把 60 组解析数据当作全波结果导入，自动选模训练，
再冻结一个 SB-SADEA 候选——即整个替代模型半环，任何机器都能跑。

---

## 工作流程

```
init ──► import-cst / import-json ──► train ──► propose ──► validate ──► （循环）
                                        ▲                      │
                                        └────── 回填 ──────────┘
                              run 命令自动执行这个闭环
```

| 命令 | 作用 |
|---|---|
| `setup [--project X.cst] [--lang en\|zh]` | **引导式建任务**：读取并体检工程、交互定义目标、给出是否需要分段的建议、写配置并建任务 |
| `init --task 目录 --config 文件` | 建任务；登记源工程的文件哈希、几何签名与固定参数签名 |
| `bounds --project A.cst [--project B.cst]` | 打印各工程的优化器范围及其并集，写配置前用 |
| `inspect --task 目录` | 导出源工程的 Run 列表、参数定义与结果树条目 |
| `import-cst --task 目录 [--project X.cst ...]` | 读取已保存的 Run；额外工程经结构校验后合并，样本 ID 加版本前缀 |
| `import-json --task 目录 --file rows.json` | 导入外部整理好的全波指标（须同结构、同求解设置）|
| `train --task 目录 [--force]` | 在留出区域上比较可用算法、选定后全量重拟合，写入 `models/model_NNN/` |
| `propose --task 目录 --count N` | 冻结 N 个候选（参数、预测、σ、模型哈希、数据哈希、配置签名）|
| `seed --task 目录 --count N` | 无数据时的拉丁超立方初始采样计划 |
| `validate --task 目录 --budget N` | 在工程的隔离副本中求解候选、校验、回填 |
| `resume --task 目录 --budget N` | 取消暂停并继续收集/验证 |
| `run --task 目录 --rounds R --count N --budget B` | 训练→生成候选→验证→重训，循环；出现全部达标即停 |
| `pause --task 目录` | 当前求解结束后暂停 |
| `discard --task 目录` | 放弃尚未启动求解的候选 |
| `materialize --task 目录 [--design ID]` | 把已验证的设计写回**源工程**求解保存，便于在 CST 里看曲线 |
| `rebaseline --task 目录` | 源工程仅新增了保存的 Run 且几何未变时，重新登记 |
| `report` / `status --task 目录` | 重新生成 `报告.md` / `报告.html` |

不带参数运行 `run_agent.cmd` 会进入中文交互菜单。

### 任务目录

```
tasks/my_task/
├─ config.json            目标、参数范围、方法、登记的签名
├─ data.json              全部入库设计：参数、指标、来源
├─ state.json             阶段、当前模型、当前批次、已启动求解次数
├─ events.jsonl           只追加的决策日志
├─ models/model_NNN/      model.joblib、selection.json（算法比较）、data_snapshot.json
├─ batches/batch_XXXX/    batch.json（冻结的预测）、cst/<候选>/（隔离工程副本）
├─ materialized.json      写回源工程的记录
├─ 报告.md / 报告.html      人读报告
└─ tradeoff.svg           约束取舍散点（无第三方绘图依赖）
```

---

## 替代模型

`train` 先按数据规模筛出可用算法，再按"各指标误差除以 `scale` 后的均值 + 误判合格惩罚"评分，
分数接近时取更快的；测试区域只在选定算法之后评估一次。

| 算法 | 适用条件 |
|---|---|
| 局部 ARD-Matérn 高斯过程（最近 120 点）| ≤1800 样本、≤30 输入、≤16 输出 |
| RBF 核岭回归 | ≤3000 样本 |
| 极端随机树集成 | 不限 |
| 贝叶斯末层网络（近似，非完整 BNN）| 不限 |
| 全样本 ARD-Matérn 高斯过程 | 需显式开启 `include_global_gp`，≤600 样本、≤20 输入、≤8 输出 |

数据按参数区域划分（KMeans 锚点在首次训练时固定，新增一个样本不会打乱划分），跨集合的近重复设计会被剔除。

## 候选生成

由 `proposal_method` 选择：

* **`random_pool`**（默认）——在当前最优附近的信任域球内撒点，用部署模型打分，按"预测最优 /
  乐观 LCB 探索 / 模型与邻近实测的分歧"三种角色挑选。
* **`optimize`**——直接在替代模型上做差分进化。`search_domain` 取 `trust_region` 或 `full`；
  约束按 μ + `conservative_sigma`·σ 判断，避免只看均值而选中 σ 很大的无支撑预测。
* **`sb_sadea`**——论文的在线闭环：取罚函数适应度最好的 λ 个设计，用 DE/current-to-best/1 生成 λ 个
  子代，每个子代用最近 τ 个样本单独训练局部代理，再用自适应 LCB 选点，每轮只做 1 次全波（`--count 1`）。
  默认 `λ = τ = 4d`；`surrogate` 取 `gp`（ω=2）或 `bnn`（ω=14）。

每个冻结候选都带预警：它到最近样本的距离相对数据最近邻中位距离的倍数（`extrapolation_ratio`），
以及其他算法对同一设计的交叉核查分歧（`disagreement_limit`）。预警不改变候选顺序，只提示可信程度。

## 引导取舍

目标即带门限的约束，罚函数适应度是各项超标量的加权和。给某项指标设 `"penalty_weight": 2`，
就是告诉搜索"违反它的代价加倍"。当搜索反复拿一项指标去换另一项时，这是最直接的手段——
在上述缝隙阵研究中，给高频段 S11 与旁瓣加权后，锚点从"牺牲它们"的设计转到"守住它们"的设计，
闭环随后补上了剩余差距。

---

## Agent 强制执行的安全规则

* 复制或求解前，源工程必须**关闭**（无 `Model.lok`）。
* 身份在 `init` 时登记、每次求解前复核：几何文件（`Model.mod`、`ModelHistory.json`、求解属性）
  **以及任务未优化的全部固定参数取值**。新增保存的 Run 没问题；改固定尺寸或几何则拒绝。
* 候选在**隔离副本**中求解，不复制旧结果与网格缓存。
* 求解后检查宏返回值、`DONE` 标记、进程退出码、日志中的宽带收敛，并逐项回读参数。
* 双预算同时生效：任务总预算 `simulation_budget` 与本次 `--budget`。
* 求解失败或被中断**不会自动重试**，现场原样保留，供诊断后再决定是否再花预算。
* 每个任务一把写锁。
* 改预算、CPU 数、等待时限不会作废已训练模型与已冻结批次；改目标、参数范围、挑点方法会。

## 局限——下结论前请先读

* 仅支持 CST 频域工程，其他求解器需要新适配器。
* 指标只来自你指定的结果树和已保存的 φ 切面；旁瓣按切面最大峰与次大峰之比定义。
* **不确定度未校准**：σ 是模型离散度，不是合格概率。
* 换结构必须新建任务，模型不可跨结构复用。
* 离散频点达标不等于频带内处处达标。
* "未找到达标方案"只意味着"在当前参数范围与预算内没找到"。
* 替代模型的上限由数据覆盖决定：优化前请把同一结构**所有版本**的结果都收齐，不要只用最新的工程文件。

## 许可、作者与引用

Copyright (C) 2026 MiraDaddy, Wangemgroup — <https://github.com/wangemgroup-poly>

采用 **GNU General Public License v3.0 或更新版本**，见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。
本程序按"原样"提供，不附带任何担保，也不保证适销性或特定用途适用性。若再分发（无论是否修改），
必须同样以 GPL 授权并提供源代码。

CST Studio Suite 是 Dassault Systèmes 的产品。本项目不包含任何 CST 组件，其 Python 库在运行时
从你自己的安装中加载。

若发表使用 SB-SADEA 挑点方式得到的结果，请引用原论文（DOI 10.1109/TAP.2022.3211732）。
本实现不是该论文的完整复现，差异列在 [`docs/使用说明.md`](docs/使用说明.md) 与
[`CHANGELOG.md`](CHANGELOG.md) 中。

详细中文使用说明：[`docs/使用说明.md`](docs/使用说明.md)。
给大模型助手的驱动说明：[`AGENTS.md`](AGENTS.md)（[中文版](docs/AGENTS.zh-CN.md)）。
