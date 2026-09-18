# Examples / 示例

## `demo_no_cst.py`

A complete run of the surrogate half of the loop on synthetic data — no CST installation, no
GPU, no network. It creates a temporary task, imports 60 analytic samples as if they were
full-wave results, selects and trains a surrogate, freezes one SB-SADEA candidate, and prints
the frozen prediction next to the best design in the database.

```bat
.venv\Scripts\python.exe examples\demo_no_cst.py
```

Use it to see the workflow and the task layout before pointing the agent at a real project.

## `config_template.json`

A starting point for a real task, written as a generic two-port component. Replace the
placeholders:

* `project` — full path to your `.cst` file;
* `cst_install` — your CST Studio Suite installation directory;
* `parameters` — every independent parameter that varies in your data, with the box you want to
  search. Take the union of the optimiser ranges of all project versions you will import **and**
  the parameter values of the saved samples themselves;
* `metrics` — the result trees, bands and limits of your own targets. Band endpoints must be
  exact frequency samples of the stored curve. Every `kind` here is `curve` — any 1D result tree —
  except the last entry, which shows the one antenna-specific kind, `sll_phi_cut`. **Delete that
  entry unless you are optimising an antenna**; the name says so to stop it being copied by
  accident. On Windows, note that every backslash in a JSON string is doubled.

Running `agent.py setup` writes a config of this shape for you, with the trees read from your own
project and the band endpoints snapped to real frequency samples.

Keep `"simulation_budget": 0` until you have imported data, trained, and looked at the first
proposed candidate. Nothing can start the solver while the budget is zero.

---

## 中文说明

### `demo_no_cst.py`

在合成数据上完整跑一遍替代模型半环——不需要 CST、不需要 GPU、不联网。脚本会建一个临时任务，
导入 60 组解析数据当作全波结果，自动选模训练，冻结一个 SB-SADEA 候选，并把冻结的预测与
数据库中最优设计并列打印。在对准真实工程之前，用它来熟悉流程和目录结构。

```bat
.venv\Scripts\python.exe examples\demo_no_cst.py
```

### `config_template.json`

真实任务的配置起点，示例写成一个通用二端口器件。替换其中的占位符：

* `project`——你的 `.cst` 文件完整路径；
* `cst_install`——CST Studio Suite 安装目录；
* `parameters`——数据中所有会变化的独立参数，以及你要搜索的范围。取你将导入的**所有工程版本**
  优化器范围的并集，**并**并上这些工程已保存样本的实际取值范围；
* `metrics`——你自己的结果树、频带和门限。频带端点必须是曲线的采样频点。除最后一项外，
  所有 `kind` 都是 `curve`，即任意一维结果树；最后一项演示唯一与天线绑定的 `sll_phi_cut`，
  **不是做天线就删掉它**——名字里已经写明，免得被顺手复制。另外在 Windows 上，JSON 字符串里的
  反斜杠都要写成两个。

直接跑 `agent.py setup` 就能生成同样结构的配置：结果树从你自己的工程里读，频带端点自动对齐到真实采样点。

在导入数据、训练、看过第一个候选之前，把 `"simulation_budget"` 保持为 0——预算为 0 时任何命令都
无法启动求解器。
