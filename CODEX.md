# Codex Notes

本文件只保留给 Codex 的必要工程约定。面向读者的项目介绍、数据集、超参数和 Quick Start 请看 `README.md`。

## 固定环境

- 默认解释器：`/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python`
- 运行、编译、烟测时优先显式使用该解释器。
- 不要新建虚拟环境，不要把依赖装到系统 Python。
- 依赖版本以 `requirements.txt` 为准。

## 当前模块职责

- `args.py`：命令行参数和参数校验。
- `main.py`：入口编排，不写模型或训练细节。
- `model.py`：模型定义和 `build_model`。
- `trainer.py`：训练、验证、测试、早停、保存结果。
- `utils.py`：数据加载、mask、指标、日志、设备封装。
- `tools.py`：只保留实际复用的通用函数，目前是 `set_seed`、`set_device`、`create_dirs`。

新增代码时保持职责边界，不要把数据加载、模型定义、训练循环重新堆到一个文件里。

## 必须保持的行为

- 支持的数据集：`cora`、`citeseer`、`pubmed`。
- 支持的模型入口：`basic_gnn`、`gcn`、`graph_transformer`。
- 数据加载由 `utils.load_graph_data` 负责，从 `dataset/` 读取现有预处理文件；不要恢复旧的 `GraphData.py`。
- 随机种子使用 `tools.set_seed`。
- 设备选择通过 `tools.set_device`，`utils.get_device` 只做参数封装。
- 输出目录保持为 `outputs/<run_name>/`，其中包括 `run.log`、`results.json`、`test_outputs.json`，以及可选 `best.pt`。
- 不要改动或删除 `dataset/` 下的原始数据和预处理结果。

## 验证要求

修改 Python 代码后至少运行：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python -m py_compile args.py utils.py model.py trainer.py main.py tools.py
```

涉及训练流程、数据加载、模型或指标时，至少再跑：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model basic_gnn --epochs 1 --device cpu --run-name smoke
```

涉及 PyG 模型时，额外跑对应 1 epoch CPU 烟测。

## 输出回答

最终回复要简洁说明：

- 改了哪些文件。
- 做了哪些验证。
- 是否有未解决风险或环境限制。
