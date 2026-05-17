# Codex 项目说明

这是给 Codex 看的工程说明，不是课程报告正文。回答和改代码时优先遵守这里的约定。

## 项目目标

- 本仓库用于研究生课程《机器学习》期末大作业。
- 任务是图上的节点分类，数据来自 `dataset/` 下的引用网络。
- 不要把项目退化成“鸢尾花分类”这类独立样本入门任务；实现和说明都要体现节点特征、图拓扑、邻居信息传播、训练/验证/测试划分和泛化评估。

## 环境

- 默认 Python 解释器：`/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python`
- 运行、编译、烟测时优先显式使用这个解释器。
- 不要新建虚拟环境，不要把依赖装到系统 Python。
- 当前主要依赖包括 PyTorch、PyG、NumPy、NetworkX。PyG 可能打印 `torch-sparse` 动态库警告，只要模型能正常运行即可。

## 代码结构

当前训练工程采用单文件模块：

- `args.py`：命令行参数和基础参数校验。
- `main.py`：主入口，只负责解析参数、设置随机种子、初始化日志、加载数据、构建模型、启动训练。
- `model.py`：模型定义和 `build_model`。
- `trainer.py`：训练循环、验证、测试、早停、保存结果和可选 checkpoint。
- `utils.py`：数据加载、特征归一化、mask 划分、指标计算、日志初始化、设备封装。
- `tools.py`：只保留当前实际复用的通用工具函数。

不要把数据加载、模型定义、训练循环和结果输出重新堆回一个脚本里。新增功能优先放进职责对应的模块。

## 工具函数约定

- 随机种子必须使用 `tools.set_seed`。
- 平台相关设备选择必须通过 `tools.set_device` 间接完成；`utils.get_device` 可以做参数层封装。
- 目录创建使用 `tools.create_dirs`。
- `tools.py` 只保留被项目实际使用的函数；新增工具前先确认是否真的需要共享。

## 数据约定

- 数据根目录：`dataset/`
- 当前支持：`cora`、`citeseer`、`pubmed`
- 每个数据集目录应包含：
  - `features.npy`
  - `id_map.json`
  - `class_map.json`
  - `<dataset>.cites` 或 `origin/<dataset>.cites`
- `utils.load_graph_data` 负责读取这些文件。
- 特征读取后做按行归一化。
- 边读取后构造双向 `edge_index`，自写基础 GNN 另构造归一化稀疏邻接矩阵。
- 模型输入维度、类别数、节点数必须从数据中推断，不要硬编码。

已知数据规模：

- Cora：2708 节点，5429 条原始引用边，1433 维特征，7 类。
- Citeseer：3312 节点，4715 条原始引用边，3703 维特征，6 类。
- Pubmed：19717 节点，44338 条原始引用边，500 维特征，3 类。

## 模型要求

至少维护这三个模型入口：

- `basic_gnn`：自己用稀疏邻接矩阵乘法实现的基础 GNN。
- `gcn`：使用 `torch_geometric.nn.GCNConv`。
- `graph_transformer`：使用 `torch_geometric.nn.TransformerConv`。

PyG 层尽量按需导入，避免运行 `basic_gnn` 时触发无关 PyG 警告。

## 日志与结果

- 训练过程必须有日志。
- 默认输出目录是 `outputs/`。
- 每次运行使用 `run_name` 生成独立目录：`outputs/<run_name>/`。
- 该目录下保存：
  - `run.log`
  - `results.json`
  - `best.pt`，仅在 `--save-model` 时生成
- 默认 `run_name` 包含数据集、模型、seed 和时间戳。
- 如果用户显式传入相同 `--run-name`，日志文件按 Python logging 默认行为追加写入；结果 JSON 用写模式覆盖。
- `--no-save-results` 可关闭结果 JSON 保存。

## 常用命令

编译检查：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python -m py_compile args.py utils.py model.py trainer.py main.py tools.py
```

快速 CPU 烟测：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model basic_gnn --epochs 1 --device cpu
```

运行三个模型：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model basic_gnn
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model gcn
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model graph_transformer
```

## 修改代码时的默认要求

- 修改后至少运行 `py_compile`。
- 涉及训练流程时，至少跑 Cora 的 1 epoch CPU 烟测。
- 不要覆盖或删除 `dataset/` 下的原始数据和预处理结果。
- 不要提交 `outputs/`、checkpoint、缓存文件；这些由 `.gitignore` 忽略。
- 如果生成了临时烟测输出，通常不用专门删除，除非用户要求清理。
- 回答用户时简洁说明改了哪些文件、验证了什么、还有什么风险。
