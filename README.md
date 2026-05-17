# NodeClassification

## 项目简介

本项目是研究生课程《机器学习》的期末大作业代码，任务是图上的节点分类。项目使用 Cora、Citeseer、Pubmed 三个引用网络数据集，目标是在给定节点特征和图结构的情况下预测节点类别。

与鸢尾花分类这类独立样本入门任务不同，节点分类需要同时处理节点属性、边连接关系、邻居信息传播、半监督式训练/验证/测试划分以及图模型的泛化评估。本项目目前实现了一个自写基础 GNN、一个 PyTorch Geometric GCN，以及一个基于 PyG TransformerConv 的 Graph Transformer。

## 数据集介绍

数据位于 `dataset/` 目录下，当前包含：

| 数据集 | 节点数 | 原始引用边数 | 特征维度 | 类别数 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| Cora | 2708 | 5429 | 1433 | 7 | 机器学习论文引用网络 |
| Citeseer | 3312 | 4715 | 3703 | 6 | 计算机科学论文引用网络 |
| Pubmed | 19717 | 44338 | 500 | 3 | 医学论文引用网络 |

每个数据集目录包含预处理结果：

- `features.npy`：节点特征矩阵。
- `id_map.json`：原始节点 ID 到连续整数 ID 的映射。
- `class_map.json`：原始节点 ID 到类别标签的映射。
- `origin/<dataset>.content`：原始节点特征和标签文件。
- `origin/<dataset>.cites`：原始引用边文件。
- `remake.py`：从原始文件重新生成预处理结果的脚本。

当前训练入口由 `utils.load_graph_data` 读取数据：特征会做按行归一化；引用边会转为双向 `edge_index`；自写 `basic_gnn` 会额外构造加入自环后的对称归一化稀疏邻接矩阵。

## 必要的环境依赖

推荐 Python 版本：3.11。核心依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

如果需要 CUDA 版本的 PyTorch，建议先按本机 CUDA 版本安装 PyTorch，再安装其余依赖。当前仓库代码可以在 CPU 上完成快速验证。

## 项目文件结构

```text
.
├── README.md                 # 面向读者的项目说明
├── CODEX.md                  # 面向 Codex 的工程约定
├── requirements.txt          # Python 依赖
├── args.py                   # 命令行参数定义和基础校验
├── main.py                   # 训练主入口
├── model.py                  # 模型定义和 build_model
├── trainer.py                # 训练、验证、测试、早停、保存结果
├── utils.py                  # 数据加载、划分、指标、日志、设备封装
├── tools.py                  # 通用工具：随机种子、设备选择、目录创建
├── dataset/
│   ├── cora/
│   ├── citeseer/
│   └── pubmed/
└── outputs/                  # 运行日志、结果和可选 checkpoint，已被 .gitignore 忽略
```

## 已实现模型

| 参数值 | 实现位置 | 说明 |
| --- | --- | --- |
| `basic_gnn` | `model.BasicAdjGNN` | 使用 `torch.sparse.mm(adj_norm, x)` 自己实现的基础 GNN。 |
| `gcn` | `model.PyGGCN` | 使用 `torch_geometric.nn.GCNConv`。 |
| `graph_transformer` | `model.PyGGraphTransformer` | 使用 `torch_geometric.nn.TransformerConv`。 |

## Quick Start

使用当前项目环境运行：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model basic_gnn
```

CPU 快速烟测：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model basic_gnn --epochs 1 --device cpu --run-name smoke_basic
```

分别运行三个模型：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model basic_gnn
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model gcn
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model graph_transformer
```

切换数据集：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset citeseer --model gcn
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset pubmed --model basic_gnn
```

保存最佳模型权重：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model gcn --save-model
```

每次运行默认生成一个独立输出目录：

```text
outputs/<run_name>/
├── run.log        # 运行日志
├── results.json   # 最终指标和训练历史
└── best.pt        # 仅在 --save-model 时生成
```

`run_name` 默认格式为 `<dataset>_<model>_seed<seed>_<timestamp>`。也可以手动指定：

```bash
/mnt/HDD2/yzy23/.conda/envs/ocs/bin/python main.py --dataset cora --model gcn --run-name cora_gcn_baseline
```

## 超参数说明

| 参数 | 默认值 | 可选值/类型 | 说明 |
| --- | --- | --- | --- |
| `--dataset` | `cora` | `cora`, `citeseer`, `pubmed` | 使用的数据集。 |
| `--data-root` | `dataset` | 路径 | 数据集根目录。 |
| `--model` | `basic_gnn` | `basic_gnn`, `gcn`, `graph_transformer` | 使用的模型。 |
| `--hidden-dim` | `64` | 正整数 | 隐藏层维度。 |
| `--num-layers` | `2` | 正整数 | 模型层数。 |
| `--dropout` | `0.5` | `[0, 1)` 浮点数 | 隐藏层 dropout 比例。 |
| `--heads` | `4` | 正整数 | Graph Transformer 的注意力头数；当 `num_layers > 1` 时，`hidden_dim` 必须能被 `heads` 整除。 |
| `--epochs` | `200` | 正整数 | 最大训练轮数。 |
| `--lr` | `0.01` | 浮点数 | Adam 学习率。 |
| `--weight-decay` | `5e-4` | 浮点数 | Adam 权重衰减。 |
| `--patience` | `50` | 正整数 | 早停耐心值；验证集准确率连续若干轮不提升后停止训练。 |
| `--train-ratio` | `0.6` | 浮点数 | 每个类别中划入训练集的比例。 |
| `--val-ratio` | `0.2` | 浮点数 | 每个类别中划入验证集的比例；剩余样本进入测试集。 |
| `--seed` | `42` | 整数 | 随机种子，用于数据划分和模型初始化。 |
| `--device` | `auto` | `auto`, `cpu`, `cuda` | 运行设备；`auto` 会按平台优先选择可用 GPU，否则回退 CPU。 |
| `--gpu` | `0` | 整数 | CUDA 设备编号。 |
| `--output-dir` | `outputs` | 路径 | 所有运行输出的根目录。 |
| `--run-name` | 自动生成 | 字符串 | 当前运行的输出子目录名。 |
| `--no-save-results` | `False` | flag | 开启后不保存 `results.json`。 |
| `--save-model` | `False` | flag | 开启后保存最佳验证集 checkpoint 到 `best.pt`。 |
| `--log-every` | `10` | 正整数 | tqdm postfix 指标更新间隔；第 1 轮总会更新。 |

注意：`--train-ratio + --val-ratio` 必须小于 1，测试集比例为剩余部分。

## 结果与指标

训练过程使用 `tqdm` 显示进度。最终在日志和 `results.json` 中记录：

- train/val/test loss
- train/val/test accuracy
- train/val/test macro F1
- train/val/test weighted F1
- best epoch
- best validation accuracy
- 每个 epoch 的训练历史
