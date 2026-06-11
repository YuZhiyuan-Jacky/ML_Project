import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

from tools import create_dirs


@dataclass
class GraphData:
    """图节点分类任务的数据容器。

    x 是节点特征矩阵，y 是类别编号，edge_index 是 COO 形式的边列表，
    adj_norm 是基础 GNN 使用的归一化稀疏邻接矩阵，三个 mask 表示数据划分。
    """

    name: str
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    adj_norm: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    label_names: List[str]

    @property
    def num_nodes(self) -> int:
        """返回图中的节点数量。"""
        return self.x.size(0)

    @property
    def num_features(self) -> int:
        """返回每个节点的输入特征维度。"""
        return self.x.size(1)

    @property
    def num_classes(self) -> int:
        """返回分类任务的类别数量。"""
        return len(self.label_names)


def ensure_dir(path: str) -> None:
    """确保单个目录存在。"""
    create_dirs([path])


def make_run_name(args) -> str:
    """生成本次实验的名字；用户传入 --run-name 时优先使用用户指定值。"""
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.dataset}_{args.model}_seed{args.seed}_{timestamp}"


def make_run_dir(output_dir: str, run_name: str) -> str:
    """把输出根目录和运行名拼成最终结果目录路径。"""
    return os.path.join(output_dir, run_name)


def setup_logger(run_dir: str) -> Tuple[logging.Logger, str]:
    """配置同时写入终端和 run.log 的日志器。"""
    ensure_dir(run_dir)
    log_path = os.path.join(run_dir, "run.log")
    logger = logging.getLogger("node_classification")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger, log_path


def row_normalize(features: np.ndarray) -> np.ndarray:
    """对节点特征做按行归一化，使每个节点的特征和为 1。"""
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return features / row_sum


def _dataset_path(data_root: str, dataset: str) -> str:
    """定位某个数据集目录，不存在时直接报错。"""
    path = os.path.join(data_root, dataset)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return path


def _edge_file(data_path: str, dataset: str) -> str:
    """寻找 citation 边文件，兼容 dataset/xxx.cites 和 dataset/origin/xxx.cites。"""
    candidates = [
        os.path.join(data_path, f"{dataset}.cites"),
        os.path.join(data_path, "origin", f"{dataset}.cites"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No citation edge file found for {dataset}: {candidates}")


def _load_id_map(data_path: str) -> Dict[str, int]:
    """读取原始论文 ID 到连续整数节点编号的映射。"""
    with open(os.path.join(data_path, "id_map.json"), "r", encoding="utf8") as fp:
        raw_map = json.load(fp)
    return {str(node_id): int(index) for node_id, index in raw_map.items()}


def _load_features(data_path: str) -> torch.Tensor:
    """读取 features.npy，并转换成 PyTorch 的 float32 特征张量。"""
    features = np.load(os.path.join(data_path, "features.npy")).astype(np.float32)
    features = row_normalize(features)
    return torch.from_numpy(features)


def _load_labels(data_path: str, id_map: Dict[str, int]) -> Tuple[torch.Tensor, List[str]]:
    """读取类别标签，并把字符串标签编码成 0 到 num_classes-1 的整数。"""
    with open(os.path.join(data_path, "class_map.json"), "r", encoding="utf8") as fp:
        class_map = json.load(fp)

    label_names = sorted({str(label).strip() for label in class_map.values()})
    label_to_index = {label: index for index, label in enumerate(label_names)}
    labels = torch.full((len(id_map),), -1, dtype=torch.long)

    for raw_node_id, raw_label in class_map.items():
        labels[id_map[str(raw_node_id)]] = label_to_index[str(raw_label).strip()]

    if (labels < 0).any():
        missing_count = int((labels < 0).sum())
        raise ValueError(f"{missing_count} nodes have no class label.")

    return labels, label_names


def _load_edge_index(data_path: str, dataset: str, id_map: Dict[str, int]) -> torch.Tensor:
    """读取引用边，并转成 PyG 常用的 edge_index 张量。"""
    path = _edge_file(data_path, dataset)
    edges = []
    with open(path, "r", encoding="utf8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            src = id_map.get(parts[0])
            dst = id_map.get(parts[1])
            if src is None or dst is None:
                continue
            # 引用关系在这里按无向图处理，所以正反两个方向都加入。
            edges.append((src, dst))
            edges.append((dst, src))

    if not edges:
        raise ValueError(f"No valid edges were loaded from {path}")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.unique(edge_index, dim=1)


def build_masks(
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按类别分层随机划分训练、验证和测试节点。"""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    generator = torch.Generator().manual_seed(seed)
    num_nodes = labels.numel()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for label in labels.unique(sorted=True):
        class_indices = torch.where(labels == label)[0]
        # 每个类别内部单独打乱，避免某些类别只出现在 train/val/test 的某一部分。
        permuted = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        train_end = max(1, int(class_indices.numel() * train_ratio))
        val_count = max(1, int(class_indices.numel() * val_ratio))
        val_end = min(train_end + val_count, class_indices.numel() - 1)

        train_mask[permuted[:train_end]] = True
        val_mask[permuted[train_end:val_end]] = True
        test_mask[permuted[val_end:]] = True

    return train_mask, val_mask, test_mask


def normalized_sparse_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """构造 GCN 常用的 D^(-1/2) A D^(-1/2) 稀疏邻接矩阵。"""
    device = edge_index.device
    # 加自环后，每个节点在聚合时也会保留自己的原始特征信息。
    loops = torch.arange(num_nodes, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loops], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    row, col = edge_index
    degree = torch.bincount(row, minlength=num_nodes).float()
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    values = degree_inv_sqrt[row] * degree_inv_sqrt[col]

    adjacency = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    return adjacency.coalesce()


def load_graph_data(args) -> GraphData:
    """从磁盘读取数据集，并构造训练所需的 GraphData。"""
    data_path = _dataset_path(args.data_root, args.dataset)
    id_map = _load_id_map(data_path)
    x = _load_features(data_path)
    y, label_names = _load_labels(data_path, id_map)
    edge_index = _load_edge_index(data_path, args.dataset, id_map)
    train_mask, val_mask, test_mask = build_masks(y, args.train_ratio, args.val_ratio, args.seed)
    adj_norm = normalized_sparse_adjacency(edge_index, x.size(0))

    return GraphData(
        name=args.dataset,
        x=x,
        y=y,
        edge_index=edge_index,
        adj_norm=adj_norm,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        label_names=label_names,
    )


def move_to_device(data: GraphData, device: torch.device) -> GraphData:
    """把 GraphData 中所有张量移动到 CPU/GPU/MPS，标签名等 Python 对象保持不变。"""
    return GraphData(
        name=data.name,
        x=data.x.to(device),
        y=data.y.to(device),
        edge_index=data.edge_index.to(device),
        adj_norm=data.adj_norm.to(device),
        train_mask=data.train_mask.to(device),
        val_mask=data.val_mask.to(device),
        test_mask=data.test_mask.to(device),
        label_names=data.label_names,
    )


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """先把模型 logits 转成类别概率，再计算分类准确率。"""
    if labels.numel() == 0:
        return 0.0
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    return (predictions == labels).float().mean().item()


def f1_scores(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    """先把模型 logits 转成类别概率，再手动计算 macro-F1 和 weighted-F1。"""
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    macro_scores = []
    weighted_sum = 0.0
    total = labels.numel()

    for class_id in range(num_classes):
        # 对每个类别分别计算 TP/FP/FN，再得到该类别的 precision、recall 和 F1。
        pred_positive = predictions == class_id
        true_positive_mask = labels == class_id
        tp = (pred_positive & true_positive_mask).sum().item()
        fp = (pred_positive & ~true_positive_mask).sum().item()
        fn = (~pred_positive & true_positive_mask).sum().item()
        support = true_positive_mask.sum().item()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        macro_scores.append(f1)
        weighted_sum += f1 * support

    macro_f1 = float(sum(macro_scores) / num_classes)
    weighted_f1 = float(weighted_sum / total) if total > 0 else 0.0
    return macro_f1, weighted_f1
