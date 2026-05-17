import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class GraphData:
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
        return self.x.size(0)

    @property
    def num_features(self) -> int:
        return self.x.size(1)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device_name: str, gpu: int = 0) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device(f"cuda:{gpu}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def row_normalize(features: np.ndarray) -> np.ndarray:
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return features / row_sum


def _dataset_path(data_root: str, dataset: str) -> str:
    path = os.path.join(data_root, dataset)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return path


def _edge_file(data_path: str, dataset: str) -> str:
    candidates = [
        os.path.join(data_path, f"{dataset}.cites"),
        os.path.join(data_path, "origin", f"{dataset}.cites"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No citation edge file found for {dataset}: {candidates}")


def _load_id_map(data_path: str) -> Dict[str, int]:
    with open(os.path.join(data_path, "id_map.json"), "r", encoding="utf8") as fp:
        raw_map = json.load(fp)
    return {str(node_id): int(index) for node_id, index in raw_map.items()}


def _load_features(data_path: str) -> torch.Tensor:
    features = np.load(os.path.join(data_path, "features.npy")).astype(np.float32)
    features = row_normalize(features)
    return torch.from_numpy(features)


def _load_labels(data_path: str, id_map: Dict[str, int]) -> Tuple[torch.Tensor, List[str]]:
    with open(os.path.join(data_path, "class_map.json"), "r", encoding="utf8") as fp:
        class_map = json.load(fp)

    label_names = sorted({str(label) for label in class_map.values()})
    label_to_index = {label: index for index, label in enumerate(label_names)}
    labels = torch.empty(len(id_map), dtype=torch.long)

    for raw_node_id, raw_label in class_map.items():
        labels[id_map[str(raw_node_id)]] = label_to_index[str(raw_label)]

    return labels, label_names


def _load_edge_index(data_path: str, dataset: str, id_map: Dict[str, int]) -> torch.Tensor:
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
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    generator = torch.Generator().manual_seed(seed)
    num_nodes = labels.numel()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for label in labels.unique(sorted=True):
        class_indices = torch.where(labels == label)[0]
        permuted = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        train_end = max(1, int(class_indices.numel() * train_ratio))
        val_count = max(1, int(class_indices.numel() * val_ratio))
        val_end = min(train_end + val_count, class_indices.numel() - 1)

        train_mask[permuted[:train_end]] = True
        val_mask[permuted[train_end:val_end]] = True
        test_mask[permuted[val_end:]] = True

    return train_mask, val_mask, test_mask


def normalized_sparse_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index.device
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
    if labels.numel() == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean().item()


def f1_scores(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    predictions = logits.argmax(dim=-1)
    macro_scores = []
    weighted_sum = 0.0
    total = labels.numel()

    for class_id in range(num_classes):
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
