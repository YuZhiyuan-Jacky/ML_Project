import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import json
import logging
import os
import time
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import accuracy, ensure_dir, f1_scores


class Trainer:
    """封装一次节点分类实验的训练、验证、测试和结果保存流程。"""

    def __init__(self, model, data, args, logger=None):
        """保存模型/数据/参数，并初始化 Adam 优化器和早停相关状态。"""
        self.model = model
        self.data = data
        self.args = args
        self.logger = logger or logging.getLogger("node_classification")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.best_state = None
        self.best_val_acc = -1.0
        self.best_epoch = 0
        self.device = next(self.model.parameters()).device

    def _snapshot_state_dict(self):
        """复制当前模型参数到 CPU，作为最佳验证集 checkpoint。"""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _synchronize_device(self) -> None:
        """在 CUDA 上等待前面的异步计算结束，保证计时和显存统计准确。"""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_cuda_peak_memory(self) -> None:
        """清空 CUDA 峰值显存统计，方便单独测量某一段代码的显存峰值。"""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _cuda_peak_allocated_mb(self):
        """返回当前统计窗口内的 CUDA 峰值显存；CPU/MPS 运行时返回 None。"""
        if self.device.type != "cuda":
            return None
        self._synchronize_device()
        return float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

    def train_epoch(self) -> float:
        """训练一个 epoch，并返回训练集交叉熵损失。"""
        self.model.train()
        self.optimizer.zero_grad()
        # 全图 GNN 一次前向会计算所有节点 logits，loss 只在 train_mask 上取。
        logits = self.model(self.data)
        loss = F.cross_entropy(logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate_split(self, mask: torch.Tensor) -> Dict[str, float]:
        """在给定 mask 对应的数据划分上计算 loss、accuracy 和 F1。"""
        self.model.eval()
        logits = self.model(self.data)
        split_logits = logits[mask]
        split_labels = self.data.y[mask]
        loss = F.cross_entropy(split_logits, split_labels).item()
        macro_f1, weighted_f1 = f1_scores(split_logits, split_labels, self.data.num_classes)
        return {
            "loss": float(loss),
            "acc": accuracy(split_logits, split_labels),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    @torch.no_grad()
    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """一次性评估训练集、验证集和测试集。"""
        return {
            "train": self.evaluate_split(self.data.train_mask),
            "val": self.evaluate_split(self.data.val_mask),
            "test": self.evaluate_split(self.data.test_mask),
        }

    @torch.no_grad()
    def collect_test_outputs(self) -> Dict[str, object]:
        """整理测试集每个节点的真实标签、预测标签、logits 和概率。"""
        self.model.eval()
        logits = self.model(self.data)
        probabilities = F.softmax(logits, dim=-1)  # 单标签多分类：把每个节点的类别分数转成概率分布。
        test_indices = torch.where(self.data.test_mask)[0]
        test_logits = logits[test_indices].detach().cpu()
        test_probabilities = probabilities[test_indices].detach().cpu()
        test_labels = self.data.y[test_indices].detach().cpu()
        test_predictions = test_logits.argmax(dim=-1)
        node_indices = test_indices.detach().cpu().tolist()

        records = []
        for row, node_index in enumerate(node_indices):
            gt = int(test_labels[row].item())
            pred = int(test_predictions[row].item())
            # records 会写入 test_outputs.json，便于演示时展示具体样本预测。
            records.append(
                {
                    "node_index": int(node_index),
                    "ground_truth": gt,
                    "ground_truth_label": self.data.label_names[gt],
                    "prediction": pred,
                    "prediction_label": self.data.label_names[pred],
                    "logits": [float(value) for value in test_logits[row].tolist()],
                    "probabilities": [float(value) for value in test_probabilities[row].tolist()],
                }
            )

        return {
            "run_name": self.args.run_name,
            "dataset": self.args.dataset,
            "model": self.args.model,
            "seed": self.args.seed,
            "best_epoch": self.best_epoch,
            "label_names": self.data.label_names,
            "num_test_nodes": len(records),
            "records": records,
        }

    @torch.no_grad()
    def profile_test_inference(self) -> Dict[str, object]:
        """统计测试阶段完整预测流程耗时，并换算平均每个测试节点耗时。"""
        self.model.eval()
        test_count = int(self.data.test_mask.sum().item())

        self._reset_cuda_peak_memory()
        self._synchronize_device()
        start_time = time.perf_counter()
        logits = self.model(self.data)
        test_probabilities = F.softmax(logits[self.data.test_mask], dim=-1)
        test_predictions = test_probabilities.argmax(dim=-1)
        if self.device.type == "cuda":
            # 触发 CUDA 张量实际 materialize，避免计时只包含调度开销。
            _ = test_predictions.detach()
        self._synchronize_device()
        elapsed_seconds = time.perf_counter() - start_time

        # 节点分类是 full-batch 推理：一次把整张图输入模型，输出所有节点类别，再筛选 test 集合。
        # 因此这里的单节点耗时是总预测耗时除以测试节点数得到的平均摊销值。
        seconds_per_node = elapsed_seconds / test_count if test_count > 0 else 0.0
        return {
            "test_num_nodes": test_count,
            "test_full_graph_seconds": float(elapsed_seconds),
            "test_seconds_per_node": float(seconds_per_node),
            "test_ms_per_node": float(seconds_per_node * 1000.0),
            "single_node_test_peak_allocated_mb": self._cuda_peak_allocated_mb(),
            "measurement_note": (
                "Full-batch GNN inference computes logits for the whole graph, then applies softmax on test nodes; "
                "single-node test time is averaged as full prediction time divided by test node count."
            ),
        }

    def fit(self) -> Dict[str, Dict[str, float]]:
        """完整训练入口：循环训练、早停、恢复最佳模型、保存指标和预测结果。"""
        bad_epochs = 0
        history = []
        self._reset_cuda_peak_memory()
        self._synchronize_device()
        train_start_time = time.perf_counter()

        progress = tqdm(
            range(1, self.args.epochs + 1),
            desc="Training",
            dynamic_ncols=True,
            leave=True,
        )

        for epoch in progress:
            train_loss = self.train_epoch()
            metrics = self.evaluate_all()
            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss
            history.append(metrics)

            val_acc = metrics["val"]["acc"]
            if val_acc > self.best_val_acc:
                # 只根据验证集 accuracy 选最佳模型，测试集只用于最后汇报。
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_state = self._snapshot_state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1

            # 第 1 个 epoch 先显示一次训练状态，之后每隔 log_every 个 epoch 更新一次进度条指标。
            if epoch == 1 or epoch % self.args.log_every == 0:
                progress.set_postfix(
                    loss=f"{train_loss:.4f}",
                    train_acc=f"{metrics['train']['acc']:.4f}",
                    val_acc=f"{metrics['val']['acc']:.4f}",
                    test_acc=f"{metrics['test']['acc']:.4f}",
                    best_val=f"{self.best_val_acc:.4f}",
                )

            if bad_epochs >= self.args.patience:
                # patience 个 epoch 验证集没有提升时提前停止训练。
                progress.close()
                self.logger.info(f"Early stopping at epoch {epoch}; best validation epoch was {self.best_epoch}.")
                break

        self._synchronize_device()
        training_seconds = time.perf_counter() - train_start_time
        training_peak_allocated_mb = self._cuda_peak_allocated_mb()
        completed_epochs = len(history)

        if self.best_state is not None:
            # 最终指标统一使用验证集表现最好的 checkpoint 来评估。
            self.model.load_state_dict(self.best_state)

        final_metrics = self.evaluate_all()
        final_metrics["best_epoch"] = self.best_epoch
        final_metrics["best_val_acc"] = self.best_val_acc
        inference_profile = self.profile_test_inference()
        runtime_profile = {
            "device": str(self.device),
            "epochs_completed": completed_epochs,
            "training_seconds": float(training_seconds),
            "avg_epoch_seconds": float(training_seconds / completed_epochs) if completed_epochs else 0.0,
            "test_num_nodes": inference_profile["test_num_nodes"],
            "test_full_graph_seconds": inference_profile["test_full_graph_seconds"],
            "test_seconds_per_node": inference_profile["test_seconds_per_node"],
            "test_ms_per_node": inference_profile["test_ms_per_node"],
            "measurement_note": inference_profile["measurement_note"],
        }
        memory_profile = {
            "device": str(self.device),
            "cuda_memory_available": self.device.type == "cuda",
            "training_peak_allocated_mb": training_peak_allocated_mb,
            "single_node_test_peak_allocated_mb": inference_profile["single_node_test_peak_allocated_mb"],
            "measurement_note": (
                "CUDA memory is reported as torch.cuda.max_memory_allocated. "
                "For full-batch GNN inference, single-node test memory is the peak allocation of one full-graph forward."
            ),
        }
        final_metrics["runtime_profile"] = runtime_profile
        final_metrics["memory_profile"] = memory_profile

        if not self.args.no_save_results:
            # results.json 保存聚合指标和训练历史；test_outputs.json 保存逐节点预测。
            ensure_dir(self.args.run_dir)
            results_path = os.path.join(self.args.run_dir, "results.json")
            payload = {
                "run_name": self.args.run_name,
                "dataset": self.args.dataset,
                "model": self.args.model,
                "seed": self.args.seed,
                "best_epoch": self.best_epoch,
                "best_val_acc": self.best_val_acc,
                "metrics": final_metrics,
                "runtime_profile": runtime_profile,
                "memory_profile": memory_profile,
                "history": history,
            }
            with open(results_path, "w", encoding="utf8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved results to {results_path}")

            test_outputs_path = os.path.join(self.args.run_dir, "test_outputs.json")
            with open(test_outputs_path, "w", encoding="utf8") as fp:
                json.dump(self.collect_test_outputs(), fp, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved test outputs to {test_outputs_path}")

        if self.args.save_model:
            ensure_dir(self.args.run_dir)
            model_path = os.path.join(self.args.run_dir, "best.pt")
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Saved best model checkpoint to {model_path}")

        return final_metrics
