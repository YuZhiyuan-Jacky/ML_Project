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
    def __init__(self, model, data, args, logger=None):
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
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_cuda_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _cuda_peak_allocated_mb(self):
        if self.device.type != "cuda":
            return None
        self._synchronize_device()
        return float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

    def train_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.data)
        loss = F.cross_entropy(logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate_split(self, mask: torch.Tensor) -> Dict[str, float]:
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
        return {
            "train": self.evaluate_split(self.data.train_mask),
            "val": self.evaluate_split(self.data.val_mask),
            "test": self.evaluate_split(self.data.test_mask),
        }

    @torch.no_grad()
    def collect_test_outputs(self) -> Dict[str, object]:
        self.model.eval()
        logits = self.model(self.data)
        probabilities = F.softmax(logits, dim=-1)
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
        self.model.eval()
        test_count = int(self.data.test_mask.sum().item())

        self._reset_cuda_peak_memory()
        self._synchronize_device()
        start_time = time.perf_counter()
        logits = self.model(self.data)
        test_predictions = logits[self.data.test_mask].argmax(dim=-1)
        if self.device.type == "cuda":
            _ = test_predictions.detach()
        self._synchronize_device()
        elapsed_seconds = time.perf_counter() - start_time

        seconds_per_node = elapsed_seconds / test_count if test_count > 0 else 0.0
        return {
            "test_num_nodes": test_count,
            "test_full_graph_seconds": float(elapsed_seconds),
            "test_seconds_per_node": float(seconds_per_node),
            "test_ms_per_node": float(seconds_per_node * 1000.0),
            "single_node_test_peak_allocated_mb": self._cuda_peak_allocated_mb(),
            "measurement_note": (
                "Full-batch GNN inference computes logits for the whole graph; "
                "single-node test time is averaged as full-graph forward time divided by test node count."
            ),
        }

    def fit(self) -> Dict[str, Dict[str, float]]:
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
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_state = self._snapshot_state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch == 1 or epoch % self.args.log_every == 0:
                progress.set_postfix(
                    loss=f"{train_loss:.4f}",
                    train_acc=f"{metrics['train']['acc']:.4f}",
                    val_acc=f"{metrics['val']['acc']:.4f}",
                    test_acc=f"{metrics['test']['acc']:.4f}",
                    best_val=f"{self.best_val_acc:.4f}",
                )

            if bad_epochs >= self.args.patience:
                progress.close()
                self.logger.info(f"Early stopping at epoch {epoch}; best validation epoch was {self.best_epoch}.")
                break

        self._synchronize_device()
        training_seconds = time.perf_counter() - train_start_time
        training_peak_allocated_mb = self._cuda_peak_allocated_mb()
        completed_epochs = len(history)

        if self.best_state is not None:
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
