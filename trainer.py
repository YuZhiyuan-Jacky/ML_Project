import copy
import json
import os
from typing import Dict

import torch
import torch.nn.functional as F

from utils import accuracy, ensure_dir, f1_scores


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.best_state = None
        self.best_val_acc = -1.0
        self.best_epoch = 0

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

    def fit(self) -> Dict[str, Dict[str, float]]:
        bad_epochs = 0
        history = []

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch()
            metrics = self.evaluate_all()
            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss
            history.append(metrics)

            val_acc = metrics["val"]["acc"]
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch == 1 or epoch % self.args.log_every == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"loss {train_loss:.4f} | "
                    f"train_acc {metrics['train']['acc']:.4f} | "
                    f"val_acc {metrics['val']['acc']:.4f} | "
                    f"test_acc {metrics['test']['acc']:.4f}"
                )

            if bad_epochs >= self.args.patience:
                print(f"Early stopping at epoch {epoch}; best validation epoch was {self.best_epoch}.")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        final_metrics = self.evaluate_all()
        final_metrics["best_epoch"] = self.best_epoch
        final_metrics["best_val_acc"] = self.best_val_acc

        if self.args.save_model:
            ensure_dir(self.args.output_dir)
            model_path = os.path.join(self.args.output_dir, f"{self.args.dataset}_{self.args.model}_best.pt")
            metrics_path = os.path.join(self.args.output_dir, f"{self.args.dataset}_{self.args.model}_metrics.json")
            torch.save(self.model.state_dict(), model_path)
            with open(metrics_path, "w", encoding="utf8") as fp:
                json.dump(final_metrics, fp, ensure_ascii=False, indent=2)

        return final_metrics
