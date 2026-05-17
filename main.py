from args import parse_args
from model import build_model
from trainer import Trainer
from utils import get_device, load_graph_data, move_to_device, set_seed


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device, args.gpu)

    data = load_graph_data(args)
    data = move_to_device(data, device)
    model = build_model(args, data.num_features, data.num_classes).to(device)

    print(
        f"Dataset: {data.name} | nodes: {data.num_nodes} | "
        f"features: {data.num_features} | classes: {data.num_classes}"
    )
    print(f"Model: {args.model} | Device: {device}")
    print(
        f"Split sizes: train={int(data.train_mask.sum())}, "
        f"val={int(data.val_mask.sum())}, test={int(data.test_mask.sum())}"
    )

    trainer = Trainer(model, data, args)
    metrics = trainer.fit()

    print("Final metrics from the best validation checkpoint:")
    for split in ["train", "val", "test"]:
        split_metrics = metrics[split]
        print(
            f"{split}: loss={split_metrics['loss']:.4f}, "
            f"acc={split_metrics['acc']:.4f}, "
            f"macro_f1={split_metrics['macro_f1']:.4f}, "
            f"weighted_f1={split_metrics['weighted_f1']:.4f}"
        )
    print(f"best_epoch={metrics['best_epoch']}, best_val_acc={metrics['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
