from args import parse_args
from model import build_model
from trainer import Trainer
from tools import set_seed
from utils import get_device, load_graph_data, make_run_dir, make_run_name, move_to_device, setup_logger


def main():
    args = parse_args()
    args.run_name = make_run_name(args)
    args.run_dir = make_run_dir(args.output_dir, args.run_name)
    logger, log_path = setup_logger(args.run_dir)

    set_seed(args.seed)
    device = get_device(args.device, args.gpu)
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Run dir: {args.run_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Arguments: {vars(args)}")

    data = load_graph_data(args)
    data = move_to_device(data, device)
    model = build_model(args, data.num_features, data.num_classes).to(device)

    logger.info(
        f"Dataset: {data.name} | nodes: {data.num_nodes} | "
        f"features: {data.num_features} | classes: {data.num_classes}"
    )
    logger.info(f"Model: {args.model} | Device: {device}")
    logger.info(
        f"Split sizes: train={int(data.train_mask.sum())}, "
        f"val={int(data.val_mask.sum())}, test={int(data.test_mask.sum())}"
    )

    trainer = Trainer(model, data, args, logger)
    metrics = trainer.fit()

    logger.info("Final metrics from the best validation checkpoint:")
    for split in ["train", "val", "test"]:
        split_metrics = metrics[split]
        logger.info(
            f"{split}: loss={split_metrics['loss']:.4f}, "
            f"acc={split_metrics['acc']:.4f}, "
            f"macro_f1={split_metrics['macro_f1']:.4f}, "
            f"weighted_f1={split_metrics['weighted_f1']:.4f}"
        )
    logger.info(f"best_epoch={metrics['best_epoch']}, best_val_acc={metrics['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
