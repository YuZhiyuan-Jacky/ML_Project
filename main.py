import torch
from args import parse_args
from model import build_model
from tools import set_device, set_seed
from trainer import Trainer
from utils import load_graph_data, make_run_dir, make_run_name, move_to_device, setup_logger


def main():
    """程序入口：解析参数、加载数据和模型，然后启动训练评估流程。"""
    args = parse_args()
    # 每次运行都单独创建一个输出目录，保存日志、指标和测试集预测结果。
    args.run_name = make_run_name(args)
    args.run_dir = make_run_dir(args.output_dir, args.run_name)
    logger, log_path = setup_logger(args.run_dir)

    # 固定随机种子，尽量让数据划分、参数初始化和训练过程可复现。
    set_seed(args.seed)
    device = torch.device(set_device(args.gpu, use_cuda=args.device == "cuda"))
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Run dir: {args.run_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Arguments: {vars(args)}")

    data = load_graph_data(args)
    data = move_to_device(data, device)
    # 输入维度来自节点特征数，输出维度来自类别数。
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
    # 训练和评估模型，返回最佳验证检查点的指标。
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

    runtime_profile = metrics.get("runtime_profile", {})
    if runtime_profile:
        logger.info(
            "Runtime profile: "
            f"training_seconds={runtime_profile['training_seconds']:.4f}, "
            f"avg_epoch_seconds={runtime_profile['avg_epoch_seconds']:.4f}, "
            f"test_full_graph_seconds={runtime_profile['test_full_graph_seconds']:.6f}, "
            f"test_ms_per_node={runtime_profile['test_ms_per_node']:.6f}"
        )

    memory_profile = metrics.get("memory_profile", {})
    if memory_profile:
        if memory_profile.get("cuda_memory_available"):
            logger.info(
                "CUDA memory profile: "
                f"training_peak_allocated_mb={memory_profile['training_peak_allocated_mb']:.2f}, "
                f"single_node_test_peak_allocated_mb={memory_profile['single_node_test_peak_allocated_mb']:.2f}"
            )
        else:
            logger.info("CUDA memory profile: unavailable on CPU device.")


if __name__ == "__main__":
    main()
