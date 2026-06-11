import argparse


def parse_args():
    """解析命令行参数，并在真正训练前做基础合法性检查。"""
    parser = argparse.ArgumentParser(description="Graph node classification experiments.")

    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="basic_gnn",
        choices=["basic_gnn", "gcn", "graph_transformer"],
        help="basic_gnn uses sparse adjacency matrix multiplication; gcn and graph_transformer use PyG layers.",
    )

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--heads", type=int, default=4, help="Attention heads for graph_transformer.")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Runtime device preference. cuda delegates to tools.set_device and falls back when unavailable.",
    )
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name used for the output subdirectory.")
    parser.add_argument("--no-save-results", action="store_true", help="Do not save final metrics to JSON.")
    parser.add_argument("--save-model", action="store_true", help="Save the best validation checkpoint.")
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()
    if args.num_layers < 1:
        parser.error("--num-layers must be at least 1")
    if args.hidden_dim < 1:
        parser.error("--hidden-dim must be at least 1")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    return args
