import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import torch
from torch import nn
import torch.nn.functional as F


class BasicAdjGNN(nn.Module):
    """基于归一化稀疏邻接矩阵的基础 GNN。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        """创建多层线性变换；每层前都先做一次邻居信息聚合。"""
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1]) for index in range(num_layers)
        )
        self.dropout = dropout

    def forward(self, data):
        """前向传播：用 A_norm @ X 聚合邻居，再经过线性层和激活函数。"""
        x = data.x
        for index, layer in enumerate(self.layers):
            # data.adj_norm 是 D^(-1/2) A D^(-1/2)，用于 GCN 风格的邻居平均。
            x = torch.sparse.mm(data.adj_norm, x)
            x = layer(x)
            if index != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PyGGCN(nn.Module):
    """使用 torch_geometric.nn.GCNConv 实现的 GCN。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        """按层数堆叠 PyG 的 GCNConv 卷积层。"""
        super().__init__()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="An issue occurred while importing \'torch-sparse\'.*",
                category=UserWarning,
            )
            from torch_geometric.nn import GCNConv

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.convs = nn.ModuleList(
            GCNConv(dimensions[index], dimensions[index + 1]) for index in range(num_layers)
        )
        self.dropout = dropout

    def forward(self, data):
        """前向传播：GCNConv 根据 edge_index 聚合邻居并更新节点表示。"""
        x = data.x
        for index, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            if index != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PyGGraphTransformer(nn.Module):
    """使用 PyG TransformerConv 实现的图 Transformer。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        heads: int,
    ):
        """构建多头注意力图卷积层，隐藏层输出按 head 拼接。"""
        # 先初始化 nn.Module 的内部状态；所有 PyTorch 模型子类都需要调用。
        super().__init__()
        # PyG 导入时可能提示 torch-sparse 的可选依赖警告，这里只屏蔽这类非致命提示。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="An issue occurred while importing \'torch-sparse\'.*",
                category=UserWarning,
            )
            # 延迟导入 TransformerConv：只有真正选择 graph_transformer 时才需要 PyG 这一层。
            from torch_geometric.nn import TransformerConv

        # 至少要有一层图卷积，否则模型没有可训练的图变换层。
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        # 注意力头数必须是正数，多头注意力才有意义。
        if heads < 1:
            raise ValueError("heads must be at least 1.")
        # 多层模型的隐藏层会把多个 head 的输出拼接起来，所以 hidden_dim 必须能被 heads 整除。
        if num_layers > 1 and hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for graph_transformer.")

        # 记录 dropout 概率，forward 中每个非最后一层后都会使用。
        self.dropout = dropout
        # ModuleList 用来保存多层 TransformerConv，并让 PyTorch 正确注册这些子模块参数。
        self.convs = nn.ModuleList()

        # 单层模型直接从输入维度映射到类别数，不再设置隐藏层。
        if num_layers == 1:
            # heads=1 且 concat=False：输出维度就是 output_dim，便于直接做分类 logits。
            self.convs.append(TransformerConv(input_dim, output_dim, heads=1, concat=False, dropout=dropout))
            # 单层已经构造完成，提前返回，避免继续添加隐藏层和输出层。
            return

        # 每个 attention head 输出 per_head_hidden 维，heads 个 head 拼接后总维度等于 hidden_dim。
        per_head_hidden = hidden_dim // heads
        # 第一层负责把原始节点特征 input_dim 投影到隐藏表示 hidden_dim。
        self.convs.append(
            # concat=True 表示多个 head 的输出拼接：per_head_hidden * heads = hidden_dim。
            TransformerConv(input_dim, per_head_hidden, heads=heads, concat=True, dropout=dropout)
        )
        # 中间层数量是 num_layers - 2：去掉第一层和最后分类输出层。
        for _ in range(num_layers - 2):
            self.convs.append(
                # 中间层输入/输出都保持 hidden_dim，便于堆叠更深的图注意力变换。
                TransformerConv(hidden_dim, per_head_hidden, heads=heads, concat=True, dropout=dropout)
            )
        # 最后一层从隐藏维度映射到类别数，作为每个节点的分类 logits。
        # concat=False 且 heads=1：输出维度严格等于 output_dim，而不是再拼接多头。
        self.convs.append(TransformerConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, data):
        """前向传播：TransformerConv 在图边上计算注意力并更新节点表示。"""
        x = data.x
        for index, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            if index != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def build_model(args, input_dim: int, output_dim: int) -> nn.Module:
    """根据 --model 参数创建对应的模型实例。"""
    if args.model == "basic_gnn":
        return BasicAdjGNN(input_dim, args.hidden_dim, output_dim, args.num_layers, args.dropout)
    if args.model == "gcn":
        return PyGGCN(input_dim, args.hidden_dim, output_dim, args.num_layers, args.dropout)
    if args.model == "graph_transformer":
        return PyGGraphTransformer(
            input_dim,
            args.hidden_dim,
            output_dim,
            args.num_layers,
            args.dropout,
            args.heads,
        )
    raise ValueError(f"Unknown model: {args.model}")
