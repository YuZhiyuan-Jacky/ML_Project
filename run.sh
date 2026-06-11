# 依次在三个 citation 数据集上运行三种模型，便于统一对比实验结果。
# 每条命令都会在 outputs/ 下生成独立的 run.log、results.json 和 test_outputs.json。

# Cora 数据集：论文分类的小规模常用基准。
python main.py --dataset cora --model basic_gnn
python main.py --dataset cora --model gcn
python main.py --dataset cora --model graph_transformer

# Citeseer 数据集：同样是 citation network，类别和图结构与 Cora 不同。
python main.py --dataset citeseer --model basic_gnn
python main.py --dataset citeseer --model gcn
python main.py --dataset citeseer --model graph_transformer

# PubMed 数据集：节点更多，用来观察模型在更大图上的表现和耗时。
python main.py --dataset pubmed --model basic_gnn
python main.py --dataset pubmed --model gcn
python main.py --dataset pubmed --model graph_transformer
