import numpy as np
import json
from os import path

# 本脚本把 Cora 的原始 .content/.cites 文件转换成训练代码直接读取的格式。
data_name = 'cora'
current_path = path.dirname(__file__)
print(current_path)


# origin/ 目录存在时优先读取原始数据目录，否则兼容数据文件直接放在当前目录的情况。
data_path = path.join(current_path, 'origin')
if not path.exists(data_path):
    data_path = current_path

# .content 每行包含：论文 ID、词袋特征、类别标签。
# .cites 每行包含：一条论文引用边。
idx_features_labels = np.genfromtxt(path.join(data_path, f'{data_name}.content'), dtype=np.dtype(str))
edges_unordered = np.genfromtxt(path.join(data_path, f'{data_name}.cites'), dtype=np.dtype(str))
features = idx_features_labels[:, 1:-1]
features = features.astype(np.float32)
print(edges_unordered.shape)
print(features.shape)

# 保存节点特征矩阵、原始 ID 到连续编号的映射、原始 ID 到类别名的映射。
np.save(path.join(current_path, 'features.npy'), features)
id_map = {i[0]: int(index) for index, i in enumerate(idx_features_labels)}
class_map = {i[0]: i[-1] for i in idx_features_labels}
with open(path.join(current_path, 'id_map.json'), 'w') as f:
    json.dump(id_map, f)
    f.close()
with open(path.join(current_path, 'class_map.json'), 'w') as f:
    json.dump(class_map, f)
    f.close()
