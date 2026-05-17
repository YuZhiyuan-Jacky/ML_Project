import numpy as np
import json
from os import path

data_name = 'cora'
current_path = path.dirname(__file__)
print(current_path)


data_path = path.join(current_path, 'origin')
if not path.exists(data_path):
    data_path = current_path

idx_features_labels = np.genfromtxt(path.join(data_path, f'{data_name}.content'), dtype=np.dtype(str))
edges_unordered = np.genfromtxt(path.join(data_path, f'{data_name}.cites'), dtype=np.dtype(str))
features = idx_features_labels[:, 1:-1]
features = features.astype(np.float32)
print(edges_unordered.shape)
print(features.shape)
np.save(path.join(current_path, 'features.npy'), features)
id_map = {i[0]: int(index) for index, i in enumerate(idx_features_labels)}
class_map = {i[0]: i[-1] for i in idx_features_labels}
with open(path.join(current_path, 'id_map.json'), 'w') as f:
    json.dump(id_map, f)
    f.close()
with open(path.join(current_path, 'class_map.json'), 'w') as f:
    json.dump(class_map, f)
    f.close()
