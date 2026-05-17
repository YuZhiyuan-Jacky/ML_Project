import os
import json
import random
import torch
import numpy as np
import networkx as nx
from ogb.nodeproppred import NodePropPredDataset
import scipy.sparse as sp
import time
import math
from tqdm.auto import tqdm, trange
from tools import load_dict


def normalize(matrix, dim):
    """
    根据给定的 axis 对输入的矩阵进行归一化操作。

    参数：
    - matrix：输入的矩阵
    - axis：指定归一化的轴，取值为 0 或 1，0 表示按列归一化，1 表示按行归一化

    返回值：
    - normalized_matrix：归一化后的矩阵
    """
    if dim == 0:
        sums = matrix.sum(axis=0)  # 求每列的和
        # 避免除以零的情况
        sums[sums == 0] = 1
        # 对矩阵进行归一化操作
        normalized_matrix = matrix / sums[np.newaxis, :]
    elif dim == 1:
        sums = matrix.sum(axis=1)  # 求每行的和e
        # 避免除以零的情况
        sums[sums == 0] = 1
        # 对矩阵进行归一化操作
        normalized_matrix = matrix / sums[:, np.newaxis]
    else:
        raise ValueError("Invalid axis value. Axis must be 0 or 1.")

    return normalized_matrix

def Get_trussness(Graph):
    print('Getting trussness...')
    Graph.remove_edges_from(nx.selfloop_edges(Graph))
    k=1
    num = len(Graph.nodes())
    id_trussness = {}
    for i in range(num):
        id_trussness[i] = 0
    while True:
        temp = []
        k_truss = nx.k_truss(G=Graph, k=k)
        for i in nx.connected_components(k_truss):
            # sum += len(i)
            # print(len(i))
            temp.extend(list(i))
        if len(temp)==0:
            # print("End Lask k:", k)
            break
        for i in temp:
            id_trussness[i] = k
        k += 1
    # print(self.id_trussness)

    # 反转dict
    max_trussness = max(id_trussness.values())
    trussness_ids = {i:[] for i in range(max_trussness+1)}
    trussness_ids[0] = list(nx.isolates(Graph))
    for node, trussness in id_trussness.items():
        trussness_ids[trussness].append(node)
    # shuffle the id list
    for ids in trussness_ids.values():
        random.shuffle(ids)
    return id_trussness, trussness_ids

def Get_coreness(Graph):
    print('Getting coreness...')
    Graph.remove_edges_from(nx.selfloop_edges(Graph))
    id_coreness = nx.core_number(Graph)
    # make coreness_ids
    max_coreness = max(id_coreness.values())
    coreness_ids = {i:[] for i in range(max_coreness+1)}
    coreness_ids[0] = list(nx.isolates(Graph))
    for node, coreness in id_coreness.items():
        coreness_ids[coreness].append(node)
    # shuffle the id list
    for ids in coreness_ids.values():
        random.shuffle(ids)
    return id_coreness, coreness_ids

def compute_h_index(G, node):
    # 获取节点的邻居节点的度数
    degrees = [G.degree(neighbor) for neighbor in G.neighbors(node)]
    # 对度数进行降序排序
    degrees.sort(reverse=True)
    
    # 计算 h-index
    h_index = 0
    for i, degree in enumerate(degrees, start=1):
        if degree >= i:
            h_index = i
        else:
            break
    
    return h_index

class GraphData():
    def __init__(self, dataname, datarootpath, all=False) -> None:
        self.dataName = dataname
        self.rootpath = datarootpath
        self.communityType = all
        self.data_path = os.path.join(self.rootpath, self.dataName)
        if not os.path.exists(self.data_path):
            print(f'ERROR: No {self.dataName} exist! Please check the name and path!')
        
        self.NodeNum = 0
        self.EdgeNum = 0
        self.FeatureShape = 0
        self.CommunityNum = 0

        self.Edge = None  # 除非一定要用，否则不用
        self.Adj = None  # 除非一定要用，否则不用

        # node information
        self.name_id = None
        self.id_name = None
        self.name_label = {}
        self.id_label = {}

        # community information
        self.label_ids = {}
        self.label_set = set()
        
        self.Graph = None
        self.Subgraph = None
        self.Features = None

        self.id_trussness = None
        self.trussness_ids = None
        self.coreness_ids = None
        self.id_coreness = None
    
    def Load(self, all=None):
        if all is not None:
            self.communityType = all
        communityList = ['amazon', 'dblp', 'friendster', 'livejournal', 'orkut', 'demo', 'MAG_CS', 'MAG_Chemistry', 'MAG_Engineering', 'youtube']
        if any(item in self.dataName for item in communityList):
            self.LoadLarges()
            self.LoadCommunity_graph()
        elif self.dataName in ['ogbn-products', 'ogbn-proteins', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-papers100M']:
            self.LoadOGB()
        else:
            self.LoadfromSet()

    def FeatureGeneration(self, feature_lenth, type='origin'):
        if os.path.exists(os.path.join(self.data_path, f'features_{type}.npy')):
            self.Features = np.load(os.path.join(self.data_path, f'features_{type}.npy'), allow_pickle=True).astype(np.float32)
        else:
            if type == 'origin':
                pass
            elif type == 'adj':
                # ADJ feature
                self.Features = np.zeros((self.NodeNum, self.NodeNum), dtype=np.float32)
                self.FeatureShape = self.NodeNum
                for node, line in enumerate(tqdm(self.Features, desc='Generate {}.Feature'.format(self.dataName), ncols=100)):
                    neighbor = list(self.Graph.neighbors(node))
                    # print(neighbor, type(neighbor))
                    self.Features[node][neighbor] = 1
                    # print(self.Features[node])
                    # break
            elif type == 'h_index':
                h_indices = {}
                for node in self.Graph.nodes:
                    h_indices[node] = compute_h_index(self.Graph, node)
                max_h_index = max(list(h_indices.values())) + 1
                self.Features = []
                for i in trange(self.NodeNum, desc='Making Features', ncols=100):
                    feature = np.zeros(max_h_index)
                    # neighbor = list(self.Graph.neighbors(i))
                    neighbor = list(nx.ego_graph(self.Graph, i, 1).nodes())
                    for n in neighbor:
                        feature[h_indices[n]] += 1
                    self.Features.append(feature)
                self.Features = np.array(self.Features, dtype=np.float32)
            elif type == 'random':
                self.Features = np.random.random_sample(size=(self.NodeNum, feature_lenth), dtype=np.float32)
            elif type == 'GT_based':
                if feature_lenth < self.CommunityNum*10:
                    gtfeaturelenth = 10
                else:
                    gtfeaturelenth = math.floor(feature_lenth/self.CommunityNum)
                self.Features = np.zeros((self.NodeNum, gtfeaturelenth*self.CommunityNum), dtype=np.float32)
                for comid, nodes in tqdm(self.label_ids.items(), desc='Generate {}.Feature'.format(self.dataName), ncols=100):
                    for n in nodes:
                        # print((comid*gtfeaturelenth, (comid+1)*gtfeaturelenth-1), math.floor(gtfeaturelenth*0.5))
                        oneids = random.sample(range(comid*gtfeaturelenth, (comid+1)*gtfeaturelenth-1), math.floor(gtfeaturelenth*0.5))
                        self.Features[n][oneids] = 1
            elif type == 'GT_add':
                self.Features = np.zeros((self.NodeNum, feature_lenth), dtype=np.float32)
                for comid, nodes in tqdm(self.label_ids.items(), desc='Generate {}.Feature'.format(self.dataName), ncols=100):
                    random_vector = np.random.random_sample(size=(feature_lenth,)).astype(np.float32)
                    for n in nodes:
                        self.Features[n] += random_vector
            elif type == 'ATC':
                self.Features = np.zeros((self.NodeNum, feature_lenth), dtype=np.float32)
                for key, nodes in self.label_ids.items():
                    com_Feature = random.sample(range(feature_lenth), 3)
                    selected_nodes = random.sample(nodes, math.floor(len(nodes)*0.8))
                    self.Features[np.ix_(selected_nodes, com_Feature)] = np.float32(1)
                        
            np.save(os.path.join(self.data_path, f'features_{type}.npy'), self.Features)

        self.FeatureShape = self.Features.shape[1]
        
    def LoadOGB(self):
        # print('Loading {}...'.format(self.dataName))
        ogb_dataset = NodePropPredDataset(name=self.dataName, root=self.rootpath)
        graph = ogb_dataset.graph
        labels = ogb_dataset.labels
        # nodes & edges
        edges_np = graph['edge_index'].T
        self.EdgeNum = edges_np.shape[0]
        self.NodeNum = graph['num_nodes']
        nodes = [i for i in trange(self.NodeNum, desc='Loading {}.Nodes'.format(self.dataName), ncols=100)]
        self.id_name = {index:i for index, i in enumerate(tqdm(nodes, desc='Functing {}.id_name'.format(self.dataName), ncols=100))}
        self.name_id = {i:index for index, i in enumerate(tqdm(nodes, desc='Functing {}.name_id'.format(self.dataName), ncols=100))}
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(tqdm(nodes, desc='Add {}.Nodes'.format(self.dataName), ncols=100))
        self.Graph.add_edges_from(tqdm(edges_np, desc='Add {}.Edges'.format(self.dataName), ncols=100))
        del nodes
        del edges_np
        # features
        self.Features = graph['node_feat'].astype(np.float32)
        self.FeatureShape = self.Features.shape[1]
        del graph
        # labels
        for index, item in enumerate(tqdm(labels, desc='Loading {}.Lables'.format(self.dataName), ncols=100)):
            self.name_label[index] = item[0]
            self.id_label[self.name_id.get(index)] = item[0]
            self.label_set.add(item[0])
            if item[0] in self.label_ids.keys():
                self.label_ids.get(item[0]).append(index)
            else:
                self.label_ids[item[0]] = [index]
        self.CommunityNum = len(list(self.label_set))
        # shuffle the id list
        for ids in self.label_ids.values():
            random.shuffle(ids)
        del labels

    def LoadfromSet(self):
        # print('Loading {}...'.format(self.dataName))
        # nodes
        with open(os.path.join(self.data_path, 'id_map.json'), 'r', encoding='utf8') as fp:
            self.name_id = json.load(fp)
            self.id_name = {id:name for name, id in self.name_id.items()}
            nodes = list(self.name_id.values())
            self.NodeNum = len(nodes)
            fp.close()

        # 获取完整进度
        with open(os.path.join(self.data_path, '{}.cites').format(self.dataName), 'r') as file:
            num_lines = sum(1 for line in file)
            file.close()
        file_tqdm = tqdm(total=num_lines, desc='Loading {}.Edges'.format(self.dataName), ncols=100)
        # edges
        with open(os.path.join(self.data_path, '{}.cites').format(self.dataName), 'r') as fp:
            edges = []
            line = fp.readline()
            while line:
                line = line.split()
                edges.append([self.name_id.get(line[0]), self.name_id.get(line[1])])
                file_tqdm.update(1)
                # time.sleep(1)
                line = fp.readline()
            self.EdgeNum = len(edges)
            file_tqdm.close()
            fp.close()
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(nodes)
        self.Graph.add_edges_from(edges)
        del nodes
        del edges
        # features
        # a = time.time()
        self.Features = np.load(os.path.join(self.data_path, 'features.npy')).astype(np.float32)
        # self.Features = self.Features.astype(np.float32)
        # b = time.time()
        # print('Read Feature Time:', b-a)
        self.FeatureShape = self.Features.shape[1]
        # c = time.time()
        self.Features = normalize(self.Features, dim=1)  # 归一化特征
        # d = time.time()
        # print('Normalize Feature Time:', d-c)
        # labels
        with open(os.path.join(self.data_path, 'class_map.json'), 'r', encoding='utf8') as fp:
            self.name_label = json.load(fp)
            self.name_label = {key:[value] for key, value in self.name_label.items()}
            self.id_label = {self.name_id.get(key):value for key, value in self.name_label.items()}
            self.label_set = set([j for i in self.name_label.values() for j in i])
            self.CommunityNum = len(list(self.label_set))
            self.label_ids = {i:[] for i in self.label_set}
            for key, value in tqdm(self.name_label.items(), desc='Loading {}.Label...'.format(self.dataName), ncols=100):
                for single_label in value:
                    # time.sleep(1)
                    self.label_ids[single_label].append(self.name_id.get(key))
            # shuffle the id list
            for ids in self.label_ids.values():
                random.shuffle(ids)
            fp.close()

    def LoadLarges(self):
        # print('Loading {}...'.format(self.dataName))
        self.Graph = nx.Graph()
        # read nodes
        self.name_id = load_dict(os.path.join(self.data_path, 'name_id.pkl'))
        self.id_name = load_dict(os.path.join(self.data_path, 'id_name.pkl'))
        # self.NodeNum = len(self.id_name.keys())
        begin = time.time()

        # # 获取完整进度
        with open(os.path.join(self.data_path, 'com-{}.ungraph.txt'.format(self.dataName)), 'r') as file:
            num_lines = sum(1 for line in file)
            file.close()

        # read edges
        edge_tqdm = tqdm(total=num_lines, desc='Loading edges for {}'.format(self.dataName), ncols=100)
        with open(os.path.join(self.data_path, 'com-{}.ungraph.txt'.format(self.dataName)), 'r') as f:
            line = f.readline()
            while line:
                edge = line.split()
                if '#' in edge or len(edge) > 2:
                    # print('skip format')
                    edge_tqdm.update(1)
                    line = f.readline()
                    continue
                single_edge_name = (edge[0], edge[1])
                single_edge_id = (self.name_id.get(edge[0]), self.name_id.get(edge[1]))
                self.Graph.add_edge(self.name_id.get(edge[0]), self.name_id.get(edge[1]))
                edge_tqdm.update(1)
                line = f.readline()
            f.close()
            edge_tqdm.close()
        self.NodeNum = len(self.Graph.nodes())
        self.EdgeNum = len(self.Graph.edges())

    def LoadCommunity_graph(self):
        # all measueres whether load all communities
        if self.communityType:
            all = 'all'
        else:
            all = 'top5000'
        communities = []

        # 获取完整进度
        with open(os.path.join(self.data_path, 'com-{}.{}.cmty.txt'.format(self.dataName, all)), 'r') as file:
            num_lines = sum(1 for line in file)
            file.close()
        file_tqdm = tqdm(total=num_lines, desc='Loading {}.Community'.format(self.dataName), ncols=100)

        # read community
        with open(os.path.join(self.data_path, 'com-{}.{}.cmty.txt'.format(self.dataName, all)), 'r') as f:
            line = f.readline()
            while line:
                communities.append(line.split())
                file_tqdm.update(1)
                # time.sleep(1)
                line = f.readline()
            f.close()
        # print('Community Num:', len(communities))
        communities = sorted(communities, key=len, reverse=True)
        # print('Biggest Community Scale', len(communities[0]))
        # print('Smallest Community Scale', len(communities[-1]))
        self.label_ids = {}
        self.name_label = {}
        for index, line in enumerate(tqdm(communities, desc='Mapping', ncols=100)):
            self.label_set.add(index)
            self.label_ids[index] = [self.name_id.get(node) for node in line]
            for node in line:
                if node in self.name_label:
                    self.name_label[node].add(index)
                    self.id_label[self.name_id.get(node)].add(index)
                else:
                    self.name_label[node] = set([index])
                    self.id_label[self.name_id.get(node)] = set([index])
        self.CommunityNum = len(list(self.label_set))
        return communities

    def MakeAdj(self):
        # For those modes which need a adjacency as input, this function should be used before cuda()
        if self.Graph is None:
            self.MakeGraph()
        self.Adj = np.array(nx.adjacency_matrix(self.Graph).todense(), dtype=np.float32)
        return self.Adj

    def Get_Edges(self):
        print('Making edges...')
        self.Graph.add_edges_from(nx.selfloop_edges(self.Graph))
        self.Edge = np.array(self.Graph.edges(), dtype=np.int64)
        print('Edges Get!')
        # print(self.Edge.dtype)
        return self.Edge

    def toTensor(self):
        if self.Adj is not None:
            self.Adj = torch.tensor(self.Adj)
        if self.Features is not None:
            self.Features = torch.tensor(self.Features)
        if self.Edge is not None:
            self.Edge = torch.tensor(self.Edge)

if __name__ == '__main__':
    dn = ['cora', 'citeseer', 'pubmed', 'MAG_Chemistry', 'MAG_CS', 'MAG_Engineering', 'reddit','dblp', 'orkut', 'livejournal']
    di = {}
    for d in dn:
        data = GraphData(d, '/mnt/HDD2/yzy23/Data', True)
        data.Load()
        di[d] = data.NodeNum
    print(di)
    # MAG_Chemistry, MAG_CS, MAG_Engineering
    # data = GraphData(dataname='MAG_Chemistry', datarootpath='/mnt/HDD2/yzy23/Data', all=True)
    # data.Load()
    # data.FeatureGeneration(feature_lenth=0,type='origin')
    # print(type(data.Features))
    # print(data.Features.shape)
    # print(data.NodeNum)

    # for label, ids in data.label_ids.items():
    #     subg = data.Graph.subgraph(list(ids))
    #     print(nx.is_connected(subg))
        
    #     components = list(nx.connected_components(subg))

    #     # 计算每个连通分量的大小
    #     component_sizes = [len(component) for component in components]
    #     print(component_sizes)

    #     # 找到最大和最小的连通分量的大小
    #     max_size = max(component_sizes)
    #     min_size = min(component_sizes)
        # print(f'all nodes:{len(list(subg.nodes()))}, component num:{nx.number_connected_components(subg)}, biggest component:{max_size}, min component:{min_size}')
    # data.FeatureGeneration()
    # print(type(data.Features))
    # print(data.Features)