import pickle
import os
import torch
import numpy as np
import random
import warnings
import time
from sklearn.exceptions import UndefinedMetricWarning
import platform

def set_device(device_num, use_cuda=False):
    system = platform.system()
    if torch.cuda.is_available() and use_cuda:
        if system == "Linux" or system == "Windows":
            device = 'cuda:'+str(device_num)
        elif system == "Darwin":
            device = 'mps:'+str(device_num)
    else:
        device = 'cpu'
    return device

def set_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_dirs(paths: list) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)

def revers_dict(dic):
    # begin = time.time()
    res = dict(zip(dic.values(), dic.keys()))
    # print(f'B cost: {(time.time() - begin):.2f}s')
    return res

def save_dict(dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dict(file_path):
    with open(file_path, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def sort_dict(dic, sample):
    # 指定用于排序的参考列表的键
    sort_key = sample

    # 获取参考列表
    ref_list = dic[sort_key]

    # 获取参考列表的排序索引
    sorted_indices = sorted(range(len(ref_list)), key=lambda k: ref_list[k])

    # 根据排序索引重新排序每一个列表
    sorted_data = {key: [value[i] for i in sorted_indices] for key, value in dic.items()}

    return sorted_data

def multi_file_presence_state(path_list, file_name_list):
    # 只要有一个文件不存在就返回false
    for index, path in enumerate(path_list):
        for file in file_name_list[index]:
            if not os.path.exists(os.path.join(path, file)):
                print(f'Not Exist: {os.path.join(path, file)}')
                return False
    return True


# 定义一个装饰器，用于包装警告信息的显示
def suppress_warnings(max_time=30):
    # 设置一个标志变量，用于记录是否已经打印过警告
    warning_printed = False
    # 设置一个时间戳变量，用于记录上次打印警告的时间
    last_warning_time = time.time()
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal warning_printed, last_warning_time
            current_time = time.time()
            if not warning_printed or (current_time - last_warning_time) > max_time:
                if not warning_printed:
                    warning_printed = True
                    last_warning_time = current_time
                warnings.simplefilter("always", category=UndefinedMetricWarning)
                result = func(*args, **kwargs)
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                return result
            return func(*args, **kwargs)
        return wrapper
    return decorator
    
def calculate_f1(predicted_set, true_set):
    """
    计算 F1 分数。
    
    参数:
        predicted_set (set): 预测输出节点集合，元素为 int 类型。
        true_set (set): 标签节点集合，元素为 int 类型。
    
    返回:
        float: F1 分数。
    """
    # print(predicted_set, true_set)
    # 计算真正例（TP）
    true_positives = len(predicted_set.intersection(true_set))
    # 计算预测为正例的数量（P）
    predicted_positives = len(predicted_set)
    # 计算实际为正例的数量（T）
    actual_positives = len(true_set)
    # 计算精确率（Precision）
    if predicted_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / predicted_positives
    # 计算召回率（Recall）
    if actual_positives == 0:
        recall = 0.0
    else:
        recall = true_positives / actual_positives
    # 计算 F1 分数
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1, precision, recall
