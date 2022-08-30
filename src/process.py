import pandas as pd
import numpy as np
import os
import scipy.sparse as sp


def normalization_laplacian(adjacent):
    adjacent += np.eye(adjacent.shape[0])
    degree_matrix = np.array(adjacent.sum(1))
    d_hat = np.diag(np.power(degree_matrix, -0.5))
    return d_hat.dot(adjacent).dot(d_hat)


path = os.path.dirname(__file__)
raw_data = pd.read_csv(os.path.join(path, '..', 'data', 'cora', 'cora.content'), sep='\t', header=None)
num = raw_data.shape[0]
# 建立元数据到[0,num]的映射
raw_index_to_ = dict(zip(list(raw_data[0]), list(raw_data.index)))
# 获取特征值
features = raw_data.iloc[:, 1:-1].to_numpy()
# 获取label
labels = raw_data.iloc[:, -1]
# 将label制作成onehot存储
labels = pd.get_dummies(labels).to_numpy()
# 读取cites文件
raw_data_cites = pd.read_csv(os.path.join(path, '..', 'data', 'cora', 'cora.cites'), sep='\t', header=None)
matrix_ = np.zeros((num, num))
# 初始化邻接矩阵
if not os.path.exists('whole_adjacency.npy'):
    for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
        x = raw_index_to_[i]
        y = raw_index_to_[j]
        matrix_[x, y] = matrix_[y, x] = 1
    # 创建归一化拉普拉斯矩阵
    matrix_ = normalization_laplacian(matrix_)
    np.save('whole_adjacency', np.asarray(matrix_).astype(np.int8))
else:
    matrix_ = np.load('whole_adjacency.npy')

print(matrix_.shape)

train_len = num // 10 * 8
valid_len = num - train_len
train_adjacency = matrix_[:train_len, :train_len]
valid_adjacency = matrix_[train_len:, train_len:]
train_features = features[:train_len, ...]
valid_features = features[train_len:, ...]
train_labels = labels[:train_len, ...]
valid_labels = labels[train_len:, ...]

print()
