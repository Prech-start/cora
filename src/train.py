import torch
import process

# 超参数定义
LEARNING_RATE = 0.1  # 学习率
WEIGHT_DACAY = 5e-4  # 正则化系数
EPOCHS = 200  # 完整遍历训练集的次数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备

train_adjacency = process.train_adjacency
train_features = process.train_features
valid_adjacency = process.valid_adjacency
valid_features = process.valid_features
train_labels = process.train_labels
valid_labels = process.valid_labels

