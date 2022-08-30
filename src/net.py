import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # 定义可学习参数
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        # 
        H_ = torch.mm(input_feature, self.weight)
        # adjacency 稀疏矩阵 使用 sparse的mm
        out_put = torch.sparse.mm(adjacency, H_)
        if self.use_bias:
            out_put += self.bias
        return out_put


class Net(nn.Module):
    def __init__(self, input_dim=1433):
        super(Net).__init__()
        self.gcn1 = GCN(input_dim=input_dim, output_dim=16)
        self.gcn2 = GCN(input_dim=16, output_dim=7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        out = self.gcn2(adjacency, h)
        return out
