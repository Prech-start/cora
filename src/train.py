import torch
import process
import net
import train_utils

# 超参数定义
LEARNING_RATE = 1e-2  # 学习率
WEIGHT_DACAY = 5e-4  # 正则化系数
EPOCHS = 3000  # 完整遍历训练集的次数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备

train_adjacency = process.train_adjacency
train_features = process.train_features
valid_adjacency = process.valid_adjacency
valid_features = process.valid_features
train_labels = process.train_labels
valid_labels = process.valid_labels
train_features = train_features / train_features.sum(1, keepdims=True)
valid_features = valid_features / valid_features.sum(1, keepdims=True)


def to_tensor(feature):
    return torch.FloatTensor(feature)


model = net.Net().to(DEVICE)

# 模型定义：Model, Loss, Optimizer
criterion = torch.nn.CrossEntropyLoss().to(DEVICE)  # 多分类交叉熵损失
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DACAY)  # Adam优化器

train_features = to_tensor(train_features).to(DEVICE)
train_adjacency = to_tensor(train_adjacency).to(DEVICE)
train_labels = to_tensor(train_labels).to(DEVICE)

valid_features = to_tensor(valid_features)
valid_adjacency = to_tensor(valid_adjacency)
valid_labels = to_tensor(valid_labels)


def acc(model, criterion):
    model.cpu()
    with torch.no_grad():  # 关闭求导
        logits = model(valid_adjacency, valid_features)  # 所有数据作前向传播
        loss = criterion(logits, valid_labels)
        pred = torch.argmax(logits, 1)
        real = torch.argmax(valid_labels, 1)
        return loss.item(), torch.sum(pred == real) / pred.shape[0]


# 训练主体函数
def train():
    loss_history = []
    valid_loss_history = []
    val_acc_history = []
    model.train()  # 训练模式
    for epoch in range(EPOCHS):  # 完整遍历一遍训练集 一个epoch做一次更新
        model.to(DEVICE)
        optimizer.zero_grad()  # 清空梯度
        pred = model(train_adjacency, train_features)  # 所有数据前向传播 （N,7）
        loss = criterion(pred, train_labels)  # 计算损失值
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
        valid_loss, valid_acc = acc(model, criterion)
        if epoch == 150:
            print()
        print("Epoch {:03d}: train_Loss {:.4f},valid_loss {:.4f}, valid_acc{:.2f}".format(
            epoch, loss.item(), valid_loss, valid_acc))
        loss_history.append(loss.item())
        valid_loss_history.append(valid_loss)
        val_acc_history.append(valid_acc)

    return loss_history, val_acc_history, valid_loss_history


loss_history, val_acc_history, valid_loss_history = train()
train_utils.save_loss(loss_history, valid_loss_history, val_acc_history)
