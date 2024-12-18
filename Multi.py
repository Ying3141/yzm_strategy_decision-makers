import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# 自定义数据集类
class MultiLabelDataset(Dataset):
    def __init__(self, filepath, num_classes, _sheet_name, normalize=True):
        """
        初始化数据集。根据需要进行归一化。

        Args:
            filepath (str): Excel 文件路径。
            num_classes (int): 多标签的类别数量。
            _sheet_name (str): 需要加载的 Excel 工作表名称。
            normalize (bool): 是否对特征进行归一化，默认为 True。
        """
        # 从 Excel 文件读取数据
        data = pd.read_excel(filepath, sheet_name=_sheet_name)
        data = data.to_numpy()

        # 假设最后 num_classes 列为标签，其余列为特征
        self.features = data[:, :-num_classes]
        self.labels = data[:, -num_classes:]  # one-hot 编码标签

        # 如果需要归一化，对特征进行标准化处理
        if normalize:
            self.mean = self.features.mean(axis=0)
            self.std = self.features.std(axis=0)
            self.features = (self.features - self.mean) / self.std  # 标准化

        # 将数据转换为 PyTorch 的 Tensor 格式
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.features)

    def __getitem__(self, idx):
        """根据索引返回对应的特征和标签。"""
        return self.features[idx], self.labels[idx]


# 定义模型
class MultiLabelClassifier(nn.Module):
    """
    多标签分类器模型，采用全连接神经网络。

    Attributes:
        fc1 (nn.Linear): 第一层全连接层。
        relu (nn.ReLU): ReLU 激活函数。
        fc2 (nn.Linear): 第二层全连接层。
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化模型。

        Args:
            input_size (int): 输入特征的维度。
            hidden_size (int): 隐藏层的节点数量。
            output_size (int): 输出的类别数量。
        """
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播过程。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出 logits（未经过 Sigmoid）。
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 测试模型
def evaluate_model(model, dataloader, threshold=0.5):
    """
    评估模型性能，通过阈值计算准确率。

    Args:
        model (nn.Module): 已训练的模型。
        dataloader (DataLoader): 测试数据的 DataLoader。
        threshold (float): 分类阈值，默认为 0.5。
    """
    model.eval()  # 切换为评估模式
    all_preds = []
    all_targets = []

    # 禁用梯度计算
    with torch.no_grad():
        for features, labels in dataloader:
            logits = model(features)
            preds = (torch.sigmoid(logits) > threshold).float()  # 应用 Sigmoid 和阈值化
            all_preds.append(preds)
            all_targets.append(labels)

    # 拼接所有预测值和目标值
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # 计算准确率
    accuracy = (preds == targets).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")

# 保存归一化参数（均值和标准差）
def save_normalization_params(mean, std, filepath="normalization_params.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump((mean, std), f)
        print(f"Normalization parameters saved to {filepath}")


# 训练过程的损失记录
loss_history = []

# 数据集加载与划分
filepath = "E://OneDrive//伊之密工作//工作汇报//2024-11//Train.xlsx"  # 替换为实际路径
# sheet_name = "Sheet8（加入人工判断尺寸特征）"
sheet_name = "Sheet6（合并优化模式）"
dataset = MultiLabelDataset(filepath, 4, sheet_name, normalize=True)


# 参数设置
input_size = dataset.features.shape[1]  # 特征维度
num_classes = dataset.labels.shape[1]  # 类别数量
hidden_size = 32  # 隐藏层大小
learning_rate = 0.005  # 学习率
num_epochs = 200  # 训练轮数
batch_size = 2  # 批大小
test_split_ratio = 0.30  # 测试集比例

# 数据集划分
total_size = len(dataset)
test_size = int(total_size * test_split_ratio)
train_size = total_size - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader 用于加载数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = MultiLabelClassifier(input_size, hidden_size, num_classes)
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 模型训练
for epoch in range(num_epochs):
    model.train()  # 切换为训练模式
    epoch_loss = 0.0

    for features, labels in train_loader:
        outputs = model(features)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        epoch_loss += loss.item()  # 累加损失

    avg_loss = epoch_loss / len(train_loader)  # 计算平均损失
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 绘制训练损失曲线
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid()
plt.show()

# 模型评估
print("Evaluating on Test Set:")
evaluate_model(model, test_loader)

# 模型评估
print("Evaluating on all data:")
evaluate_model(model, data_loader)

# 保存模型
model_save_path = "multi_label_classifier.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# 保存归一化参数
save_normalization_params(dataset.mean, dataset.std)
