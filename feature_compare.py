import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shap  # 新增：用于计算特征重要性


# 自定义数据集类
class MultiLabelDataset(Dataset):
    def __init__(self, filepath, num_classes, _sheet_name, normalize=True):
        self.filepath = filepath
        self._sheet_name = _sheet_name
        data = pd.read_excel(filepath, sheet_name=_sheet_name)
        data = data.to_numpy()

        self.features = data[:, :-num_classes]
        self.labels = data[:, -num_classes:]  # one-hot 编码标签

        if normalize:
            self.mean = self.features.mean(axis=0)
            self.std = self.features.std(axis=0)
            self.features = (self.features - self.mean) / self.std  # 标准化

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 定义模型
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 评估模型性能
def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels in dataloader:
            logits = model(features)
            preds = (torch.sigmoid(logits) > threshold).float()
            all_preds.append(preds)
            all_targets.append(labels)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    accuracy = (preds == targets).float().mean().item()
    return accuracy


# 保存归一化参数
def save_normalization_params(mean, std, filepath="normalization_params.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump((mean, std), f)


# 训练模型
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return loss_history


# SHAP 特征重要性分析
def shap_feature_importance(model, dataset):
    print("Calculating SHAP values for feature importance...")
    inputs = dataset.features.numpy()

    explainer = shap.DeepExplainer(model, torch.tensor(inputs, dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(inputs, dtype=torch.float32))

    shap.summary_plot(shap_values, inputs)


# 置零法特征重要性分析
def zero_feature_importance(model, dataset, input_size, hidden_size, num_classes, learning_rate, num_epochs, batch_size, test_split_ratio):
    print("Evaluating original model performance...")
    original_train_size = int(len(dataset) * (1 - test_split_ratio))
    train_dataset, test_dataset = random_split(dataset, [original_train_size, len(dataset) - original_train_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 原始模型性能
    train_model(model, train_loader, optimizer, criterion, num_epochs)
    original_accuracy = evaluate_model(model, test_loader)

    feature_importances = []
    for i in range(input_size):
        print(f"Evaluating model without feature {i + 1}...")
        reduced_features = dataset.features.clone()
        reduced_features[:, i] = 0.0  # 移除第 i 个特征
        reduced_dataset = MultiLabelDataset(dataset.filepath, dataset.labels.shape[1], dataset._sheet_name, normalize=False)
        reduced_dataset.features = reduced_features

        train_dataset, test_dataset = random_split(reduced_dataset, [original_train_size, len(reduced_dataset) - original_train_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 重新初始化模型
        reduced_model = MultiLabelClassifier(input_size, hidden_size, num_classes)
        optimizer = optim.Adam(reduced_model.parameters(), lr=learning_rate)

        train_model(reduced_model, train_loader, optimizer, criterion, num_epochs)
        reduced_accuracy = evaluate_model(reduced_model, test_loader)

        importance = original_accuracy - reduced_accuracy
        feature_importances.append(importance)

    plt.bar(range(1, input_size + 1), feature_importances)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Feature Importance (Zeroing Method)")
    plt.show()


# 主流程
filepath = "E://OneDrive//伊之密工作//工作汇报//2024-11//Train.xlsx"
sheet_name = "Sheet6（合并优化模式）"
dataset = MultiLabelDataset(filepath, 4, sheet_name, normalize=True)

input_size = dataset.features.shape[1]
num_classes = dataset.labels.shape[1]
hidden_size = 32
learning_rate = 0.005
num_epochs = 150
batch_size = 1
test_split_ratio = 0.30

# 数据集划分
train_size = int(len(dataset) * (1 - test_split_ratio))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = MultiLabelClassifier(input_size, hidden_size, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# 训练模型
loss_history = train_model(model, train_loader, optimizer, criterion, num_epochs)
plt.plot(range(1, num_epochs + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# 评估模型
test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 特征重要性分析
shap_feature_importance(model, dataset)
zero_feature_importance(model, dataset, input_size, hidden_size, num_classes, learning_rate, num_epochs, batch_size, test_split_ratio)
