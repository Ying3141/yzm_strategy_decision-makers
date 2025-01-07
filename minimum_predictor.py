import torch
import pickle
import torch.nn as nn


# 定义多标签分类器模型结构
class MultiLabelClassifier(nn.Module):
    """
    多标签分类器模型，使用两层全连接层进行特征提取和分类。

    Attributes:
        fc1 (nn.Linear): 第一层全连接层。
        relu (nn.ReLU): ReLU 激活函数。
        fc2 (nn.Linear): 第二层全连接层，输出为 logits。
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化模型。

        Args:
            input_size (int): 输入特征的维度。
            hidden_size (int): 隐藏层的节点数量。
            output_size (int): 输出标签的数量（即类别数）。
        """
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入到隐藏层
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        """
        前向传播过程。

        Args:
            x (torch.Tensor): 输入特征张量。

        Returns:
            torch.Tensor: 输出 logits（未经 Sigmoid 激活的原始输出）。
        """
        x = self.fc1(x)  # 第一层全连接
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # 第二层全连接，得到 logits
        return x


# 加载完整预测模型
model_path = 'whole_model.pth'
model = torch.load(model_path)

# 加载归一化参数
normalization_params_path = "normalization_params.pkl"  # 模型参数路径


# 加载归一化参数：均值和标准差，用于归一化输入特征
def load_normalization_params(filepath=normalization_params_path):
    with open(filepath, 'rb') as f:
        mean, std = pickle.load(f)  # 从 pickle 文件中读取均值和标准差
    return mean, std


# 归一化函数：根据均值和标准差对特征进行标准化处理
def normalize(features, mean, std):
    return [(feature - m) / s for feature, m, s in zip(features, mean, std)]


# 预测函数
def predict(model, features, threshold=0.6):
    """
    对给定的特征进行预测，并返回预测的概率和标签。

    Args:
        model (MultiLabelClassifier): 已训练的模型。
        features (torch.Tensor): 输入的特征数据。
        threshold (float): 用于转换概率为二进制标签的阈值，默认为 0.5。

    Returns:
        probabilities (torch.Tensor): 模型输出的概率。
        predictions (torch.Tensor): 转换为二进制标签后的预测结果。
    """
    with torch.no_grad():  # 禁用梯度计算，节省内存
        logits = model(features)  # 获取模型输出的 logits
        probabilities = torch.sigmoid(logits)  # 对 logits 进行 Sigmoid 激活，得到概率
        predictions = (probabilities > threshold).float()  # 根据阈值转换为二进制标签

        # 如果没有超过阈值的结果，则将最大可能性的结果置2
        if 1.0 not in predictions:
            predictions[0][probabilities.argmax()] = 2
    return probabilities, predictions


features = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0.500, 0.700, 21.580, 65.000, 65.400, 155.000, 30.000, 0.714, 0.023, 0.006,
            0.994, 0.194]
mean, std = load_normalization_params()
normalized_features = normalize(features, mean, std)
features_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)
probabilities, predictions = predict(model, features_tensor)

# 预测概率：是算法输出的原始概率结果
# 预测结果：四个数字表示4种工艺策略的适用性: 1->推荐使用；2->可能适用；3->不推荐使用
# 预测结果的四个数字分别表示“动态VP切换模式”、“动态保压压力模式”、“动态VP+保压模式”、“标准模式”的预测适用性，注意标签的顺序！
print(f"概率分布：{probabilities}, 预测结果：{predictions}")

strategy = ["动态VP切换模式", "动态保压压力模式", "动态VP+保压模式", "标准模式"]
label = ['不适用', '适用', '可能适用']

for i in range(4):
    print(f"{strategy[i]}:{label[int(predictions[0][i])]}")
