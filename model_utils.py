import torch
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


# 加载模型函数
def load_model(model_path, input_size, hidden_size, num_classes):
    """
    从指定路径加载训练好的模型。

    Args:
        model_path (str): 模型文件路径。
        input_size (int): 输入特征的维度。
        hidden_size (int): 隐藏层的节点数量。
        num_classes (int): 输出标签的数量。

    Returns:
        model (MultiLabelClassifier): 加载好的模型。
    """
    model = MultiLabelClassifier(input_size, hidden_size, num_classes)  # 初始化模型
    model.load_state_dict(torch.load(model_path))  # 加载模型参数
    model.eval()  # 切换模型为评估模式
    return model


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
        if 1.0 not in predictions:
            predictions[0][probabilities.argmax()] = 2
    return probabilities, predictions
