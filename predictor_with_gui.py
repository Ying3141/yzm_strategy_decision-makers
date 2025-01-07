import tkinter as tk
from tkinter import messagebox
import torch
from model_utils import load_model, predict
import pickle

# 加载模型相关参数
model_path = "multi_label_classifier.pth"  # 加载完整预测模型
normalization_params_path = "normalization_params.pkl"  # 加载归一化参数

# 定义类别名称，用于显示模型预测的结果
class_labels = ["动态VP切换模式", "动态保压压力模式", "动态VP+保压模式", "标准模式"]

# 定义模型相关参数
input_size = 22  # 输入特征数，需要与训练时一致
hidden_size = 32  # 隐藏层大小
num_classes = 4  # 输出类别数

# 调用自定义的加载模型函数，加载训练好的模型
model = load_model(model_path, input_size, hidden_size, num_classes)

# 创建 tkinter 窗口
root = tk.Tk()
root.title("工艺策略预测器")  # 窗口标题
root.geometry("430x800")  # 设置窗口尺寸
root.config(bg="#f0f0f0")  # 设置背景颜色
root.resizable(width=False, height=False)  # 禁止改变窗口大小

# 设置统一的字体
font = ("宋体", 12)


# 加载归一化参数：均值和标准差，用于归一化输入特征
def load_normalization_params(filepath=normalization_params_path):
    with open(filepath, 'rb') as f:
        mean, std = pickle.load(f)  # 从 pickle 文件中读取均值和标准差
    return mean, std


# 归一化函数：根据均值和标准差对特征进行标准化处理
def normalize(features, mean, std):
    return [(feature - m) / s for feature, m, s in zip(features, mean, std)]


# 材料特征选择框（7个复选框）
material_vars = [tk.IntVar() for _ in range(7)]  # 创建7个复选框变量


# 设置材料特征的选择函数：确保只有一个复选框被选中
def set_material_feature(value, idx):
    for i, var in enumerate(material_vars):
        if i != idx:
            var.set(0)  # 取消选中其他复选框
    material_vars[idx].set(value)  # 设置当前复选框的值


# 创建材料特征选择框容器
material_feature_frame = tk.LabelFrame(root, text="材料特征", padx=10, pady=10, font=font, bg="#f0f0f0")
material_feature_frame.grid(row=0, column=0, padx=10, pady=15, sticky="ew")

# 创建7个复选框，表示不同的材料类型，并将其分为两排
materials = ["PA系", "PP", "PC", "ABS", "PC+ABS", "PMMA", "其他"]
for i, material in enumerate(materials):
    row = i // 4  # 第一排放前4个复选框，第二排放后3个
    col = i % 4  # 每一排最多4个复选框
    tk_btn = tk.Checkbutton(material_feature_frame, text=material, variable=material_vars[i],
                            command=lambda idx=i: set_material_feature(1, idx), font=font, bg="#f0f0f0")
    tk_btn.grid(row=row, column=col, padx=10, pady=5, sticky="w")

# 浇口特征选择框（点胶口和直浇口）
var_is_spot_gate = tk.IntVar()
var_is_straight_gate = tk.IntVar()


# 设置浇口特征的选择函数：确保只有一个选项被选中
def set_gate_feature(value, idx):
    if idx == 0:
        var_is_spot_gate.set(value)
        var_is_straight_gate.set(0)
    else:
        var_is_straight_gate.set(value)
        var_is_spot_gate.set(0)


# 创建浇口特征选择框容器
gate_feature_frame = tk.LabelFrame(root, text="浇口特征", padx=10, pady=10, font=font, bg="#f0f0f0")
gate_feature_frame.grid(row=1, column=0, padx=10, pady=15, sticky="ew")

# 创建浇口选择框
ckbt_is_spot_gate = tk.Checkbutton(gate_feature_frame, text="点胶口", variable=var_is_spot_gate,
                                   command=lambda: set_gate_feature(1, 0), font=font, bg="#f0f0f0")
ckbt_is_straight_gate = tk.Checkbutton(gate_feature_frame, text="直浇口", variable=var_is_straight_gate,
                                       command=lambda: set_gate_feature(1, 1), font=font, bg="#f0f0f0")

ckbt_is_spot_gate.grid(row=0, column=0, padx=10)
ckbt_is_straight_gate.grid(row=0, column=1, padx=10)

# 流道特征选择框（冷流道和热流道）
var_is_cold_runner = tk.IntVar()
var_is_hot_runner = tk.IntVar()


# 设置流道特征的选择函数：确保只有一个选项被选中
def set_runner_feature(value, idx):
    if idx == 0:
        var_is_cold_runner.set(value)
        var_is_hot_runner.set(0)
    else:
        var_is_hot_runner.set(value)
        var_is_cold_runner.set(0)


# 创建流道特征选择框容器
runner_feature_frame = tk.LabelFrame(root, text="流道特征", padx=10, pady=10, font=font, bg="#f0f0f0")
runner_feature_frame.grid(row=2, column=0, padx=10, pady=15, sticky="ew")

# 创建流道选择框
ckbt_is_cold_runner = tk.Checkbutton(runner_feature_frame, text="冷流道", variable=var_is_cold_runner,
                                     command=lambda: set_runner_feature(1, 0), font=font, bg="#f0f0f0")
ckbt_is_hot_runner = tk.Checkbutton(runner_feature_frame, text="热流道", variable=var_is_hot_runner,
                                    command=lambda: set_runner_feature(1, 1), font=font, bg="#f0f0f0")

ckbt_is_cold_runner.grid(row=0, column=0, padx=10)
ckbt_is_hot_runner.grid(row=0, column=1, padx=10)

# 工艺特征输入框
feature_labels = ["注射时间", "保压时间", "周期时间", "注射行程", "总行程", "峰压", "一段保压"]
entries = {}

# 创建工艺特征输入框容器
para_feature_frame = tk.LabelFrame(root, text="工艺特征", padx=10, pady=10, font=font, bg="#f0f0f0")
para_feature_frame.grid(row=3, column=0, padx=10, pady=15, sticky="ew")

# 为每个工艺特征创建输入框
for i, label in enumerate(feature_labels):
    row = i // 2
    col = i % 2

    label_widget = tk.Label(para_feature_frame, text=label, width=10, font=font, bg="#f0f0f0")
    entry_widget = tk.Entry(para_feature_frame, font=font, width=10)
    label_widget.grid(row=row, column=col * 2, padx=0, pady=5, sticky="ew")
    entry_widget.grid(row=row, column=col * 2 + 1, padx=0, pady=5, sticky="ew")

    entries[feature_labels[i]] = entry_widget

# 创建框架
prediction_frame = tk.LabelFrame(root, text="模式预测结果", padx=10, pady=10, font=font, bg="#f0f0f0")
prediction_frame.grid(row=5, column=0, padx=10, pady=15, sticky="ew")

# 创建一个列表来存储预测类别标签和颜色方块
prediction_labels = []
prediction_boxes = []

# 创建预测类别标签和对应的颜色方块
for i, label in enumerate(class_labels):
    row = i // 2
    col = i % 2

    color_box = tk.Label(prediction_frame, width=2, height=1, bg="white")  # 初始颜色为白色
    color_box.grid(row=row, column=col * 2, padx=10, pady=5)
    prediction_boxes.append(color_box)

    mode_label = tk.Label(prediction_frame, text=label, font=font, bg="#f0f0f0")
    mode_label.grid(row=row, column=col * 2 + 1, padx=10, pady=5, sticky="w")
    prediction_labels.append(mode_label)

# 创建颜色说明条
color_legend_frame = tk.Frame(root, bg="#f0f0f0")
color_legend_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

# 假设颜色和标签对应
color_legend = [("不适用", "red"), ("适用", "green"), ("可能适用", "yellow")]

# 创建颜色说明条
for i, (color_name, color_value) in enumerate(color_legend):
    # 颜色方块
    color_box = tk.Label(color_legend_frame, width=2, height=1, bg=color_value)
    color_box.grid(row=0, column=2 * i, padx=10, pady=0, sticky="e")  # 右对齐颜色方块

    # 标签文字，右对齐
    label = tk.Label(color_legend_frame, text=color_name, font=("宋体", 8), bg="#f0f0f0", anchor="e")
    label.grid(row=0, column=2 * i + 1, padx=10, pady=0, sticky="e")  # 右对齐文本


# 预测函数：获取用户输入，进行归一化并使用模型预测
def on_predict():
    features = []

    # 获取材料特征
    for var in material_vars[:6]:
        features.append(var.get())  # 取复选框的值（0 或 1）

    # 获取浇口特征
    features.append(var_is_spot_gate.get())
    features.append(var_is_straight_gate.get())

    # 获取流道特征
    features.append(var_is_cold_runner.get())
    features.append(var_is_hot_runner.get())

    # 获取工艺特征
    for name in feature_labels:
        try:
            value = float(entries[name].get())  # 转换输入的数值
            features.append(value)
        except ValueError:
            messagebox.showerror("错误", f"请确保特征 {name} 为有效的数值。")
            return

    # 计算一些额外的衍生特征
    features.append(float(entries["注射时间"].get()) / float(entries["保压时间"].get()))
    features.append(float(entries["注射时间"].get()) / float(entries["周期时间"].get()))
    features.append(
        (float(entries["总行程"].get()) - float(entries["注射行程"].get())) / float(entries["注射行程"].get()))
    features.append(float(entries["注射行程"].get()) / float(entries["总行程"].get()))
    features.append(float(entries["一段保压"].get()) / float(entries["峰压"].get()))

    # 检查特征数量是否符合要求
    if len(features) != input_size:
        messagebox.showerror("错误", f"输入特征数量错误，应为 {input_size} 个。")
        return

    # 归一化特征并进行预测
    mean, std = load_normalization_params()
    normalized_features = normalize(features, mean, std)
    features_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)
    probabilities, predictions = predict(model, features_tensor)

    # 显示预测结果
    result_text = "\n预测结果:\n"
    for i, label in enumerate(class_labels):
        result_text += f"{label}: {'是' if predictions[0, i] == 1 else '否'} (概率: {probabilities[0, i].item():.2f})\n"

    # 更新预测结果显示
    for i, prediction in enumerate(predictions[0]):
        if prediction == 1:
            color = "green"
        elif prediction == 2:
            color = "yellow"
        else:
            color = "red"
        prediction_boxes[i].config(bg=color)  # 更新颜色方块背景色


# 创建预测按钮
predict_button = tk.Button(root, text="预测", font=("Arial", 12), command=on_predict, bg="#4CAF50", fg="white", padx=20,
                           pady=10)
predict_button.grid(row=7, column=0, pady=20, sticky="ew")

# 启动 tkinter 主循环
root.mainloop()
