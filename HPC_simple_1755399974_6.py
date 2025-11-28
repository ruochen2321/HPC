import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 设置设备（GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== 数据加载与预处理 ======================
# 加载训练集和测试集
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 合并数据集（去除ID列），用于统一预处理
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# ====================== 数据预处理 ======================
# 1. 数值特征标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)

# 2. 处理缺失值（标准化后填充0）
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 3. 处理分类特征（独热编码，包括缺失值作为独立类别）
all_features = pd.get_dummies(all_features, dummy_na=True)

# 4. 转换布尔列为整型（0/1）
bool_columns = all_features.select_dtypes(include=[bool]).columns
for col in bool_columns:
    all_features[col] = all_features[col].astype(int)

# ====================== 数据集划分 ======================
n_train = train_data.shape[0]
# 转换为PyTorch张量
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# ====================== 创建数据加载器 ======================
batch_size = 32
dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True)


# ====================== 模型定义 ======================
class AttentionNet(torch.nn.Module):
    """带注意力机制的房价预测模型"""

    def __init__(self, in_features, hidden_size, num_heads=4):
        super().__init__()

        # 输入嵌入层
        self.embedding = torch.nn.Linear(in_features, hidden_size)
        self.norm1 = torch.nn.LayerNorm(hidden_size)

        # 多头注意力机制
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = torch.nn.Dropout(0.1)

        # 前馈神经网络
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = torch.nn.LayerNorm(hidden_size)

        # 输出层
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # 嵌入层处理
        x = self.embedding(x)
        residual = x
        x = self.norm1(x)

        # 多头注意力计算
        batch_size = x.size(0)
        seq_len = 1  # 每个样本是单一向量

        # 重塑张量形状用于多头注意力计算
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attention_dropout(attn)

        # 应用注意力权重
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = context.squeeze(1)  # 移除序列维度

        # 残差连接
        x = residual + x
        residual = x
        x = self.norm2(x)

        # 前馈网络
        x = self.ffn(x)
        x = residual + x

        # 输出预测
        return self.output(x)


# ====================== 模型初始化 ======================
in_features = train_features.shape[1]
hidden_size = 200
model = AttentionNet(in_features, hidden_size).to(device)
print(f"模型结构:\n{model}")
print(f"输入特征维度: {in_features}")

# ====================== 训练配置 ======================
criterion = torch.nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # Adam优化器


# ====================== 训练函数 ======================
def train_model(loader, epochs=200):
    """模型训练函数"""
    train_losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)

            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 记录每个epoch的平均损失
        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()


# ====================== 模型训练 ======================
train_model(train_loader)


# ====================== 预测与结果保存 ======================
def generate_submission(test_features):
    """生成提交文件"""
    model.eval()
    with torch.no_grad():
        test_features = test_features.to(device)
        preds = model(test_features).cpu().numpy()

    # 创建提交格式
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': preds.squeeze()
    })

    # 保存结果
    submission.to_csv('submission.csv', index=False)
    return submission


# 生成并保存预测结果
submission = generate_submission(test_features)
print("预测结果已保存至 submission.csv")
print('程序结束')