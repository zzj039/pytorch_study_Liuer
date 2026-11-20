import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. 明确指定 dtype 和设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 2. 构建数据（明确 float32 类型）
x_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32).to(device)
y_data = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32).to(device)

# 3. 使用 DataLoader（即使数据量小，也推荐用，方便扩展）
dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 4. 定义模型（推荐使用 Snake Case 命名类）
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()  # 简化 super 写法
        self.linear = torch.nn.Linear(1, 1)
        # 可选：显式初始化权重
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

# 5. 实例化模型并移到设备
model = LinearModel().to(device)

# 6. 损失函数（reduction='mean' 是默认值，可省略）
criterion = torch.nn.MSELoss()

# 7. 优化器（SGD 没问题，但 Adam 通常是更好的默认选择）
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 如果坚持用 SGD，可以加上 momentum：
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 8. 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()  # 设置训练模式（对 Dropout/BatchNorm 等层有影响）
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        # 前向传播
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 打印平均损失（每 100 轮打一次，避免刷屏）
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss / len(dataloader):.6f}')

# 9. 评估模式（关闭梯度计算）
model.eval()
with torch.no_grad():  # 完全禁用梯度计算，节省内存和计算
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    print(f'w = {w:.4f}')
    print(f'b = {b:.4f}')
    
    x_test = torch.tensor([[4.0]], dtype=torch.float32).to(device)
    y_test = model(x_test)
    print(f'y_pred for x=4.0: {y_test.item():.4f}')

# 10. 保存模型（推荐保存 state_dict 而不是整个模型）
# torch.save(model.state_dict(), 'linear_model.pth')