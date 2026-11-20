import torch

#两层中括号构建出三行一列的二维张量
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]]) 

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # 输入、输出数据的特征维度均为一；torch.nn.Linear已经构造y = w*x + b

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(reduction = 'mean') # 新写法为 reduction = 'mean' or 'sum' or 'none'
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# model.parameters() 会返回模型中所有需要学习的参数。此处为w, b
# 优化器拿到这些参数后，就知道：在 optimizer.step() 时更新哪些变量；在 optimizer.zero_grad() 时清空哪些变量的梯度

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print('epoch = ', epoch, 'loss = ', loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_test = ', y_test)

# 取一个数值（标量）​用 .item()
# 取一个脱离计算图的张量（不追踪梯度）​用 .detach()
# .data 为旧版本，不适用