import torch
import torch.nn.functional as F

# 二分类问题
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]]) 

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()
criterion = torch.nn.BCELoss(reduction = 'mean') # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # 样本量少，可尝试0.1

for epoch in range(5000): # 样本量少，可尝试5000
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
print('y_test = ', y_test) # 输出如 tensor([[0.6552]]) 表示模型预测该样本属于正类（y = 1）的概率为 65.52%