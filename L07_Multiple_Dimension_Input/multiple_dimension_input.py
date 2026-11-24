import torch
import numpy as np
import matplotlib.pyplot as plt

# 多维输入的二分类问题
# diabetes prediction
xy = np.loadtxt('pima-indians-diabetes.csv', delimiter = ',', dtype = np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction = 'mean') # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epoch_list = []
loss_list = []

for epoch in range(20000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    # print('epoch = ', epoch, 'loss = ', loss.item())
    if (epoch + 1) % 1000 == 0:  # 每1000轮打印一次
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()