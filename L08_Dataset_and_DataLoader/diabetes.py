import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter = ',', dtype = np.float32) # 把 CSV 整个读成一个 NumPy 二维数组
        self.len = xy.shape[0] # shape[0] 就是行数 = 样本个数，存起来给 __len__ 用
        self.x_data = torch.from_numpy(xy[:, :-1]) # NumPy 数组 → PyTorch 张量
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index): # 按索引取一个样本
        return self.x_data[index], self.y_data[index]

    def __len__(self): # 返回数据集大小，DataLoader才能计算batch-size等
        return self.len

dataset = DiabetesDataset('pima-indians-diabetes.csv')
train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = True) # num_workers = 2

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
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # 使用Adam优化器
epoch_list = []
loss_list = []

for epoch in range(200):
    total_loss = 0.0
    batch_count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        total_loss += loss.item()      # 累加本 epoch 的 loss
        batch_count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / batch_count   # 一个 epoch 一个平均 loss
    epoch_list.append(epoch)
    loss_list.append(avg_loss)
    print(f'Epoch [{epoch+1}/100], Avg Loss: {avg_loss:.4f}')  # 每轮只打印一行

plt.plot(epoch_list, loss_list, color='tab:blue')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)
plt.title('Training Loss')
plt.show()