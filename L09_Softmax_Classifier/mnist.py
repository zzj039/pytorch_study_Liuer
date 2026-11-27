import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
'''
原始 MNIST 图片是 PIL 图像（像素值 0~255 的整数，H×W 排列）。
ToTensor() 把它转成float类型，同时把范围压缩至[0, 1]，以及把维度由28×28转换为1×28×28
Normalize((0.1307,), (0.3081,)) 做标准化；这两个数字是整个 MNIST 训练集算出来的均值和标准差，
核心是让输入分布"以 0 为中心、尺度为 1"：各维度量级一致，损失函数的等高线更接近圆形，梯度下降不容易走锯齿、收敛更快；
同时配合 ReLU 这类激活，正负均衡，不容易一上来就全落在同一侧导致梯度消失
另外注意 (0.1307,) 是元组，逗号不能省，因为有 C 个通道就要给 C 个均值（RGB 图就写三个），这里灰度图只有 1 个通道
'''
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

train_dataset = datasets.MNIST(root = 'dataset/mnist', train = True, download = True, transform = transform)
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)

test_dataset = datasets.MNIST(root = 'dataset/mnist', train = False, download = True, transform = transform)
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 把二维图片转换成一维向量
        # Linear(784, 512) 本质是一个矩阵乘法 Wx+b，
        # W 的形状是 512×784，要求输入是长度为 784 的向量。而 DataLoader 出来的一个 batch 形状是[64, 1, 28, 28]
        # 所以要把后三维压成一维
        x = x.reshape(-1, 784) # 或者更规范：x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)

model = Net()

# 衡量模型预测的概率分布和真实答案差多远。
# 真实答案是 one-hot（比如第 3 类是 1，其余全 0），所以公式简化成只看"正确类别那个位置预测了多少概率"
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
# 原始SGD只看当前这一步的梯度，加入动量后，会带有历史速度

def train(epoch):
    running_loss = 0.0

    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()# forward + backward + update

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            # outputs 形状是 [64, 10]：64 张图，每张 10 个类别分数。torch.max 沿 dim=1（在 10 个类别里比）找最大值
            # 只关心"预测的是哪个数字"，最大值本身没用，所以用 _ 这个惯用变量名把它丢掉
            total += labels.size(0)
            # labels 形状是 [64]，size(0) 就是 batch 里的样本数，也就是累计看了多少张
            correct += (predicted == labels).sum().item() # 注意有item(),否则是张量，还要有计算图
    print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
