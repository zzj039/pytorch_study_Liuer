import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]
w = 1
n = 0.01
epoch_list = []
loss_list = []

def f(x, w):
    return x * w

def loss(x, y, w):
    return (f(x, w) - y) ** 2


def gradient(x, y, w):
    return 2 * x * (w * x - y)

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x,y, w)
        grad_val = gradient(x, y, w)
        w = w - n * grad_val
        epoch_list.append(epoch)
        loss_list.append(loss_val)
        print('epoch: ', epoch, 'w = ', w, 'loss: ', loss_val)

m = 4
print('predict:\n 输入: ', m,' 输出：', f(m, w))
