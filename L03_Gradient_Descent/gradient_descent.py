import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]
w = 1
n = 0.01
epoch_list = []
cost_list = []
def f(x, w):
    return x * w

def cost(xs, ys, w):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = f(x, w)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

def gradient(xs, ys, w):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * (x * w - y) * x
    return grad / len(xs)

for epoch in range(200):
    cost_val = cost(x_data, y_data, w)
    grad_val = gradient(x_data, y_data, w)
    w = w - n * grad_val
    epoch_list.append(epoch)
    cost_list.append(cost_val)
    print('epoch: ', epoch, 'w = ', w, 'cost: ', cost_val)

m = 4
print('predict:\n 输入: ', m,' 输出：', f(m, w))

plt.plot(epoch_list, cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()