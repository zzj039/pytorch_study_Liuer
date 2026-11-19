import torch

x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(1.0, requires_grad = True) #仍然是张量，但是0维，（视为标量）
optimizer = torch.optim.SGD([w], lr = 0.01) # 优化器

def f(x, w):
    return x * w

def loss(x, y, w):
    y_pred = f(x, w)
    return (y_pred - y) ** 2

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        optimizer.zero_grad() #清零梯度
        l = loss(x, y, w)
        l.backward()
        # print('\tgrad:', x, y, w.grad.item())
        # with torch.no_grad():
        #     w -= 0.01 * w.grad
        optimizer.step()
    print('progress:', epoch, l.item())

print(f(4, w).item())