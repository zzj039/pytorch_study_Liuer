import numpy as np
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]

def f(x):
    return x*w

def loss(x, y):
    y_pred = f(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
mse_list = []

for w in np.arange(0.0, 4.0, 0.1):
    print('w = ', w)
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = f(x_val)
        loss_val = loss(x_val, y_val)
        loss_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE = ', loss_sum / len(x_data))
    w_list.append(w)
    mse_list.append(loss_sum / len(x_data))

plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('cost')
plt.show()