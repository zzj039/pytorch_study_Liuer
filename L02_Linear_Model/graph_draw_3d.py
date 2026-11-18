import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1, 2, 3]
y_data = [2, 4, 6]

def f(x, w, b):
    return x * w +b

def loss(x, y, w, b):
    y_pred = f(x, w, b)
    return (y_pred - y) * (y_pred - y)

w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.0, 0.1):
    for b in np.arange(-1.0, 1.0, 0.1):
        print('w = ', w)
        print('b = ', b)
        loss_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = f(x_val, w, b)
            loss_val = loss(x_val, y_val, w, b)
            loss_sum += loss_val
            mse = loss_sum / len(x_data)
            print('\t', x_val, y_val, y_pred_val, loss_val)
            print('MSE = ', loss_sum / len(x_data))
        b_list.append(b)
        mse_list.append(mse)
        w_list.append(w)
    
# 定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection = '3d')

# 绘图
scatter = ax1.scatter(w_list, b_list, mse_list, c = mse_list, cmap = 'viridis') # 按照数值映射颜色
fig.colorbar(scatter, ax=ax1, label='MSE') # 颜色条
ax1.set_xlabel('Weight (w)')
ax1.set_ylabel('Bias (b)')
ax1.set_zlabel('MSE')
plt.show()