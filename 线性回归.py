# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:25:13 2021

@author: 潘登
"""

#%%线性回归
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(2)
x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)

sns.set()
sns.regplot(x, y)
plt.show()
#为了求解w0的截距项，我们给x矩阵左边加上一列全为1的x0
x_b = np.concatenate((np.ones((100,1)),x),axis = 1)

w = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

print(w) #依次为w0 w1...

#绘图查看
x_plot = np.array([[0,2]]).T
x_plot_b = np.concatenate((np.ones((2,1)),x_plot),axis = 1)
y_plot = x_plot_b.dot(w)

plt.plot(x_plot,y_plot,'r-')
plt.plot(x,y,'b.')
plt.show()
#%% 多元线性回归
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(2)
x1 = 2*np.random.rand(100,1)
np.random.seed(3)
x2 = 3*np.random.rand(100,1)

#观察值                误差（服从正态分布）
y = 5 + 4*x1 + 3*x2 + np.random.randn(100,1)

#为了求解w0的截距项，我们给x矩阵左边加上一列全为1的x0
x_b = np.concatenate((np.ones((100,1)),x1,x2),axis = 1)

w = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

print(w) #依次为w0 w1...

#绘图查看

x_plot = np.array([[0,2],
                   [0,3]]).T
x_plot_b = np.concatenate((np.ones((2,1)),x_plot),axis = 1)
y_plot = x_plot_b.dot(w)



sns.set_style('ticks')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1,x2,y,color = 'r')
ax.plot(x_plot[:,0].tolist(),x_plot[:,1].tolist(),y_plot[:,0].tolist(),zdir='z',c = 'b')
#这个傻逼画图只能传列表
plt.show()

#x1维度上
plt.plot(x_plot[:,0],y_plot,'r-')
plt.plot(x,y,'b.')
plt.show()

#x2维度上
plt.plot(x_plot[:,1],y_plot,'r-')
plt.plot(x,y,'b.')
plt.show()




















