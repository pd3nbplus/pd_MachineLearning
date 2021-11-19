# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:02:55 2021

@author: 潘登
"""

#%%逻辑回归的梯度下降
import numpy as np
import matplotlib.pyplot as plt


y = np.random.choice([0,1],100).reshape(-1,1)
x = 2*np.random.rand(100,1)* y + np.random.randn(100,1)
x_b = np.concatenate((np.ones((100,1)),x),axis = 1)


#创建超参数
learning_rate = 0.001
n_lterations = 100000
n_epoches = 100  #学习100轮
#初始化w0 w1 ... 标准正太分布创建w
theta = np.random.randn(2,1)


def logistic(theta,x):   #theta是n*1的列向量，x是m*n的矩阵
     return 1/(1+np.exp(-1*x.dot(theta)))

for _ in range(n_lterations):
    #求梯度，计算gradient
    gradient = x_b.T.dot(logistic(theta,x_b)-y)   #gradient中每一行代表一个梯度
    #应用梯度下降法公式去调整theta（直接采用矩阵运算） 
    theta = theta -learning_rate * gradient
print(theta)

x_plot = np.array([[-5,5]]).T
x_plot_b = np.concatenate((np.ones((2,1)),x_plot),axis = 1)
y_plot = 1/(1+np.exp(-1*x_plot_b.dot(theta)))  #绘制分界线



plt.plot(x_plot,y_plot,'r-')
plt.plot(x,y,'b.')
plt.show()

