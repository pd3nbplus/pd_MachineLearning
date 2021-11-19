# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:53:16 2021

@author: 潘登
"""

#%%全变量梯度下降
import numpy as np


x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)
x_b = np.concatenate((np.ones((100,1)),x),axis = 1)


#创建超参数
learning_rate = 0.001
n_lterations = 100000
n_epoches = 100  #学习100轮
#初始化w0 w1 ... 标准正太分布创建w
theta = np.random.randn(2,1)

for _ in range(n_lterations):
    #求梯度，计算gradient
    gradient = x_b.T.dot(x_b.dot(theta)-y)   #gradient中每一行代表一个梯度
    #应用梯度下降法公式去调整theta（直接采用矩阵运算） 
    theta = theta -learning_rate * gradient
print(theta)


#%%随机梯度下降
import numpy as np


x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)
x_b = np.concatenate((np.ones((100,1)),x),axis = 1)

#创建超参数
learning_rate = 0.001
m = 100   #样本个数
n_epoches = 10000  #学习10000轮

#初始化w0 w1 ... 标准正太分布创建w
theta = np.random.randn(2,1)

for _ in range(n_epoches):
    for _ in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradient = x_b.T.dot(x_b.dot(theta)-y)
        theta = theta -learning_rate * gradient
        
print(theta)


#%%小批量梯度下降
import numpy as np


x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)
x_b = np.concatenate((np.ones((100,1)),x),axis = 1)

#创建超参数
learning_rate = 0.001
n_epoches = 10000  #学习10000轮
batch_size = 10
num_batches = int(m/batch_size)

#初始化w0 w1 ... 标准正太分布创建w
theta = np.random.randn(2,1)

for _ in  range(n_epoches):
    for _ in range(num_batches):
        random_index = np.random.randint(m)
        x_batch = x_b[random_index:random_index+batch_size]  #这里超出的索引不会报错，会切到最后一个为止
        y_batch = y[random_index:random_index+batch_size]
        gradient = x_b.T.dot(x_b.dot(theta)-y)
        theta = theta -learning_rate * gradient

print(theta)



#%% 随机和小批量的改进
#改进方案1
#先打乱顺序
#按照顺序取
#

import numpy as np


x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)
x_b = np.concatenate((np.ones((100,1)),x),axis = 1)

#创建超参数
learning_rate = 0.001
m = 100   #样本个数
n_epoches = 10000  #学习10000轮

#初始化w0 w1 ... 标准正太分布创建w
theta = np.random.randn(2,1)

for _ in range(n_epoches):
    #双层for循环之间，每个轮次开始分批次迭代之前打乱数据索引顺序
    arr = np.arange(len(x_b))
    np.random.shuffle(arr)  #打乱顺序
    x_b = x_b[arr]  
    y = y[arr]     #同时保证y也要打乱顺序
    for i in range(m):
        xi = x_b[i:i+1]
        yi = y[i:i+1]
        gradient = x_b.T.dot(x_b.dot(theta)-y)
        theta = theta -learning_rate * gradient
        
print(theta)

print('*'*30)
#小批量改进


#创建超参数
learning_rate = 0.001
n_epoches = 10000  #学习10000轮
batch_size = 10
num_batches = int(m/batch_size)

#初始化w0 w1 ... 标准正太分布创建w
theta = np.random.randn(2,1)

for _ in  range(n_epoches):
    #双层for循环之间，每个轮次开始分批次迭代之前打乱数据索引顺序
    arr = np.arange(len(x_b))
    np.random.shuffle(arr)  #打乱顺序
    x_b = x_b[arr]  
    y = y[arr]     #同时保证y也要打乱顺序
    for i in range(num_batches):
        x_batch = x_b[i*batch_size:i*batch_size+batch_size]  #这里超出的索引不会报错，会切到最后一个为止
        y_batch = y[i*batch_size:i*batch_size+batch_size]
        gradient = x_b.T.dot(x_b.dot(theta)-y)
        theta = theta -learning_rate * gradient

print(theta)
#可以看出最终两者的结果是一样的
#%%学习率的调整
#全变量
import numpy as np


x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)
x_b = np.concatenate((np.ones((100,1)),x),axis = 1)


#创建超参数
# learning_rate = 0.001
#定义一个学习率调整的函数
t0,t1 = 5,500
def learning_rate_adj(t):
    return t0/(t+t1)

n_lterations = 100000
n_epoches = 100  #学习100轮
#初始化w0 w1 ... 标准正太分布创建w
theta = np.random.randn(2,1)

for i in range(n_lterations):
    learning_rate = learning_rate_adj(i)
    #求梯度，计算gradient
    gradient = x_b.T.dot(x_b.dot(theta)-y)   #gradient中每一行代表一个梯度
    #应用梯度下降法公式去调整theta（直接采用矩阵运算） 
    theta = theta -learning_rate * gradient
print(theta)

#%%
#小批量改进
import numpy as np


x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)
x_b = np.concatenate((np.ones((100,1)),x),axis = 1)

#创建超参数
t0,t1 = 5,500
def learning_rate_adj(t):
    return t0/(t+t1)

n_epoches = 10000  #学习10000轮
batch_size = 10
num_batches = int(m/batch_size)

#初始化w0 w1 ... 标准正太分布创建w
theta = np.random.randn(2,1)

for epoch in  range(n_epoches):
    #双层for循环之间，每个轮次开始分批次迭代之前打乱数据索引顺序
    arr = np.arange(len(x_b))
    np.random.shuffle(arr)  #打乱顺序
    x_b = x_b[arr]  
    y = y[arr]     #同时保证y也要打乱顺序
    for i in range(num_batches):
        learning_rate = learning_rate_adj(epoch*num_batches+i)
        x_batch = x_b[i*batch_size:i*batch_size+batch_size]  #这里超出的索引不会报错，会切到最后一个为止
        y_batch = y[i*batch_size:i*batch_size+batch_size]
        gradient = x_b.T.dot(x_b.dot(theta)-y)
        theta = theta -learning_rate * gradient

print(theta)






















