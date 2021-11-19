# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:47:03 2021

@author: 潘登
"""

#%%sigmoid 函数
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

x = np.arange(-10,10,0.01)
y = sigmoid(x)

plt.plot(x,y)
plt.show()

#%%逻辑回归loss图像
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
data = load_breast_cancer()
x,y = scale(data['data'][:,:2]),data['target']   #这里为了方便画图，所以只取x的两列
#求出两个维度对应的数据在逻辑回归算法下的最优解
lr = LogisticRegression(fit_intercept=False)  #为了方便画图，不加截距项
lr.fit(x,y)
#把参数取出来
w = lr.coef_

#已知w的情况下，传进来数据x，返回数据的y_predict
def p_theta_function(feature,w):
    Z  = feature.dot(w.T)     #Z = xθ
    return 1/(1+np.exp(-1*Z))


def loss_fuction(samples_features,samples_labels,w):
    result = 0
    #遍历数据集中的每一条数据，并且计算每条样本的损失，加到result身上得到整体数据集损失
    for feature,label in zip(samples_features,samples_labels):
        #这是计算一条样本的y_predict
        p_result = p_theta_function(feature,w)
        loss_result = -1*label*np.log(p_result) - (1-label)*np.log(1-p_result)
        result += loss_result
    return result
   
w1_space = np.arange(w[:,0]-0.6,w[:,0]+0.6,1.2/49)
w2_space = np.arange(w[:,1]-0.6,w[:,1]+0.6,1.2/49)
w1,w2 = np.meshgrid(w1_space, w2_space)
w_list = []
for i in range(50):
    temp = []
    for j in range(50):
        temp.append(loss_fuction(x,y,np.array([w1[i][j],w2[i][j]])))
    w_list.append(temp)
    
result = np.array(w_list)

fig=plt.figure()
ax1 = plt.axes(projection='3d')
ax1.contour(w1,w2,result,30)
ax1.view_init(elev=90., azim=140)
plt.show()

fig=plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot_surface(w1,w2,result,rstride = 1, cstride = 1,cmap='rainbow')
ax2.view_init(elev=20., azim=140)
plt.show()

