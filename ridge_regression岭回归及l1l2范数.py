# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:58:47 2021

@author: 潘登
"""

#%%ridge_regression岭回归(l2)
import numpy as np
from sklearn.linear_model import Ridge

np.random.seed(2)
x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)

reg = Ridge(alpha = 0.4,solver = 'sag')  #sag表示随机梯度下降
reg.fit(x,y)   #这里的x不用传截距项
print(reg.predict([[2]]))  #这里预测值一定要是矩阵形式（二维数组）
print(reg.intercept_)
print(reg.coef_)

#%%lasso regression(L1正则项)
import numpy as np
from sklearn.linear_model import Lasso

np.random.seed(2)
x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)

reg = Lasso(alpha = 0.4,max_iter=1000000)  #这里无需传入梯度下降的方法 max_iter表示迭代次数
reg.fit(x,y)   #这里的x不用传截距项
print(reg.predict([[2]]))  #这里预测值一定要是矩阵形式（二维数组）
print(reg.intercept_)
print(reg.coef_)

#%%ElasticNet （既使用L1，也使用L2）
import numpy as np
from sklearn.linear_model import ElasticNet

np.random.seed(2)
x = 2*np.random.rand(100,1)
#观察值         误差（服从正态分布）
y = 5 + 4*x + np.random.randn(100,1)

reg = ElasticNet(alpha = 0.4,l1_ratio = 0.5,max_iter=1000000)  #这里无需传入梯度下降的方法 max_iter表示迭代次数
reg.fit(x,y)   #这里的x不用传截距项
print(reg.predict([[2]]))  #这里预测值一定要是矩阵形式（二维数组）
print(reg.intercept_)
print(reg.coef_)