# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:37:10 2021

@author: 潘登
"""

#%%归一化
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
scaler = MinMaxScaler()
temp = np.arange(5)
print('最大值最小值归一化...')
print(scaler.fit_transform(temp.reshape(-1,1)))  #先摆成列再做归一化

scaler1 = StandardScaler()
print('均值归一化...')
print(scaler1.fit_transform(temp.reshape(-1,1)))
# 打印结果
print('均值为:', scaler1.mean_)
print('标准差为:',scaler1.var_)



