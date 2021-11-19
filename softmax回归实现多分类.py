# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 08:43:32 2021

@author: 潘登
"""

#%%softmax回归实现多分类
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
path = 'iris.csv'
data = pd.read_csv(path)
np.random.seed(2)
x = data.iloc[:,2:4]
y = data.iloc[:,4]
y.replace(['setosa','versicolor','virginica'],[0,1,2],inplace = True)#把品种转化为0 1 2

sns.set()
sns.scatterplot(data['petal_length'],data['petal_width'],hue = data['species'])
#划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

log_reg = LogisticRegression(solver='sag',max_iter=100000,multi_class='multinomial')  #multinomial表示softmax 回归
#公式中默认自带penalty L2正则项   和  L2 前面的系数1/C  C 默认是1
log_reg.fit(x_train,y_train)
log_reg.coef_

y_hat = log_reg.predict_proba(x_test)
#这个第一列表示是零这一类的的概率
#将概率转化为预测结果
idx = np.argmax(y_hat, axis=1)

#比较预测结果与实际值
print(idx == y_test) 
#发现结果中有一个错误
