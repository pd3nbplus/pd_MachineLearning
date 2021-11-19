# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:36:20 2021

@author: 潘登
"""

#%%多项式回归升维

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


np.random.seed(42)
m = 100
X = 6*np.random.rand(m, 1) - 3
y = 0.5*X**2 + X + 2 + np.random.randn(m, 1)
plt.plot(X, y, 'b.')

X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = X[80:]

# 进行多项式升维, 升维为 2, 10
d = {1: 'g-', 2: 'r+', 10: 'y*'}
for i in d:
    print('现在维度为:', i)
    poly_features = PolynomialFeatures(degree=i, include_bias=True)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.fit_transform(X_test)
    print(X_train[0])
    print(X_poly_train[0])
    print(X_train.shape)
    print(X_poly_train.shape)
    #把测试集和训练集扔进去后就会自动在数据的左边拼接上升维后的数据
    #注意 多项式回归只是一个预处理工具，不是model

    lin_reg = LinearRegression(fit_intercept=False)
    lin_reg.fit(X_poly_train, y_train)
    print(lin_reg.intercept_, lin_reg.coef_)

    # 看看是否随着degree的增加升维，是否过拟合了
    y_train_predict = lin_reg.predict(X_poly_train)
    y_test_predict = lin_reg.predict(X_poly_test)

    plt.plot(X_poly_train[:, 1], y_train_predict, d[i])
    
    print('训练集的MSE')
    print(mean_squared_error(y_train, y_train_predict))  #训练集的MSE
    print('测试集的MSE')
    print(mean_squared_error(y_test, y_test_predict))   #测试集的MSE

plt.show()




