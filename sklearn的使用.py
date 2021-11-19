# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:38:34 2021

@author: 潘登
"""

#%% sklearn的使用
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
x1 = 2*np.random.rand(100,1)
np.random.seed(3)
x2 = 3*np.random.rand(100,1)
x_b = np.concatenate((x1,x2),axis = 1)

y = 5 + 4*x1 + 3*x2 + np.random.randn(100,1)

reg = linear_model.LinearRegression(fit_intercept=True)  #这个True就自动帮助创建截距项
reg.fit(x_b,y)
print(reg.intercept_,reg.coef_)


#预测
x_predict = np.array([[0,0],
                      [2,1],
                      [2,4]])
y_predict = reg.predict(x_predict)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1,x2,y,color = 'b')
ax.plot(x_predict[:,0].tolist(),x_predict[:,1].tolist(),y_predict[:,0].tolist(),'r-')
plt.show()