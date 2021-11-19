# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:40:37 2021

@author: 潘登
"""

#%% 图示傅里叶级数
import matplotlib.pyplot as plt
import numpy as np


x = np.array(np.linspace(0.1, 10, 100))
y = [1] * 30 + [-1] * 30 + [1] * 30 + [-1] * 10
y1 = 4/np.pi*np.sin(x)
y2 = 4/(3*np.pi)*np.sin(3*x)
y3 = y1 + y2
y4 = 4/(5*np.pi)*np.sin(5*x)
y5 = y3 + y4
y6 = 4/(7*np.pi)*np.sin(7*x)
y7 = y5 + y6
plt.rcParams['axes.facecolor']='black'
plt.figure(figsize=(10,6))
plt.plot(x, y1)
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'r')
plt.plot(x, y5, 'y')
plt.plot(x, y7, 'gray')
plt.show()

for i in range(3, 100, 2):
    y1 += 4/(i*np.pi)*np.sin(i*x)

plt.figure(figsize=(10,6))
plt.plot(x, y, 'gray')
plt.show()
#%% 时域频域两边看
plt.rcParams['axes.facecolor']='snow'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=335)


ax.plot(x, [1]*len(x), y1, 'red')
ax.plot(x, [3]*len(x), y2, 'green')
ax.plot(x, [5]*len(x), y4, 'cyan')
ax.plot(x, [0]*len(x), y5, 'black')
plt.xlabel('时域')
plt.ylabel('频域')
ax.set_zlabel('振幅')
plt.show()
#%% 时间差
plt.rcParams['axes.facecolor']='snow'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=45, azim=330)

x1 = [i for i in x if i <= np.pi/2]
x2 = [i for i in x if i <= np.pi/6]
x4 = [i for i in x if i <= np.pi/10]

ax.plot([0]*6, range(0, 6), 0, 'blue')
ax.plot(x1, [1]*len(x1), 0, 'red')
ax.plot(x2, [3]*len(x2), 0, 'green')
ax.plot(x4, [5]*len(x4), 0, 'cyan')
# ax.plot(x, [0]*len(x), y5, 'black')
ax.set_xlabel('时域')
ax.set_xlim([0, 2])
ax.set_ylabel('频域')
ax.set_ylim([0, 5.5])
ax.set_zlabel('振幅')
ax.set_zlim([0, 1])
plt.show()

#%% 相位谱
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=45, azim=330)

ax.plot([0]*6, range(0, 6), 0, 'blue')
ax.plot(x1, [1]*len(x1), 0, 'red')
ax.plot(x1, [3]*len(x2)*3, 0, 'green')
ax.plot(x1, [5]*len(x4)*5, 0, 'cyan')
# ax.plot(x, [0]*len(x), y5, 'black')
ax.set_xlabel('时域')
ax.set_xlim([0, 2])
ax.set_ylabel('频域')
ax.set_ylim([0, 5.5])
ax.set_zlabel('振幅')
ax.set_zlim([0, 1])
plt.show()








