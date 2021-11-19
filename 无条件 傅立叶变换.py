# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:32:18 2021

@author: 潘登
"""

#%% 无条件 傅立叶变换
from scipy import fft #fft是傅立叶变换
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt

path = r'C:\Users\潘登\Documents\python全系列\人工智能\无条件-陈奕迅(30秒).wav'
(sample_rate,x) = wavfile.read(path)
print(sample_rate,x.shape)   #sample_rate每秒采样多少样本   x是(1395072, 2)双元组表示这首歌是双通道 用x/sample_rate可以得到这首歌的时长（s)



def plotSpec(file_name):
    plt.subplot(1,2,1)
    sample_rate,x = wavfile.read(file_name)
    x = x[:,0] # 将双通道变成单通道
    specgram(x,Fs = sample_rate,xextent = (0,30))
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.grid(True,linestyle='-',color = '0.25')
    plt.title('时域--无条件-陈奕迅(30秒)')
    plt.subplot(1,2,2)
    plt.xlabel('frequency')
    plt.xlim(0, 4000)
    plt.ylabel('amplitude')
    plt.title('FFT of 无条件-陈奕迅(30秒)')
    plt.plot(fft(x,sample_rate))
    plt.show()

    
plt.figure(figsize = (18,9),dpi = 80, facecolor = 'w', edgecolor = 'k')
plotSpec(path)