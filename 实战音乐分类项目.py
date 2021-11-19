# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:03:06 2021

@author: 潘登
"""

#%%实战音乐分类项目
from scipy import fft #fft是傅立叶变换
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt

(sample_rate,x) = wavfile.read(r'C:\Users\潘登\Documents\python全系列\人工智能\genres\blues\converted\blues.00000.au.wav')
print(sample_rate,x.shape)   #sample_rate每秒采样多少样本   x是(661794,)单元组表示这首歌是单通道 用x/sample_rate可以得到这首歌的时长（s)

# plt.figure(figsize = (10,4),dpi = 80)
# plt.xlabel('time')
# plt.ylabel('frequency')
# plt.grid(True,linestyle='-',color = '0.25')
# specgram(x,Fs = sample_rate,xextent = (0,30))
# plt.show()

def plotSpec(g,n):
    file_name = r'C:\Users\潘登\Documents\python全系列\人工智能\genres' + '\\' + g + '\converted' + '\\' + g + '.' + n + '.au.wav'
    sample_rate,x = wavfile.read(file_name)
    specgram(x,Fs = sample_rate,xextent = (0,30))
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.grid(True,linestyle='-',color = '0.25')
    plt.title(g+'_'+n[-1])
    
plt.figure(figsize = (18,9),dpi = 80, facecolor = 'w', edgecolor = 'k')
plt.subplot(6,3,1);plotSpec('classical','00001')
plt.subplot(6,3,2);plotSpec('classical','00002')
plt.subplot(6,3,3);plotSpec('classical','00003')
plt.subplot(6,3,4);plotSpec('jazz','00001')
plt.subplot(6,3,5);plotSpec('jazz','00002')
plt.subplot(6,3,6);plotSpec('jazz','00003')
plt.subplot(6,3,7);plotSpec('pop','00001')
plt.subplot(6,3,8);plotSpec('pop','00002')
plt.subplot(6,3,9);plotSpec('pop','00003')
plt.subplot(6,3,10);plotSpec('rock','00001')
plt.subplot(6,3,11);plotSpec('rock','00002')
plt.subplot(6,3,12);plotSpec('rock','00003')
plt.subplot(6,3,13);plotSpec('country','00001')
plt.subplot(6,3,14);plotSpec('country','00002')
plt.subplot(6,3,15);plotSpec('country','00003')
plt.subplot(6,3,16);plotSpec('metal','00001')
plt.subplot(6,3,17);plotSpec('metal','00002')
plt.subplot(6,3,18);plotSpec('metal','00003')

plt.tight_layout(pad = 0.4,w_pad = 0,h_pad = 1)
plt.show()

#%%用自己创建的音乐文件进行傅里叶变换
plt.figure(num = None,figsize = (10,8),dpi = 80,facecolor = 'w',edgecolor = 'k')
plt.subplot(2,2,1)
sample_rate,x = wavfile.read(r'C:\Users\潘登\Documents\python全系列\人工智能\sine_a.wav')
plt.xlabel('time')
plt.ylabel('frequency')
plt.title('400 HZ sine wave')
specgram(x, Fs = sample_rate, xextent = (0,30))

plt.subplot(2,2,2)
plt.xlabel('frequency')
plt.xlim(0, 1000)
plt.ylabel('amplitude')
plt.title('FFT of 400 HZ sine wave')
plt.plot(fft(x,sample_rate))
plt.show()

plt.subplot(2,2,3)
sample_rate,x = wavfile.read(r'C:\Users\潘登\Documents\python全系列\人工智能\sine_b.wav')
plt.xlabel('time')
plt.ylabel('frequency')
plt.title('3000 HZ sine wave')
specgram(x, Fs = sample_rate, xextent = (0,30))

plt.subplot(2,2,4)
plt.xlabel('frequency')
plt.xlim(1000, 4000)
plt.ylabel('amplitude')
plt.title('FFT of 3000 HZ sine wave')
plt.plot(fft(x,sample_rate))
plt.show()

#%%图示傅里叶变换
def plotFFT(g,n):
    file_name = r'C:\Users\潘登\Documents\python全系列\人工智能\genres' + '\\' + g + '\converted' + '\\' + g + '.' + n + '.au.wav'
    sample_rate,x = wavfile.read(file_name)
    plt.plot(fft(x,sample_rate))
    plt.xlabel('frequency')
    plt.xlim(0,3000)   #太高频率的 我们听不见
    plt.ylabel('amplitude')
    plt.title(g+'_'+n[-1])

plt.figure(num = None,figsize = (10,8),dpi = 80,facecolor = 'w',edgecolor = 'k')
plt.subplot(6,2,1);plotSpec('classical','00001')
plt.subplot(6,2,2);plotFFT('classical','00001')
plt.subplot(6,2,3);plotSpec('jazz','00001')
plt.subplot(6,2,4);plotFFT('jazz','00001')
plt.subplot(6,2,5);plotSpec('country','00001')
plt.subplot(6,2,6);plotFFT('country','00001')
plt.subplot(6,2,7);plotSpec('pop','00001')
plt.subplot(6,2,8);plotFFT('pop','00001')
plt.subplot(6,2,9);plotSpec('rock','00001')
plt.subplot(6,2,10);plotFFT('rock','00001')
plt.subplot(6,2,11);plotSpec('metal','00001')
plt.subplot(6,2,12);plotFFT('metal','00001')
plt.show()

#%%准备傅里叶变换，把音乐文件一个个的去使用傅里叶变换，并把结果保存
#提取特征
import numpy as np
def creat_fft(g,n):
    file_name = r'C:\Users\潘登\Documents\python全系列\人工智能\genres' + '\\' + g + '\converted' + '\\' + g + '.' + str(n).zfill(5) + '.au.wav'
    sample_rate,x = wavfile.read(file_name)
    fft_features = abs(fft(x)[:1000])  #不需要特别多的数据，再往后都是噪音
    sad = r'C:\Users\潘登\Documents\python全系列\人工智能\trainset' + '\\' + g + '.' + str(n).zfill(5) + '.fft'
    np.save(sad,fft_features)

genre_list = ['classical','jazz','country','pop','rock','metal']
for g in genre_list:
    for n in range(100):
        creat_fft(g,n)

#%%读取傅里叶变换之后的数据集，将其转换成机器学习所需要的x和y
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
genre_list = ['classical','jazz','country','pop','rock','metal']
x = []
y = []
for g in genre_list:
    for n in range(100):
            file_name = r'C:\Users\潘登\Documents\python全系列\人工智能\trainset' + '\\' + g +  '.' + str(n).zfill(5) + '.fft.npy'
            fft_features = np.load(file_name)
            x.append(fft_features)
            y.append(genre_list.index(g))
            
x = np.array(x)
y = np.array(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state=5)

#训练模型并且保存模型
model = LogisticRegression(multi_class='multinomial',solver = 'sag',max_iter=1000)#这里multinomial 最好与 sag一起使用，用默认的求解方法会报错
model.fit(x_train,y_train)

output = open('model.pkl','wb')  #保存模型到model.pkl  wb是以二进制写入
pickle.dump(model,output)
output.close()

#%%把刚才训练的模型读入，进行预测
import pickle
from pprint import pprint 
from sklearn.metrics import confusion_matrix

pkl_file = open('model.pkl','rb')
model_loaded = pickle.load(pkl_file)
pprint(model_loaded)
pkl_file.close()

temp = model_loaded.predict(x_test)
print(confusion_matrix(y_test,temp,labels = range(len(genre_list))))
print(np.trace(confusion_matrix(y_test,temp,labels = range(len(genre_list))))/180) 

#%%从网上下载音乐来查看model
print('Starting read wavfile...')
music_name = '无条件-陈奕迅(30秒).wav'
# music_name = '月光曲(30秒).wav'
sample_rate,x = wavfile.read(music_name)

print(x.shape)
x = np.reshape(x,(1,-1))[0]  #把双通道的音频转化成单通道

test_fft_features = abs(fft(x)[:1000])

temp = model_loaded.predict([test_fft_features])
print(genre_list[int(temp)])

