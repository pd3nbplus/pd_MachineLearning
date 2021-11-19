# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:34:39 2021

@author: 潘登
"""
#%%波形显示
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from pydub import AudioSegment
 
# 1秒=1000毫秒
SECOND = 1000
# 音乐文件
AUDIO_PATH = '无条件-陈奕迅(30秒).wav'
 
def split_music(begin, end, filepath):
 # 导入音乐
    song = AudioSegment.from_wav(filepath)
 
 # 取begin秒到end秒间的片段
    song = song[begin*SECOND: end*SECOND]
 
 # 存储为临时文件做备份
    a = filepath.split('.')
    temp_path = a[0] + '(波形显示).' + a[1]
    song.export(temp_path, format='wav')
 
    return temp_path
 
music, sr = librosa.load(split_music(0, 1, AUDIO_PATH))
 
# 宽高比为14:5的图
plt.figure(figsize=(14, 5))
librosa.display.waveplot(music, sr=sr)
plt.show() 

# 放大
n0 = 9000
n1 = 10000
 
music = np.array([mic for mic in music])
plt.figure(figsize=(14, 5))
plt.plot(music[n0:n1])
plt.grid()
 
# 显示图
plt.show()

# 只显示正半段的
music = np.array([mic for mic in music if mic > 0])
plt.figure(figsize=(14, 5))
plt.plot(music[n0:n1])
plt.grid()
 
# 显示图
plt.show() 