# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:26:58 2021

@author: 潘登
"""

#%%音乐剪辑
from pydub import AudioSegment

# 1秒=1000毫秒
SECOND = 1000
# 导入音乐
song = AudioSegment.from_wav("无条件-陈奕迅.wav")

# 取33秒到63秒间的片段
song = song[33*SECOND:63*SECOND]

# # 入场部分提高6分贝, 退场部分减少5分贝
# ten_seconds = 10 * SECOND
# last_five_seconds = -5 * SECOND
# beginning = song[:ten_seconds] + 6
# ending = song[last_five_seconds:] - 5

# # 形成新片段
# new_song = beginning + song[ten_seconds:last_five_seconds] + ending
new_song = song
# 导出音乐
new_song.export('无条件-陈奕迅(30秒).wav', format='wav')

