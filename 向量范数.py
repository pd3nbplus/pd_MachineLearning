# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:48:40 2021

@author: 潘登
"""

#%%向量范数
import  numpy as np
a = np.array([1,-5,7,9,-456])
L1 = np.sum(np.abs(a))
L2 = np.sqrt(np.sum(np.square(a)))
E = np.identity(5)
