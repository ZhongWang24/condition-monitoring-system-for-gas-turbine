# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 23:04:45 2018

Module for anomaly detection.

Available functions:

- anormaly_detection: online detecion based on the reference value limit and p-ratio limit.


@author: 仲
"""

import numpy as np
from sklearn.externals import joblib

from _function.steady_state_detection import steady_detection
def anomaly_detection(variable,power,lower_limit,upper_limit): 
    #v是变量名称字符，variable是变量数组，power是功率数组,boundary是边界条件,time当前时刻索引
    #导入稳态模型
    unsteady_model = joblib.load('model/unsteady_model.pkl')
    steady_model = joblib.load('model/steady_model.pkl')
    steady_ratio = joblib.load('model/steady_ratio.pkl')
     # 稳态判别.若数据是稳态，则接着进行异常检测
    delta_power = np.array(power.iloc[-1])-np.array(power.iloc[-2])   #loc是根据索引读数      
    indicator_steady = steady_detection(delta_power,steady_model,unsteady_model,steady_ratio)
    if (indicator_steady == 0):         
        if (variable.iloc[-1]<lower_limit) or (variable.iloc[-1]>upper_limit):# 基于基准值区间的异常检测
            indicator = 1
        else:
            indicator = 0
    else:
        indicator = -1  
    return indicator             #输出-1：非稳态；输出0：正常；输出1：异常

 