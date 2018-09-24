# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:39:25 2018

Module for P_ratio statistic training and online detection.

Available functions:
- p_ratio_training: p_ratio statistics training.
- p_ratio_detection:anormaly detection based on the P-ratio value. 
@author: 仲
"""

import numpy as np
import scipy.stats as stats
from sklearn.externals import joblib

def p_ratio_training(v,parameter,power,number = 20): 
    """
    P-ratio model training. The P-ratio statistic is based on the p-value of the T-test.
    
    Input:
    - v: the name of the characteristic variable(str)
    - parameter: the value of the characteristic variable
    - power: the output power
    - number : the length of the interval of T-test, default is 20.
    Output:
    -  trained P_limit is saved to local
    """
    #求取功率和特征参数的差值
    delta_power = np.array(power[1:])-np.array(power[:-1]) 
    delta_parameter = np.array(parameter[1:])-np.array(parameter[:-1])
    mean0_power = 0
    mean0_parameter = 0
    P_ratio = []
    for i in range(number+1, len(delta_power)):
        test_power = delta_power[i-number:i]
        test_parameter = delta_parameter[i-number:i]
        variance_power = np.var(test_power,ddof=0)  #样本方差不用无偏估计
        mean_power = np.mean(test_power)
        variance_parameter = np.var(test_parameter,ddof=0)  #样本方差不用无偏估计
        mean_parameter = np.mean(test_parameter)      
        # 功率和参数分别求t检验的p值
        T_power = np.sqrt(number) * (mean_power - mean0_power) / (np.sqrt(variance_power)+1e-10)
        T_parameter = np.sqrt(number) * (mean_parameter - mean0_parameter) / (np.sqrt(variance_parameter)+1e-10)
        P_power=(1-stats.t.cdf(abs(T_power),number-1))*2 
        P_parameter = (1-stats.t.cdf(abs(T_parameter),number-1))*2 
        # p-ratio特征统计量计算
        P_ratio.append(np.log((P_parameter+1)/(P_power+1)))
    # 根据统计量的分布，假设服从高斯分布，计算其分布的阈值下限,没有选用kde估计其概率密度
    P_limit = np.mean(P_ratio)-np.sqrt(np.var(P_ratio))*3
    #print(str(v)+'P_ratio统计量下限是：',P_limit)
    joblib.dump(P_limit,'F:/system_program/monitoring_condition/model/P_limit'+str(v)+'.pkl')
    return P_limit,P_ratio

def p_ratio_detection(delta_parameter,delta_power,P_limit,number = 20): 
    """
    利用变化速率进行异常检测
    """
    # 输入是功率和参数的前后差值的序列，而不是单值 
    mean0_power = 0
    mean0_parameter = 0
    if (len(delta_parameter)>number and len(delta_power)>number):
        power = delta_power[-number:]
        parameter = delta_parameter[-number:]
        variance_power = np.var(power,ddof=0)  #样本方差不用无偏估计
        mean_power = np.mean(power)
        variance_parameter = np.var(parameter,ddof=0)  #样本方差不用无偏估计
        mean_parameter = np.mean(parameter)      
        # 功率和参数分别求t检验的p值
        T_power = np.sqrt(number) * (mean_power - mean0_power) / (np.sqrt(variance_power)+1e-10)
        T_parameter = np.sqrt(number) * (mean_parameter - mean0_parameter) / (np.sqrt(variance_parameter)+1e-10)
        P_power=(1-stats.t.cdf(abs(T_power),number-1))*2 
        P_parameter = (1-stats.t.cdf(abs(T_parameter),number-1))*2 
        # p-ratio特征统计量计算
        P_ratio = np.log((P_parameter+1)/(P_power+1))
        if  P_ratio<P_limit:
            indicator = 1   #异常
        else:
            indicator = 0   #正常
    else: 
        indicator = 0      #不作判断，认为正常
        P_ratio = 0
    return indicator,P_ratio


