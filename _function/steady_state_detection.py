# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:29:50 2018

Module for steady_state detection.

Available functions:
- steady_division: for the off-line data division between the steady and unsteady state. 
- steady_training: for the steady and unsteady model training 
- steady_detection: for the on-line data detection used the trained model.

@author: 仲
"""

import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib


def steady_division(power,interval = 20, alpha = 0.05):
    """
    Divide the steady training data and the unsteady training data
    by the interval estimation of the expectation(mean) of the delta_power.
    
    Input: 
    - output power of the unit(Series),
    - length of the estimnation interval, default is 20 
    - confidence level,default is 0.05
    Output:
    - index of the steady and unsteady training data respectively
    - steady and unsteady delta-power respectively
    ratio of the steady and unsteady training data is saved to local.
    
    """
    delta_power = np.array(power[1:])-np.array(power[:-1])
    u = stats.norm.ppf(1 - alpha/2)
    sigma = 0.4288  
    #sigma = np.sqrt(np.var(power[4400:4800]))    
    index_steady = []
    index_unsteady = []
    delta_power_steady = []
    delta_power_unsteady = []
    for i in range(0, len(delta_power), interval):
        mu = np.mean(delta_power[i:i + interval])
        mu1,mu2 = mu - u * sigma / np.sqrt(interval),mu + u * sigma / np.sqrt(interval)
        if (mu1*mu2<0) :
            index_steady.extend(list(range(i,i + interval))) 
            delta_power_steady.extend(delta_power[i:i + interval])   
        else :
            index_unsteady.extend(list(range(i,i + interval)))
            delta_power_unsteady.extend(delta_power[i:i + interval]) 
    steady_ratio = len(index_steady)/(len(index_steady)+len(index_unsteady))
    #print('稳态数据的比例是 = ',steady_ratio)
    joblib.dump(steady_ratio,'F:/system_program/monitoring_condition/model/steady_ratio.pkl')
    index = (index_steady,index_unsteady)
    delta_power = (delta_power_steady,delta_power_unsteady)
    return index,delta_power


def steady_training(delta_power_steady,delta_power_unsteady, number = 5):
    """
    Training the steady and unsteady model by the delta_power 
    based on the Gaussian model and Gaussian mixture model respectively.
    
    Input:
    - steady delta_power (Array)
    - unsteady delta_power
    - number of the sub-models of the GMM, default is 5
    Output:
      None
    The trained unsteady and steady model is saved to local.  
    
    """
    # 训练稳态样本的高斯模型
    mean0 = np.mean(delta_power_steady)
    s_variance0 = np.sqrt(np.var(delta_power_steady,ddof=1))  # 方差无偏估计
    # 训练非稳态样本的高斯混合模型 
    gmm = GaussianMixture(n_components=number,covariance_type='tied')   
    gmm.fit(np.array(delta_power_unsteady).reshape(-1,1))   
    # 需要对输入数组进行转置，其中行向量代表样本数，列向量代表维度
    # 模型保存到本地路径
    joblib.dump(gmm,'F:/system_program/monitoring_condition/model/unsteady_model.pkl')
    joblib.dump((mean0,s_variance0),'F:/system_program/monitoring_condition/model/steady_model.pkl')
    

def steady_detection(delta_power,steady_model,gmm,prior=0.9): 
    """
    Steady detection of the online data.
    The method used here is the Gaussian discriminant analysis, a generation model
    
    Input:
    - difference  between the power at the current time and the adjacent previous time.
    - trained steady-model
    - trained unsteady-model(GMM)
    - prior probability of the steady data.
    Output:
    - detection result. "0" represents steady, and "1" represents unsteady
    """    
    p_1 = 1-prior
    p_0 = prior
    pdf_0 = stats.norm.pdf(delta_power, loc=steady_model[0], scale=steady_model[1])
    pdf_1 = np.exp(gmm.score_samples(delta_power))   
    value = (pdf_0 * p_0)/(pdf_1 * p_1)
    if(value>1):
        indicator = 0
        #print('steady')
    else:
        indicator = 1
        #print('unsteady')
    return indicator


