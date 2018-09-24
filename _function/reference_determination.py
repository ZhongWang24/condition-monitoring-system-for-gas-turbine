# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:51:08 2018

Module for regression of the reference value.

Available functions:
- reference_regression: reference value regression model training.
@author: 仲
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib

def reference_regression(cl,v,x,y,num):
    """
    Reference_value regression model.
    
    Input:
    - cl:'ref',reference;'std',standard variance 
    - v: variable name: str
    - x: boundary variable.
    - y: objective variable.
    - num: polynomial regression order
    Output：
    - reference：reference value under typical working condition 
    """
    def evaluate(y_true,y_pred):
        mse = mean_squared_error(y_true,y_pred)
        r2 = r2_score(y_true,y_pred)
        return mse,r2

    #广义线性回归
    GLM = Pipeline([('poly', PolynomialFeatures(degree=num)),
                      ('linear', LinearRegression())])
    GLM.fit(x,y)
    y_pred_G = GLM.predict(x)
    mse,r2 = evaluate(y,y_pred_G)
    joblib.dump(GLM,'F:/system_program/monitoring_condition/model/GLM_{0}_{1}.pkl'.format(cl,v))
    # 贝叶斯线性回归
    #linear_re = BayesianRidge()
    #linear_re.fit(x,y)
    #y_pred = linear_re.predict(x)
    #mse,r2 = evaluate(y,y_pred)
    #return linear_re,linear_re.coef_,linear_re.intercept_
    #返回 模型，系数，截距，评价准则
    return GLM,GLM.named_steps['linear'].coef_,GLM.named_steps['linear'].intercept_,mse,r2

# In[1]
if __name__=='__main__':
    #基准值回归    
    reference = pd.read_csv('F:/system_program/monitoring_condition/data/reference_samples.csv').drop(columns=['clusters'])
    # 测试样本    
    X_test = {'P':[],'T':[]}    # x表示功率，y表示温度
    Z = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    cof = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    interpret = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    mse = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    r2 = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    for i in np.linspace(150,300,50):
        for j in np.linspace(0,30,50):
            X_test['P'].append(i)
            X_test['T'].append(j)
    X = pd.DataFrame(X_test)
    for v in ['t1', 'p1', 't2', 'p2', 't4', 'p4', 'm4','m_gas']:
        model,cof[v],interpret[v],mse[v],r2[v] = reference_regression('ref',v,reference[['power','T']],reference[v],2)
        Z[v] = model.predict(X)
# In[]    
    # 画图
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    plt.style.use('seaborn')  #设置绘图样式
    # matlab风格画图
    for v in ['t1', 'p1', 't2', 'p2', 't4', 'p4', 'm4','m_gas']:
        fig = plt.figure()   # 创建图形
        ax = plt.axes(projection='3d')
        ax.scatter3D(reference['power'],reference['T'],reference[v])
        ax.scatter3D(X['P'],X['T'],Z[v],color='gray',alpha=0.2)
        ax.set_xlabel('Power')
        ax.set_ylabel('Temperature')
        ax.set_zlabel('{0}'.format(v))
    
# In[2]
    # 标准差回归 
    reference_sd = pd.read_csv('F:/system_program/monitoring_condition/data/reference_std_samples.csv').drop(columns=['clusters'])    
    # 测试样本    
    X_test_std = {'P':[],'T':[]}    # x表示功率，y表示温度
    Z_std = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    cof_std = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    interpret_std = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    mse_std = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    r2_std = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
    for i in np.linspace(150,300,50):
        for j in np.linspace(0,30,50):
            X_test_std['P'].append(i)
            X_test_std['T'].append(j)
    X_std = pd.DataFrame(X_test_std)
    for v in ['t1', 'p1', 't2', 'p2', 't4', 'p4', 'm4','m_gas']:
        model_std,cof_std[v],interpret_std[v],mse_std[v],r2_std[v] = reference_regression('std',v,reference[['power','T']],reference_sd[v],1)
        Z_std[v] = model_std.predict(X_std)
# In[]    
    # 画图
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    plt.style.use('seaborn')  #设置绘图样式
    # matlab风格画图
    for v in ['t1', 'p1', 't2', 'p2', 't4', 'p4', 'm4','m_gas']:
        fig = plt.figure()   # 创建图形
        ax = plt.axes(projection='3d')
        ax.scatter3D(reference['power'],reference_sd['T'],reference_sd[v])
        ax.scatter3D(X_std['P'],X_std['T'],Z_std[v],color='gray',alpha=0.2)
        ax.set_xlabel('Power')
        ax.set_ylabel('Temperature')
        ax.set_zlabel('{0}'.format(v))


