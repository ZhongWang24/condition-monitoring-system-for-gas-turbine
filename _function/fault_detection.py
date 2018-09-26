# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:05:58 2018

@author: 仲

fault detection for gas turbine based on Bayesian Network using the pgmpy 

Available functions:
- network_construction
- network_inference_online:  used for the input in Series format

 
"""
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.externals import joblib

def network_construction():
    """
    Construction of the Bayesian_network
    
    input\output:none
    if the sencente "check the Bayesian network again" happens,please check the model parameters again!
    
    Symbol description:
        mode：   CS: 压气机喘振；CF: 压气机结垢；CC：压气机磨损腐蚀；CI:压气机进气口结冰；TF：透平结垢；TC:透平磨损腐蚀；
                 TD:透平叶片机械损伤；BF：燃烧室故障；BP：燃烧脉动；HW:轮间温度高；HB:叶片通道温差大
        symptom：v：轰鸣声音；s：转速波动；f1：压气机压力波动；dp：压气机入口压差；r：压比；m1：压气机入口流量；m2:透平出口流量
                t2：压气机出口温度；p2：压气机出口压力；t4：透平排烟温度；f2 燃烧室压力波动；d：排烟分散度
                ce：压气机效率；te：透平效率；fi:空气过滤器滤芯结冰；htw:运行中某一点轮间温度高于限定值；
                htb：叶片通道（BPT）温度与平均值温度偏差大于报警限值
    """
   
    fault_model = BayesianModel([('CS','v'),('CS','s'),('CS','f1'),
                             ('CF','dp'),('CF','m1'),('CF','r'),('CF','ce'),
                             ('CC','m1'),('CC','ce'),
                             ('ce','p2'), ('ce','t2'),
                             ('TF','te'),('TF','m2'),
                             ('TC','te'),('TC','m2'),
                             ('te','p2'),('te','t4'),
                             ('BF','d'),
                             ('BP','f2'),('BP','d'),
                             ('CI','fi'),
                             ('TD','te'),
                             ('HB','htb'),
                             ('HW','htw')])
    # defining the parameters(conditional probability).      
    # 故障模式的先验概率 (共11个故障模式)
    cs_cpd = TabularCPD(variable='CS',variable_card=2,values=[[0.05,0.95]])#压气机喘振
    cf_cpd = TabularCPD(variable='CF',variable_card=2,values=[[0.2,0.8]])#压气机叶片结垢
    cc_cpd = TabularCPD(variable='CC',variable_card=2,values=[[0.1,0.9]])#压气机叶片磨损腐蚀
    ci_cpd = TabularCPD(variable='CI',variable_card=2,values=[[0.03,0.97]])#压气机进气口结冰
    tf_cpd = TabularCPD(variable='TF',variable_card=2,values=[[0.1,0.9]])#透平叶片结垢
    tc_cpd = TabularCPD(variable='TC',variable_card=2,values=[[0.1,0.9]])#透平叶片磨损腐蚀
    td_cpd = TabularCPD(variable='TD',variable_card=2,values=[[0.05,0.95]])#透平叶片机械损伤
    bf_cpd = TabularCPD(variable='BF',variable_card=2,values=[[0.1,0.9]])#燃烧室故障 
    bp_cpd = TabularCPD(variable='BP',variable_card=2,values=[[0.1,0.9]])#燃烧脉动
    hw_cpd = TabularCPD(variable='HW',variable_card=2,values=[[0.1,0.9]])#轮间温度高
    hb_cpd = TabularCPD(variable='HB',variable_card=2,values=[[0.1,0.9]])#叶片通道温差大
    # 故障征兆的条件概率。(共14个征兆，其中2个是能效指标)以Noise-or为原则进行赋值。
    #父节点认为相互独立；泄露概率0.01；0.9 强烈关联；0.8代表有关联；0.7代表可能关联 0.6代表不确定关联是否成立
    ce_cpd = TabularCPD(variable = 'ce',variable_card = 2,evidence = ['CF','CC'],evidence_card = [2,2],
                        values = [[0.99,0.1,0.1,0.0099],[0.01,0.9,0.9,0.9901]])    #能效异常模式
    te_cpd = TabularCPD(variable = 'te',variable_card = 2,evidence = ['TF','TC','TD'],evidence_card = [2,2,2],
                        values = [[0.99,0.1,0.1,0.0099,0.1,0.0099,0.0099,0.00099],[0.01,0.9,0.9,0.9901,0.9,0.9901,0.9901,0.99901]])     #能效异常模式
    v_cpd = TabularCPD(variable = 'v',variable_card = 2,evidence = ['CS'],evidence_card = [2],
                       values = [[0.99,0.1],[0.01,0.9]])
    s_cpd = TabularCPD(variable = 's',variable_card = 2,evidence = ['CS'],evidence_card = [2],
                       values = [[0.99,0.1],[0.01,0.9]])
    f1_cpd = TabularCPD(variable = 'f1',variable_card = 2,evidence = ['CS'],evidence_card = [2],
                        values = [[0.99,0.1],[0.01,0.9]])
    dp_cpd = TabularCPD(variable = 'dp',variable_card = 2,evidence = ['CF'],evidence_card = [2],
                        values = [[0.99,0.1],[0.01,0.9]])
    m1_cpd = TabularCPD(variable = 'm1',variable_card = 2,evidence = ['CF','CC'],evidence_card = [2,2],
                        values = [[0.99,0.1,0.2,0.0198],[0.01,0.9,0.8,0.9802]])
    r_cpd = TabularCPD(variable = 'r',variable_card = 2,evidence = ['CF'],evidence_card = [2],
                       values = [[0.99,0.1],[0.01,0.9]])
    t2_cpd = TabularCPD(variable = 't2',variable_card = 2,evidence = ['ce'],evidence_card = [2],
                        values = [[0.99,0.1],[0.01,0.9]])
    p2_cpd = TabularCPD(variable = 'p2',variable_card = 2,evidence = ['ce','te'],evidence_card = [2,2],
                        values = [[0.99,0.3,0.2,0.0594],[0.01,0.7,0.8,0.9406]])
    t4_cpd = TabularCPD(variable = 't4',variable_card = 2,evidence = ['te'],evidence_card = [2],
                        values = [[0.99,0.1],[0.01,0.9]])
    m2_cpd = TabularCPD(variable = 'm2',variable_card = 2,evidence = ['TF','TC'],evidence_card = [2,2],
                        values = [[0.99,0.2,0.1,0.0198],[0.01,0.8,0.9,0.9802]])
    f2_cpd = TabularCPD(variable = 'f2',variable_card = 2,evidence = ['BP'],evidence_card = [2],
                        values = [[0.99,0.1],[0.01,0.9]])
    d_cpd = TabularCPD(variable = 'd',variable_card = 2,evidence = ['BF','BP'],evidence_card = [2,2],
                       values = [[0.99,0.2,0.1,0.0198],[0.01,0.8,0.9,0.9802]])
    fi_cpd = TabularCPD(variable = 'fi',variable_card = 2,evidence = ['CI'],evidence_card = [2],
                        values = [[0.99,0.1],[0.01,0.9]])
    htb_cpd = TabularCPD(variable = 'htb',variable_card = 2,evidence = ['HB'],evidence_card = [2],
                        values = [[0.99,0.1],[0.01,0.9]])
    htw_cpd = TabularCPD(variable = 'htw',variable_card = 2,evidence = ['HW'],evidence_card = [2],
                        values = [[0.99,0.1],[0.01,0.9]])
    #Associating the parameters with the model structure.
    fault_model.add_cpds(cs_cpd,cf_cpd,cc_cpd,ci_cpd,tf_cpd,tc_cpd,td_cpd,bf_cpd,bp_cpd,hb_cpd,hw_cpd,
                         ce_cpd,te_cpd,
                         v_cpd,s_cpd,f1_cpd,dp_cpd,
                         m1_cpd,r_cpd,t2_cpd,p2_cpd,t4_cpd,m2_cpd,
                         f2_cpd,d_cpd,fi_cpd,htb_cpd,htw_cpd)
    # Checking if the cpds are valid for the model.
    try:
        fault_model.check_model()
    except ValueError:
        print('check the Bayesian network again')
    else:
        joblib.dump(fault_model,'model/fault_model.pkl')  #模型存储
    return fault_model

def feature_extraction(data):
    feature = pd.DataFrame({'p2':[0],'t2':[0],'t4':[0],'r':[0],'m1':[0],'m2':[0],'ce':[0],'te':[0],
               'f1':[0],'dp':[0],'f2':[0],'d':[0],'fi':[0],'v':[0],'s':[0],'htb':[0],'htw':[0]})
    try:        
        if (data['p2']>0.5):
            feature['p2'] = [1]
    except:
        pass        
    try:   
        if data['t2']>0.5:
            feature['t2'] = [1]
    except:
        pass       
    try:
        if data['m4']>0.5:
            feature['m2'] = [1]
    except:
        pass
    try:  # m1:压气机出口空气流量        
        if (data['m1']>0.5):    
            feature['m1'] = [1]
    except:
        pass 
    try:  # r:压比        
        if (data['c_pai']>0.5):    
            feature['r'] = [1]
    except:
        pass 
    try:  # ce:压气机效率        
        if (data['c_efficiency']>0.5):    
            feature['ce'] = [1]
    except:
        pass 
    try:  # te:透平效率        
        if (data['t_efficiency']>0.5):    
            feature['te'] = [1]
    except:
        pass     
    return feature
        

def network_inference(network_model,data):
    """    
    predict the probability of the state of the fault mode(all the missing variables).
    
    input:pandas Series object at each time!!!
    output:the predicted state
    """
    # 输入情况异常处理
    if set(data.index) == set(network_model.nodes()):
        raise ValueError("No variable missing in data. Nothing to predict")
    elif set(data.index) - set(network_model.nodes()):
        raise ValueError("Data has variables which are not in the model")

    missing_variables = set(network_model.nodes()) - set(data.index)
    # 选择精确推理，变量消除
    model_inference = VariableElimination(network_model)
    # iterrows 和 下面的 items 分别是针对datafram和字典创建的生成迭代器  
    states_dict = model_inference.query(variables=missing_variables, evidence=data.to_dict()) #对每行（条）状态进行推理
    for k, v in states_dict.items():
        l = len(v.values)-1
        if v.values[l]>0.5:
            print(k ,'probabilily occured: %.3f'%v.values[l])
        else:
            #print('Normal')
            return states_dict 

        
if __name__=='__main__':
    fault_model = network_construction()
    data = pd.DataFrame({'p2':[0],'t2':[0],'m4':[0]}) 
    feature = feature_extraction(data)
    result = fault_model.predict(feature)
    #y = network_inference(fault_model,feature)



                






