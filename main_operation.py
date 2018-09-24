# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:42:29 2018
Main Module

Available functions:

@author: 仲
"""

import sys
sys.path.append('F:\\system_program\\monitoring_condition') 
import numpy as np
import pandas as pd
from sklearn.externals import joblib
# import offline_program  离线模块训练
from _function.database_operation import query_data,write_data,timecircle
import _function.parameter_calculation as pc
from _function.anormal_detection import anomaly_detection

# In[0]
# 数据写入
#os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'   #设置环境变量，防乱码          
start_time = '2016-06-01 00:01:00'                          #时间，会从主函数读入 #sys.argv[1] 
end_time = timecircle(start_time,1)
# 一次传入两个参数         
data = query_data("select * from TB_RJ_REAL_RUN where TURID = 11 and CYTIME between '%s' and '%s'"%(end_time,start_time))                                          
# In[1]
# 数据分类
#大气相对湿度:V167、大气温度:V168、大气压力:V169、燃机功率:V1、
boundary = pd.DataFrame({'Power':data['V1'],'T':data['V168'],'H':data['V167'],'P':data['V169']})       
#压气机进口空气温度：V7;压气机进口空气压力：V8;压气机出口空气温度：V9;压气机出口空气压力：V10;燃机排气温度：V63;燃机排气压力：V64;排气流量:V67;
#天然气流量：V59;天然气温度：V60;天然气压力：V61;
gt= pd.DataFrame({'t1':data['V7'],'p1':data['V8'],'t2':data['V9'],'p2':data['V10'],
                  't4':data['V63'],'p4':data['V64'],'m4':data['V67'],'m_gas':data['V59'],
                  't_gas':data['V60'],'p_gas':data['V61']}) 
CLCSO = data['V186'] #计算T3

#余热锅炉进口烟温度：V104;余热锅炉排烟温度：V105;余热锅炉高压蒸汽流量：V106;余热锅炉高压蒸汽温度：V107;余热锅炉高压蒸汽压力：V108;
#余热锅炉中压蒸汽流量：V109;余热锅炉中压蒸汽温度：110;余热锅炉中压蒸汽压力：V111;余热锅炉低压蒸汽流量：112;余热锅炉低压蒸汽温度：V113;余热锅炉低压蒸汽压力：V114;
#余热锅炉冷再热蒸汽压力：V115;余热锅炉冷再热蒸汽温度：V116;余热锅炉冷再热蒸汽流量：V117;余热锅炉热再热蒸汽流量：V118;余热锅炉热再热蒸汽温度：V119;余热锅炉热再热蒸汽压力：V120;余热锅炉轴封加热器出口凝结水温度：V121;
Boiler_variables=data[['V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114','V115','V116','V117','V118','V119','V120','V121']]
#高压缸入口蒸汽压力：V125;高压缸入口蒸汽温度:V126;高压缸排汽压力:V127;高压缸排汽温度：V128;
#中压缸入口蒸汽压力:V129;中压缸入口蒸汽温度:V130;中压缸排汽压力:V131;中压缸排汽温度:V132;
#低压缸入口蒸汽压力:V133;低压缸入口蒸汽温度:V134;低压缸排汽温度:V135;凝汽器真空:V136;循环水入口温度:V137;
#蒸汽轮机功率:V122、机组热负荷:V166
ST_variables=data[['V125','V126','V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V122','V166']] 
# In[]
# 数据无效性处理

# In[2]
# 指标计算
gt_power = boundary['Power'] 
ambient_pressure = boundary['P']/10000 
# Gasturbine
t1 = gt['t1']+273.15
p1 = gt['p1']/1000+ambient_pressure    #P1单位是kpa
t2 = gt['t2']+273.15
p2 = gt['p2']+ambient_pressure
t4 = gt['t4']+273.15
p4 = gt['p4']/1000+ambient_pressure   #P4单位是kpa
m4 = gt['m4']
m_gas = gt['m_gas']
t_gas = gt['t_gas']+273.15 # 无数据
p_gas = gt['p_gas']+ambient_pressure # 无数据
# 透平进口温度压力计算
t3 = CLCSO/100*700+700+273
p3 = p2*0.99
# 燃气轮机指标
gasturbine = pc.Gasturbine(gt_power.iloc[-1],t1.iloc[-1],p1.iloc[-1],t2.iloc[-1],p2.iloc[-1],
                           t3.iloc[-1],p3.iloc[-1],
                           t4.iloc[-1],p4.iloc[-1],m4.iloc[-1],t_gas.iloc[-1],p_gas.iloc[-1],m_gas.iloc[-1])
c_pai = gasturbine.compressor.pressure_ratio()
gc_efficiency = gasturbine.compressor.c_efficiency(c_pai)
t_pai = gasturbine.turbine.expansion_ratio()
gt_efficiency = gasturbine.turbine.t_efficiency(t_pai)
g_efficiency = gasturbine.g_efficiency()  
g_heat_rate = gasturbine.heat_rate()        
# In[3]
# 基准值模型导入
# 燃气轮机
#压气机进口温度;压气机进口压力;压气机出口温度;压气机出口压力;透平出口温度;透平出口压力;
#排气流量;天然气流量;
reference_model = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
std_model = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[]}
for v in ['t1', 'p1', 't2', 'p2', 't4', 'p4', 'm4','m_gas']:
    reference_model[v] = joblib.load('F:/system_program/monitoring_condition/model/GLM_ref_{0}.pkl'.format(v))
    std_model[v] = joblib.load('F:/system_program/monitoring_condition/model/GLM_std_{0}.pkl'.format(v))

# 基准值计算
reference = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[],
                   'cpai':[],'ce':[],'te':[],'ge':[],'grc':[]}  #指标的基准值通过参数基准值计算得到
std = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[],
                   'cpai':[],'ce':[],'te':[],'ge':[],'grc':[]}
for v in ['t1', 'p1', 't2', 'p2', 't4', 'p4', 'm4','m_gas']:
    reference[v] = reference_model[v].predict(np.array(boundary[['Power','T']].iloc[-1]).reshape(1,-1))
    std[v] = std_model[v].predict(np.arrary(boundary[['Power','T']].iloc[-1]).reshape(1,-1))
#指标基准值计算,#压比；压气机效率；透平效率；燃气轮机效率；燃气轮机热耗率
# In[3]
# 异常检测(基于基准值区间)
lower_limit = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[],
                   'cpai':[],'ce':[],'te':[],'ge':[],'grc':[]}
upper_limit = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[],
                   'cpai':[],'ce':[],'te':[],'ge':[],'grc':[]}
indicator = {'t1':[], 'p1':[], 't2':[], 'p2':[], 't4':[], 'p4':[], 'm4':[],'m_gas':[],
                   'cpai':[],'ce':[],'te':[],'ge':[],'grc':[]}

# 涉及稳态检测
for v in ['t1', 'p1', 't2', 'p2', 't4', 'p4', 'm4','m_gas']:
    lower_limit[v] = reference[v]-3*std[v]
    upper_limit[v] = reference[v]+3*std[v]
    indicator[v] = anomaly_detection(gt[v],gt_power,lower_limit[v],upper_limit[v])
# In[]
# 特征提取



# In[4]
# 故障诊断

# In[5]
##写入oracle数据库
#指标
#透平入口烟气温度:T3,V176;透平入口烟气压力:P3,V177;
#压气机压比:c_pai,V174;压气机效率:gc_efficiency,V175;透平膨胀比:t_pai,V178;透平绝热效率:gt_efficiency,V179;
#燃机效率:g_efficiency,V180;燃机热耗率:g_heat_rate,V181;

sql_gt_1="INSERT INTO TB_RJ_REAL_RUN(ID,TURID,CYTIME,V176,V177,V174,V175,V178,V179,V180,V181)\
VALUES (seq_rj_common.nextval,11,'%s','%f','%f','%f','%f','%f','%f','%f','%f')"\
%(start_time,t3.iloc[-1],p3.iloc[-1],c_pai,gc_efficiency,t_pai,gt_efficiency,g_efficiency,g_heat_rate)
write_data(sql_gt_1)   #第？台机组

#基准值存入位置说明：
#压气机进口温度：V7;压气机进口压力：V8;压气机出口温度：V9;压气机出口压力：V10;透平出口温度：V63;透平出口压力：V64;
#天然气流量：V59;排气流量：V67;
#压气机压比：V174;压气机效率：,V175,透平绝热效率：V179;燃机效率：V180;燃机气耗率：V181;
sql_gt_2_realtime="INSERT INTO TB_RJ_REAL_REFERENCE(ID,TURID,CYTIME,V7,V8,V9,V10,V63,V64,V59,V67)\
VALUES (seq_rj_common.nextval,16,'%s','%f','%f','%f','%f','%f','%f','%f','%f')"\
%(start_time,reference['t1'],reference['p1'],reference['t2'],reference['p2'],reference['t4'],reference['p4'],reference['m4'],reference['m_gas'])
write_data(sql_gt_2_realtime)
#下限
sql_gt_3_realtime="INSERT INTO TB_RJ_REAL_REFERENCE_LOWER(ID,TURID,CYTIME,V7,V8,V9,V10,V63,V64,V59,V67)\
VALUES (seq_rj_common.nextval,16,'%s','%f','%f','%f','%f','%f','%f','%f','%f')"\
%(start_time,lower_limit['t1'],lower_limit['p1'],lower_limit['t2'],lower_limit['p2'],lower_limit['t4'],lower_limit['p4'],lower_limit['m4'],lower_limit['m_gas'])
write_data(sql_gt_3_realtime)
#上限
sql_gt_4_realtime="INSERT INTO TB_RJ_REAL_REFERENCE_UPPER(ID,TURID,CYTIME,V7,V8,V9,V10,V63,V64,V59,V67)\
VALUES (seq_rj_common.nextval,16,'%s','%f','%f','%f','%f','%f','%f','%f','%f')"\
%(start_time,upper_limit['t1'],upper_limit['p1'],upper_limit['t2'],upper_limit['p2'],upper_limit['t4'],upper_limit['p4'],upper_limit['m4'],upper_limit['m_gas'])
write_data(sql_gt_4_realtime)
#异常检测
sql_gt_5_realtime="INSERT INTO TB_RJ_REAL_ANORMALY(ID,TURID,CYTIME,V7,V8,V9,V10,V63,V64,V59,V67)\
VALUES (seq_rj_common.nextval,16,'%s','%f','%f','%f','%f','%f','%f','%f','%f')"\
%(start_time,indicator['t1'],indicator['p1'],indicator['t2'],indicator['p2'],indicator['t4'],indicator['p4'],indicator['m4'],indicator['m_gas'])
write_data(sql_gt_5_realtime)



     
    