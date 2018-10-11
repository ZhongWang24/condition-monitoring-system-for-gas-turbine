# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:16:50 2018

利用python将csv数据导入数据库

@author: 仲
"""
import pandas as pd
import cx_Oracle
import time
import datetime
import sys
sys.path.append('F:\\system_program\\monitoring_condition') 

tns = cx_Oracle.makedsn('202.204.75.39',1521,'orcl10g') # 连接服务器地址
db = cx_Oracle.connect('imsoft_rj','imsoft_rj',tns)     # 连接服务器上数据库
cr = db.cursor()                                        # 创建cursor   

def write_data(sql):   
    """
    Input expression
    """
    global cr,db
    cr.execute(sql)  
    return db.commit()

def timecircle(start_time,interval):
    a = time.strptime(start_time,"%Y-%m-%d %H:%M:%S")     #
    Y,m,d,H,M,S = a[0:6]
    a = datetime.datetime(Y,m,d,H,M,S)
    end_time = a + datetime.timedelta(minutes = interval)
    end_time = str(end_time)
    return end_time



                          
# In[0] 数据读取
data1 = pd.read_csv('data/3yue.csv')   #调整路径
data2 = pd.read_csv('data/6yue.csv')
data3 = pd.read_csv('data/9yue.csv')
data4 = pd.read_csv('data/12yue.csv')
d1 = data1.iloc[7199:14399]
d2 = data2.iloc[0:7200]
d3 = data3.iloc[7199:14399]
d4 = data4.iloc[7199:14399]
#data = [d1,d2,d3,d4]
#data_train = pd.concat(data,ignore_index = True)
data_train = data4[:1440]
# In[2]数据写入
start_time = '2018-09-10 00:00:00'
save_time = 20180901
for i in range(len(data_train)):
    sample = data_train.iloc[i]
    power = sample['Power']
    h = sample['H']
    t = sample['T']
    p = sample['P']
    t1 = sample['t1']
    p1 = sample['p1']
    t2 = sample['t2']
    p2 = sample['p2']
    t4 = sample['t4']
    p4 = sample['p4']
    m4 = sample['m4']
    m_gas = sample['m_gas'] 
    clcso = sample['CLCSO']
    sql_gt_1 = "INSERT INTO TB_RJ_REAL_RUN(ID,TURID,CYTIME,V1,V167,V168,V169,V7,V8,V9,V10,V63,V64,V67,V59,V186)VALUES(seq_rj_common.nextval,11,'%s','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f')"%(start_time,power,h,t,p,t1,p1,t2,p2,t4,p4,m4,m_gas,clcso)
    sql_gt_2 = "INSERT INTO TB_RJ_HIS_RUN(ID,TURID,CYTIME,SAVETIME,V1,V167,V168,V169,V7,V8,V9,V10,V63,V64,V67,V59,V186)VALUES(seq_rj_common.nextval,11,'%s','%s','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f')"%(start_time,save_time,power,h,t,p,t1,p1,t2,p2,t4,p4,m4,m_gas,clcso)
    write_data(sql_gt_1)  #第？台机组
    write_data(sql_gt_2)
    start_time = timecircle(start_time,1) 



