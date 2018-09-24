# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:42:29 2018
Module for query and write data from Oracle 
Available functions:
-query_data
-write_data
-timecircle
@author: 仲
"""
import pandas as pd
import cx_Oracle
import time
import datetime

tns = cx_Oracle.makedsn('202.204.75.39',1521,'orcl10g') # 连接服务器地址
db = cx_Oracle.connect('imsoft_rj','imsoft_rj',tns)     # 连接服务器上数据库
cr = db.cursor()                                        # 创建cursor                  
def query_data(strSql):   
    """
    Input expression
    """
    global cr
    cr.execute(strSql)
    data=pd.DataFrame(cr.fetchall())
    title=[i[0] for i in cr.description]                #获取数据表的列名，并输出
    data.columns=title
    return data

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
    end_time = a - datetime.timedelta(minutes = interval)
    end_time = str(end_time)
    return end_time

 
if __name__=="__main__":
    start_time = '2016-06-01 01:24:00' 
    end_time = timecircle(start_time,30)         
    data = query_data("select * from TB_RJ_REAL_RUN where TURID = 16 and CYTIME between '%s' and '%s'"%(start_time,end_time)) 
    for index,row in data.iterrows():              #iterrow()函数的作用是对DateFrame进行遍历，输出index和row
        power = row['V1']                         #取了某一机组一段时间的数据，但一次只能取一组数据
        boundary = row['V7']                                       

