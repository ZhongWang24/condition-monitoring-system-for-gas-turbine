# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:30:38 2018

@author: 仲
"""


def Time (powerprice = 0, gasprice = 0 ):
    # powerprice  #上网电价
    # gasprice 天然气价格
    Pgt = 0 #燃气轮机功率
    Pst = 0 #蒸汽轮机功率，
    waterprice = 0 #：除盐水成本   
    Timeinterval = 0 #水洗停机时间
    G0 = 0 #G0:机组清洁状燃料消耗率
    T1 = (waterprice+(Pgt+Pst)*Timeinterval*powerprice)-G0*gasprice*Pgt*Timeinterval
    T2 = 0.05/100*Pgt*(0.5*powerprice+0.125*G0*gasprice)
    time = (T1/T2)**0.5     
    return time
time_interval = Time()
#time_interval  写入数据库



def Drop (presratio,presratio0): 
    #presratio：实际压比   # presratio0：基准压比
    drop=(presratio-presratio0)/presratio0  
    
    return drop

drop_upper = 0.5
# drop_upper 存入数据库