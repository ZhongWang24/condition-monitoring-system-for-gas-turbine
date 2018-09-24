# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:01:26 2018

Class for the unit which has the attributes：the measureble parameters, 
and the methods: the calculation of the hidden variables. 

Available Classes:
- Gasturbine
- Heatboiler
- Steamboiler
- Unit
- Sets
@author: 仲
"""

from pyXSteam.XSteam import XSteam


class Compressor():
    """
    A compressor of gas turbine.
    Temperature：K； Pressure：Mpa
    """
    _R=8.3145
    def __init__(self,t1,p1,t2,p2):
        self.t1 = t1   #压气机进口温度
        self.p1 = p1   #压气机进口压力
        self.t2 = t2   #压气机出口温度
        self.p2 = p2   #压气机出口压力
    # 压比
    def pressure_ratio(self):
        pai = self.p2/self.p1
        return pai
    # 压气机效率（等熵压缩）
    def c_efficiency(self,pai):
        t_average = (self.t1+self.t2)/2         #压气机段空气平均温度
        cp = 1.048-3.83*t_average/10**4+9.45*t_average**2/10**7-5.49*t_average**3/10**10+7.92*t_average**4/10**14 #空气定压比热
        cv = cp-Compressor._R/28.97                         #空气定容比热，28是空气分子量
        k = cp/cv                               #近似等于1.4
        t2_s = self.t1*pai**((k-1)/k)           #压气机出口等熵温度（理想温度）
        efficiency = (t2_s-self.t1)/(self.t2-self.t1)   
        return efficiency

class Turbine():
    """
    A turbine of gas turbine
    Temperature：K； Pressure：Mpa
    """
    _R = 8.3145
    _k = 1.37
    def __init__(self,t3,p3,t4,p4,m4):
        self.t3 = t3          #透平进口温度
        self.p3 = p3          #透平进口压力
        self.t4 = t4          #透平排气温度
        self.p4 = p4          #透平排气压力
        self.m4 = m4          #透平排气流量（标况下）
    # 透平膨胀比
    def expansion_ratio(self):
        pai = self.p3/self.p4
        return pai
    # 透平效率（等熵膨胀）
    def t_efficiency(self,pai):
        #t_average = (self.t3+self.t4)/2         #透平段燃气平均温度
        #cp = 0.991+6.997*t_average/10**5+2.712*t_average**2/10**7-1.2244*t_average**3/10**10 #燃气定压比热
        #cv = cp-_R/29.16                         #燃气定容比热
        #k = cp/cv                               #近似等于1.25
        t4_s = self.t3/(pai**((1.37-1)/1.37))         #透平出口等熵温度
        efficiency = (self.t3-self.t4)/(self.t3-t4_s) 
        return efficiency
class Naturalgas():
    """
    Some parameters about the natural gas.
    """
    h = 35                                     #天然气热值
    def __init__(self,t_gas,p_gas,m_gas):
        self.t_gas = t_gas                     #天然气温度
        self.p_gas = p_gas                     #天然气压力
        self.m_gas = m_gas                     #天然气流量(标况下）                           
        
class Gasturbine():
    """
    A gas turbine contains compressor and turbine and imput natural gas.
    power:MW
    """
    def __init__(self,power,t1,p1,t2,p2,t3,p3,t4,p4,m4,t_gas,p_gas,m_gas):
        self.power = power
        self.compressor = Compressor(t1,p1,t2,p2)
        self.turbine = Turbine(t3,p3,t4,p4,m4)
        self.naturalgas = Naturalgas(t_gas,p_gas,m_gas)
    # 燃气轮机发电效率
    def g_efficiency(self):
        efficiency = self.power*3600/(self.naturalgas.m_gas*self.naturalgas.h)    #正平衡
        return efficiency
    # 燃气轮机发电热耗率（KJ/KWh）
    def heat_rate(self):
        rate = (self.naturalgas.m_gas*self.naturalgas.h)/self.power
        return rate
    #燃气轮机发电气耗率（标准立方米/KWh）
    def gas_rate(self):
        rate = self.naturalgas.m_gas/(self.power*1000)
        return rate
 
       
class Boiler():
    """
    Three pressure reheat waste heat boiler
    """
    steamTable = XSteam(XSteam.UNIT_SYSTEM_BARE)      #单位：m/kg/sec/K/MPa/W
    
    def __init__(self,t_in,t_out,t_high,p_high,m_high,t_mid,p_mid,m_mid,t_low,p_low,m_low,
                 t_recold,p_recold,m_cold,t_rehot,p_rehot,m_rehot,t_heater,p_heater = 3.0 ):
        self.t_in = t_in                       #余热锅炉进口烟温度
        self.t_out = t_out                     #余热锅炉排烟温度        
        self.t_high = t_high                   #高压蒸汽温度
        self.p_high = p_high                   #高压气包出口蒸汽压力
        self.m_high = m_high                   #高压蒸汽流量        
        self.t_mid = t_mid                     #中压蒸汽温度
        self.p_mid = p_mid                     #中压蒸汽压力
        self.m_mid = m_mid                     #中压蒸汽流量        
        self.t_low = t_low                     #低压蒸汽温度
        self.p_low = p_low                     #低压蒸汽压力
        self.m_low = m_low                     #低压蒸汽流量
        self.t_recold = t_recold               #冷再热蒸汽温度
        self.p_recold = p_recold               #冷再热蒸汽压力
        self.m_recold = m_cold                 #冷再热蒸汽流量   
        self.t_rehot = t_rehot                 #热再热蒸汽温度
        self.p_rehot = p_rehot                 #热再热蒸汽压力
        self.m_rehot = m_rehot                 #热再热蒸汽流量   
        self.t_heater = t_heater               #轴封加热器出口凝结水温度
        self.p_heater = p_heater               #轴封加热器出口凝结水压力
        self.steamTable = XSteam()
    #锅炉吸热量
    def heat_absorption(self):        
        h_high = Boiler.steamTable.h_pt(self.p_high,self.t_high)           #高压汽包出口主蒸汽焓值
        h_mid = Boiler.steamTable.h_pt(self.p_mid,self.t_mid)              #中压汽包出口主蒸汽焓值
        h_low = Boiler.steamTable.h_pt(self.p_low,self.t_low)              #低压汽包出口主蒸汽焓值
        h_recold = Boiler.steamTable.h_pt(self.p_recold,self.t_record)     #冷再热蒸汽焓值
        h_rehot = Boiler.steamTable.h_pt(self.p_rehot,self.t_rehot)        #热再热蒸汽焓值
        h_heater = Boiler.steamTable.h_pt(self.p_heater,self.t_heater)     #轴封加热器出口凝结水焓值
        Q = (self.m_high*(h_high-h_heater)+self.m_recold*(h_rehot-h_recold )+
             self.m_mid*(h_rehot-h_heater)+self.m_lowL*(h_low-h_heater))   
        return Q
    #锅炉效率
    def b_efficiency(self,t_0):
        efficiency = (self.t_in-self.t_out)/(self.t_in-t_0)                 #t:环境温度
        return efficiency

class Steamturbine():
    """
    steam turbine and condenser system
    """
    steamTable = XSteam(XSteam.UNIT_SYSTEM_BARE)      #单位：m/kg/sec/K/MPa/W
    def __init__(self,power,heat,p_highi,t_highi,m_high,p_higho,t_higho,p_midi,t_midi,m_mid,p_mido,t_mido,
                 p_lowi,t_lowi,t_lowo,vacuum,t_wateri):
        self.power = power                     #蒸汽轮机发电量
        self.heat = heat                       #蒸汽轮机热负荷 
        self.p_highi = p_highi                 #高压缸入口蒸汽压力
        self.t_highi = t_highi                 #高压缸入口蒸汽温度
        self.m_high = m_high                   #高压缸入口蒸汽流量
        self.p_higho = p_higho                 #高压缸排气压力
        self.t_higho = t_higho                 #高压缸排气温度
        self.p_midi = p_midi                   #中压缸入口蒸汽压力
        self.t_midi = t_midi                   #中压缸入口蒸汽温度
        self.m_mid = m_mid                     #中压缸入口蒸汽流量
        self.p_mido = p_mido                   #中压缸排气压力
        self.t_mido = t_mido                   #中压缸排气温度
        self.p_lowi = p_lowi                   #低压缸入口蒸汽压力
        self.t_lowi = t_lowi                   #低压缸入口蒸汽温度
        self.t_lowo = t_lowo                   #低压缸排气温度
        self.vacuum = vacuum                   #凝汽器真空（低压缸排气压力）
        self.t_wateri = t_wateri               #循环水入口温度
    #高压缸效率
    def cylinder_efficiency_h(self):
        h_i = Steamturbine.steamTable.h_pt(self.p_highi,self.t_highi)   #高压缸进气汽焓
        h_o = Steamturbine.steamTable.h_pt(self.p_higho,self.t_higho)   #高压缸排汽焓
        s_i = Steamturbine.steamTable.s_pt(self.p_highi,self.t_highi)   #高压缸进汽熵
        t_s = Steamturbine.steamTable.t_ps(self.p_higho,s_i)       #高压缸等熵排汽温度
        h_os = Steamturbine.steamTable.h_pt(self.p_higho,t_s)      #高压缸等熵排汽焓
        efficiency = (h_i-h_o)/(h_i-h_os)        
        return efficiency
    
    #中压缸效率
    def cylinder_efficiency_m(self):
        h_i = Steamturbine.steamTable.h_pt(self.p_midi,self.t_midi)   #中压缸进气汽焓
        h_o = Steamturbine.steamTable.h_pt(self.p_mido,self.t_mido)   #中压缸排汽焓
        s_i = Steamturbine.steamTable.s_pt(self.p_midi,self.t_midi)   #中压缸进汽熵
        t_s = Steamturbine.steamTable.t_ps(self.p_mido,s_i)       #中压缸等熵排汽温度
        h_os = Steamturbine.steamTable.h_pt(self.p_mido,t_s)      #中压缸等熵排汽焓
        efficiency = (h_i-h_o)/(h_i-h_os)                 
        return efficiency
    
    #低压缸效率
    def cylinder_efficiency_l(self):
        h_i = Steamturbine.steamTable.h_pt(self.p_lowi,self.t_lowi)   #低压缸进气汽焓
        h_o = Steamturbine.steamTable.h_pt(self.vacuum,self.t_lowo)   #低压缸排汽焓
        s_i = Steamturbine.steamTable.s_pt(self.p_lowi,self.t_lowi)   #低压缸进汽熵
        t_s = Steamturbine.steamTable.t_ps(self.vacuum,s_i)       #低压缸等熵排汽温度
        h_os = Steamturbine.steamTable.h_pt(self.vacuum,t_s)      #低压缸等熵排汽焓
        efficiency = (h_i-h_o)/(h_i-h_os) 
        return efficiency

        #蒸汽轮机热效率
    def s_efficiency(self,Q):
        efficiency = (self.power*3600+self.heat*10**3)/Q
        return efficiency
    #蒸汽轮机热耗率（KJ/KWh）
    def s_heat_rat(self,Q):
        rate = Q/self.power
        return rate
    
    
def unit_calculation(gas_power,steam_power,heat,Q):
    """
    calculation of the indexes of a unit.where,gas_power contains ouput_power of three gas turbines,
    and output_power of three steam turbines.
    heat indicates the output heat supply of the unit,and Q indicates the heat generated by fuel consumption.
    """
    h = 35                                                        #天然气热值
    power = sum(gas_power)+sum(steam_power)                       #联合循环机组总功率
    ratio = sum(steam_power)/sum(gas_power)                       #蒸燃功比
    efficiency = (power+sum(heat))/sum(Q)                         #联合循环机组热效率
    heat_rate = sum(Q)/power                                      #联合循环机组发电热耗率
    gas_rate = heat_rate/(h*1000)                                 #联合循环机组气耗率
    
    return power,ratio,efficiency,heat_rate,gas_rate

       
    
if __name__=="__main__":
    # 实例化
    compressor = Compressor(299.6, 0.099, 706.6, 1.49)
    turbine = Turbine(1598.63, 1.43, 851.7, 0.1, 10000)
    naturalgas = Naturalgas(3.71, 484, 59082)
    gasturbine = Gasturbine(203, 299.6, 0.099, 706.6, 1.49, 1598.63, 1.43, 851.7, 0.1, 10000,3.71, 484, 59082)  
   
    # 二级指标计算
    c_pai = gasturbine.compressor.pressure_ratio()             
    c_e = gasturbine.compressor.c_efficiency(c_pai)
    t_pai = gasturbine.turbine.expansion_ratio()
    t_e = gasturbine.turbine.t_efficiency(t_pai)
    generation_power_e = gasturbine.g_efficiency()
    heat_consumption_e = gasturbine.heat_rate()
    gas_consumption_e = gasturbine.gas_rate()
    print('压比%.3f'%c_pai,'压气机效率%.3f'%c_e,'膨胀比%.3f'%t_pai,'透平效率%.3f'%t_e)
    print('燃机发电效率%.3f'%generation_power_e,'燃机发电热耗率%.3f'%heat_consumption_e,'燃机气耗率%.3f'%gas_consumption_e)
    


    
        
        
        
       


            

