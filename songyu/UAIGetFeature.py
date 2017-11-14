# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:35:52 2017

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
#plot A->B orders count by time

weatherPath ='./Data/weather.csv'
posPath ='./Data/poi.csv'




def getPoiFeature():
    # 11
    resultDic ={}
    pos=pd.read_csv(posPath,encoding = "gbk",header=None)
    posName=pos[0]
    
    for i in range(len(posName)):
        result=[]
        #print (pos.loc[(pos[0]==poStr),range(2,21,2)])
        result.append(pos.loc[(pos[0]==posName[i]),range(2,21,2)].index[0])
        result.extend(pos.loc[(pos[0]==posName[i]),range(2,21,2)].iloc[0])
        #print (len(result))
        resultDic[posName[i]]=result
    #print (resultDic)
    
    return resultDic


def getWeatherFeature():
    
    #12
    # 日期编号 当前小时内气候是否变化 当前所属小时 code,temperature,feels_like,pressure,humidity,visibility,wind_direction_degree,wind_speed,wind_scale
    raweather=pd.read_csv(weatherPath)
    raweather['hour']=raweather['date'].str.split(' ',expand=True)[1].str.split(':',expand=True)[0]
    raweather['mdate']=raweather['date'].str.split(' ',expand=True)[0]
    dateIndex=(raweather['date'].str.split(' ',expand=True)[0].unique())
    result = {}
    i=0
    while i < len(raweather):
        tmp=[]
        key=[0,0]
        flag=False
        if i+1<len(raweather): 
         if raweather.iloc[i][12] == raweather.iloc[i+1][12]:
             if raweather.iloc[i][2] == raweather.iloc[i+1][2]:
                key=[list(dateIndex).index(raweather.iloc[i][13]),0]
             else:
                if not((raweather.iloc[i][2]==0 and raweather.iloc[i+1][2]==1) \
                or (raweather.iloc[i][2]==1 and raweather.iloc[i+1][2]==0)):
                    key=[list(dateIndex).index(raweather.iloc[i][13]),1]
                else:
                    key=[list(dateIndex).index(raweather.iloc[i][13]),0]
             flag=True
         else:
             key=[list(dateIndex).index(raweather.iloc[i][13]),0]
        
        tmp.extend(key)
        tmp.extend(raweather.iloc[i][[12]].apply(lambda x : int(x)))
        tmp.extend(raweather.iloc[i][[2,3,4,5,6,7,9,10,11]].tolist())
        
        result[str(key[0])+'-'+raweather.iloc[i][12]]=tmp
        
        if flag:
            i=i+1
        i=i+1
    

    #print (result)
   
  

    return result

# 10 features

def getDateFeature(): 
    #10
    #dateIndex=pd.Series(pd.read_csv(weatherPath)['date'].str.split(' ',expand=True)[0].unique())
    #print (dateIndex)
    NumofDays = 38
    #星期几 是否是工作日 工作日第几天 是否是节假日 节假日第几天 是否节前 是否节后
    weekFeature =[
                [6,0,0,1,1,0,0], # 星期六
                [7,0,0,1,2,0,0],
                [1,1,1,0,0,0,1],
                [2,1,2,0,0,0,0],
                [3,1,3,0,0,0,0],
                [4,1,4,0,0,0,0],
                [5,1,5,0,0,1,0]
             ]
    #是否月初 1-3，是否月中14-16，是否月末 29-31
    result = {}
    for i in range(NumofDays):
        tmp =[]
        tmp2=[0,0,0]
        if i in range(1,3):
            tmp2[0]=1
        if i in range(14,16):
            tmp2[1]=1
        if i in range(29,31):
            tmp2[2]=1
        tmp.extend(weekFeature[i%7])
        tmp.extend(tmp2)
        
        result[i]=tmp
    #print ((result))
    
    return result
        
    
if __name__=="__main__":
    #trainSet = ReadTrain()
    #testSet = ReadTest()
    #DataSplit(trainSet)
    #getSplitedTrainData(trainSet,testSet)
    #getAverage(31,31)
    #getAverageWithnormal(31,31,1)
    #getSplitedWeather(trainSet,testSet)
    #analysis(trainSet,testSet)
    #getPoiFeature()
    getPoiFeature()
    
    
    
    