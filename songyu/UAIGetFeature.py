# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:35:52 2017

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import UAIContest as UC
import numpy as np

weatherPath ='./Data/weather.csv'
posPath ='./Data/poi.csv'



def getridofOutlier(preSeries):
    s=preSeries.quantile([.25, .5, .75])
    for i in range (len(preSeries)):
        delta = s.iloc[2]+1.5*(s.iloc[2]-s.iloc[0])
        if preSeries[i]> delta:
            preSeries[i] = delta
    return preSeries


def getAvgHourFeature(NumofDays):

    dir= './getAvgHourFeature/'
    
    for i in range (31-NumofDays,31):
        res = pd.read_csv(dir+str(i)+'.csv').values
        if i == 31-NumofDays:
            trainSet= np.array(res) 
        else:
            trainSet=trainSet+np.array(res) 
    trainSet = np.array(trainSet) / NumofDays
    print ((trainSet[0]))
    return trainSet

def getHourFeature(NumofDays):
    # 11
    #dateIndex=trainSet['create_date'].unique()
    dir= './getAvgHourFeature/'
    
    for i in range (31-NumofDays,31):
        tmp = pd.read_csv(dir+str(i)+'.csv')
        
        if i == 31-NumofDays:
            trainSetSortedByDate= tmp
            
        else:
            trainSetSortedByDate=pd.concat([trainSetSortedByDate,tmp],axis=1)
    
    size = trainSetSortedByDate.shape
    print (size[1])
    
    trainSetSortedByHour = pd.DataFrame()
    trainSetSortedByHour.insert (0,0,trainSetSortedByDate.iloc[:,0]) 
    cnt = 0
    for i in range (24):
        for j in range (1,size[1]):
            if j % 24 == i:
                cnt =cnt+1
                trainSetSortedByHour.insert(cnt,cnt,trainSetSortedByDate.iloc[:,j])
    
    #print (trainSetSortedByHour.mean(1))
    #print (trainSetSortedByDate.mean(1))
    
   # wres = pd.DataFrame()
   # wres['test_id'] = range(5000)
    
    result = trainSetSortedByHour.values
    trainMean = []  
    pre= []
    
    print (np.mean(result[0]))
    for num in range(len(result)):
        print (num)
    #for num in range(10):
        res = []
        ress = []
        for i in range (24):
            
            if num == 0:
                pass
                print ('\n',NumofDays*i,NumofDays*i+NumofDays)
                #print (result[num][NumofDays*i:NumofDays*i+NumofDays])
                print (np.mean(result[num][NumofDays*i:NumofDays*i+NumofDays]))
                print (np.mean(getridofOutlier(pd.Series(result[num][NumofDays*i:NumofDays*i+NumofDays])).tolist()))
            
            ress.append(np.mean(result[num][NumofDays*i:NumofDays*i+NumofDays]))
            res.append(np.mean(getridofOutlier(pd.Series(result[num][NumofDays*i:NumofDays*i+NumofDays])).tolist()))
        trainMean.append(res)
        pre.append(ress)
    
    
    #wres['count']= trainMean
    wres = pd.DataFrame(trainMean)
    wres.to_csv(str(NumofDays)+'.csv',encoding='utf-8',index = False)
    pd.DataFrame(pre).to_csv(str(NumofDays)+'pre.csv',encoding='utf-8',index = False)
    return trainSetSortedByHour
        
    '''
        tmpMax = trainSet.max(1)
        tmpAvg = trainSet.mean(1)
        tmpMedian = trainSet.Median(1)
        tmpMode = trainSet.mode(1)
        
        
        trainSet.to_csv(dir+str(NumofDays)+'avgHour.csv',encoding='utf-8',index = False)
    ''' 
    


def getEverydayHour(testSet):
    # 11
    #dateIndex=trainSet['create_date'].unique()
    dir= './getAvgHourFeature/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir)
    
    for i in range (38):
        print (i)
        trainSet= pd.read_csv('./SplitedData/'+str(i)+'.csv')
        result = []
        for j in range(len(testSet)):
            if j%500 ==0 :
                print ('j',j)
            p=[]
            for k in range (24):
                 x = testSet.iloc[j]
                 startId = x['start_geo_id']
                 endId = x['end_geo_id']
                 tmp = trainSet[(trainSet['start_geo_id']==startId) & (trainSet['end_geo_id']==endId) \
                     & (trainSet['create_hour']== k)].count()[0]
                 p.append(tmp)
            result.append(p)
        pd.DataFrame(result).to_csv(dir+str(i)+'.csv',encoding='utf-8',index = False)
        
    
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
    
    print (raweather.iloc[i][[2,4,7,11]])
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
        #tmp.extend(raweather.iloc[i][[2,3,4,5,6,7,9,10,11]].tolist())
        tmp.extend(raweather.iloc[i][[2,4,7,11]].tolist())
        
        result[str(key[0])+'-'+raweather.iloc[i][12]]=tmp
        
        if flag:
            i=i+1
        i=i+1
    

    #print (result)
   
  

    return result

# 10 features

def getDateFeature(): 
    #14
    #dateIndex=pd.Series(pd.read_csv(weatherPath)['date'].str.split(' ',expand=True)[0].unique())
    #print (dateIndex)
    NumofDays = 38
    #是否星期几 是否是工作日  是否是节假日  是否节前 是否节后
    weekFeature =[
                [0,0,0,0,0,1,0 ,0,1,0,0], # 星期六
                [0,0,0,0,0,0,1 ,0,1,0,0],
                [1,0,0,0,0,0,0 ,1,0,0,1],
                [0,1,0,0,0,0,0 ,1,0,0,0],
                [0,0,1,0,0,0,0 ,1,0,0,0],
                [0,0,0,1,0,0,0 ,1,0,0,0],
                [0,0,0,0,1,0,0 ,1,0,1,0]
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
    #trainSet = UC.ReadTrain()
    #testSet = UC.ReadTest()
    #DataSplit(trainSet)
    #getSplitedTrainData(trainSet,testSet)
    #getAverage(31,31)
    #getAverageWithnormal(31,31,1)
    #getSplitedWeather(trainSet,testSet)
    #analysis(trainSet,testSet)
    #getPoiFeature()
    #getPoiFeature()
    getWeatherFeature()
    #getHourFeature(31)
    #getAvgHourFeature(31)
    #getAvgHourFeature(31)
    #getEverydayHour(testSet)
    
    
    
    