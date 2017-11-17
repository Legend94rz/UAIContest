import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import numpy as np


#plot A->B orders count by time
trainPath1 = '.\\Data\\train_July.csv'
trainPath2 = '.\\Data\\train_Aug.csv'
testPath = '.\\Data\\test_id_Aug_agg_public5k.csv'
SplitedTrainData = './SplitedTrainData/everyday.csv'
weatherPath ='./Data/weather.csv'

meanPath = './xgbmodel/2.08.csv'
modelPath = './xgbmodel/prediction.csv'

averagePath = './Average/31_31.csv'


def ResultAnalysis():
    everyday = pd.read_csv(meanPath)
    print (everyday.describe())
    
    everyday = pd.read_csv(averagePath)
    print (everyday.describe())



def getMean():
    
    dir= './getMean/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    meanResult = pd.read_csv(meanPath)
    modelResult = pd.read_csv(modelPath)
    
    result = pd.DataFrame()
    result['test_id'] = meanResult['test_id']
    result['count'] = ((meanResult['count']+modelResult['count'])/2).apply(np.round)
    result.to_csv(dir+'prediction.csv',encoding='utf-8',index = False)

def ReadTrain():
    train1 = pd.read_csv(trainPath1)
    train2 = pd.read_csv(trainPath2)
    return pd.concat([train1,train2])

def ReadTest():
    return pd.read_csv(testPath)

def DataSplit(trainSet):
    dateIndex=trainSet['create_date'].unique()
    dir= './SplitedData/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir)
    for i in range(len(dateIndex)):
        tmp=trainSet[(trainSet['create_date']==dateIndex[i])]
        tmp.to_csv(dir+str(i)+'.csv',encoding='utf-8',index = False)

def getSplitedTrainData(trainSet,testSet):
    dateIndex=trainSet['create_date'].unique()
    dir= './SplitedTrainData/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir)
    
    result = pd.DataFrame()
    result['test_id'] = testSet['test_id']
    
    for i in range(len(dateIndex)):
        print(i)
        tmp=trainSet[(trainSet['create_date']==dateIndex[i])]
        p=[]
        for j in range(len(testSet)):
            x = testSet.iloc[j]
            startId = x['start_geo_id']
            endId = x['end_geo_id']
            hour = x['create_hour']
            tmp2 = tmp[(tmp['start_geo_id']==startId) & (tmp['end_geo_id']==endId) \
                        & (tmp['create_hour']==hour)].count()[0]
            p.append(tmp2)
        result[dateIndex[i]]=p
    result.to_csv(dir+'.csv',encoding='utf-8',index = False)



def getSplitedWeather(trainSet,testSet):
    '''
    dateIndex=pd.Series(trainSet['create_date'].unique())
    year=dateIndex.str.split('-',expand=True)[0].apply(lambda x:int(x))
    month=dateIndex.str.split('-',expand=True)[1].apply(lambda x:int(x))
    day=dateIndex.str.split('-',expand=True)[2].apply(lambda x:int(x))
    NewdateIndex=year.apply(lambda x:str(x))+'-'+month.apply(lambda x:str(x))+'-'+day.apply(lambda x:str(x))+' '
    print(NewdateIndex)
    '''

    
    raweather=pd.read_csv(weatherPath)
    dateIndex=pd.Series(raweather['date'].str.split(' ',expand=True)[0].unique())
    
    dir= './SplitedTrainData/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir)
    result = pd.DataFrame()
    
    weather = pd.DataFrame()
    weather['date']=raweather['date'].str.split(' ',expand=True)[0]
    weather['hour']=raweather['date'].str.split(' ',expand=True)[1].str.split(':',expand=True)[0]
    weather['code']=raweather['code']
   
    print(weather[35:45])
    
    
    result['test_id'] = testSet['test_id']
    for i in range(len(dateIndex)):
        print(i)
        p=[]
        for j in range(len(testSet)):
            x = testSet.iloc[j]
            hour = x['create_hour']
            tmp2 = weather.loc[(weather['date']==dateIndex[i]) & (weather['hour']==str(hour)),'code']
            #print (dateIndex[i],hour,tmp2.iloc[0])
            p.append(tmp2.iloc[0])
        result[dateIndex[i]]=p
    result.to_csv(dir+'Splitedweather.csv',encoding='utf-8',index = False)
    
        
def getAverage(NumofDays,LastDay):
    train=pd.read_csv(SplitedTrainData)
    
    result = pd.DataFrame()
    result['test_id'] = train['test_id']
    
    tmpstd=(train.iloc[:,LastDay+1-NumofDays:LastDay+1]).std(1)
    tmpmean=(train.iloc[:,LastDay+1-NumofDays:LastDay+1]).mean(1)
    tmpmedian=(train.iloc[:,LastDay+1-NumofDays:LastDay+1]).median(1)
    
    '''
    tmp=tmpmean
    tmp[tmpstd>1.5]=tmpmedian[tmpstd>1.5]
    
    print (tmpstd>1.5)
    print (tmpmean[:15])
    print (tmpmedian[:15])
    print (tmp[:15])
    
    print (tmp[tmpstd>1.5])
   '''
    print (tmpstd[:10])
    
    dir= './Average/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir)
    result['count']=tmpstd
    #result['count']=tmp.apply(np.round)
    result.to_csv(dir+str(NumofDays)+'_'+str(LastDay)+'.csv',encoding='utf-8',index = False)
    
    return tmpstd.tolist(),tmpmean.tolist()
    
def getAverageWithnormal(NumofDays,LastDay,BadDays):
    # BadDays*2 < NumofDays      LasyDay+1>NumofDays
    train=pd.read_csv(SplitedTrainData)
    
    result = pd.DataFrame()
    result['test_id'] = train['test_id']
    tmp=train.iloc[:,LastDay+1-NumofDays:LastDay+1]
    p=[]
    for i in range(len(tmp)):
        x = tmp.iloc[i]
        m=x.sort_values()[BadDays-1:-BadDays].mean()   
        p.append(m)
    dir= './Average/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir)
    result['count']=p
    result.to_csv(dir+str(NumofDays)+'_'+str(LastDay)+'.csv',encoding='utf-8',index = False)        

def analysis(trainSet,testSet):
    result = pd.DataFrame()
    result['test_id'] = testSet['test_id']
    current_date=['2017-08-01','2017-08-02','2017-08-03','2017-08-04','2017-08-05','2017-08-06','2017-08-07']
    past_date=['2017-07-25','2017-07-26','2017-07-27','2017-07-28','2017-07-29','2017-07-30','2017-07-31']
    ser = pd.Series(past_date,index = current_date)
    
    print (ser)
    
    p=[]
    
    for i in range(len(testSet)):
        if i%500==0:
             print('Current at %d' %(i))
        x = testSet.iloc[i]
        startId = x['start_geo_id']
        endId = x['end_geo_id']
        date = x['create_date']
        hour = x['create_hour']
        #dt.datetime.strptime(str,'%Y-%m-%d')
        #x.replace(hour=9)
        tmp = trainSet[(trainSet['start_geo_id']==startId) & (trainSet['end_geo_id']==endId) \
                       & (trainSet['create_hour']==hour)& (trainSet['create_date']==ser[date])].count()[0]
      
        print (tmp)
        
     
        '''
        s = 0
        if(hour>0):
            s = s + tmp[tmp['create_hour']==hour-1].shape[0]
        if(hour<23):
            s = s + tmp[tmp['create_hour']==hour+1].shape[0]
        if(hour>0 and hour<23):
            s = s/2.0;
        '''
        p.append(tmp)
    result['count']=p
    result.to_csv('prediction.csv',encoding='utf-8',index = False)

if __name__=="__main__":
    #trainSet = ReadTrain()
    #testSet = ReadTest()
    # DataSplit(trainSet)
    #getSplitedTrainData(trainSet,testSet)
    getAverage(31,31)
    #getAverageWithnormal(31,31,1)
    #getMean()
    ResultAnalysis()
    #getSplitedWeather(trainSet,testSet)
    #analysis(trainSet,testSet)