# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:39:57 2017

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

import UAIGetFeature as UF
import UAIContest as UC

import xgboost as xgb
import numpy as np

SplitedTrainData = './SplitedTrainData/everyday.csv'


def Myxgb(train,y_label,validate,validate_label,test):
     
     dtrain = xgb.DMatrix(train, label=y_label)
     dvalidate = xgb.DMatrix(validate, label=validate_label)
     dtest  = xgb.DMatrix(test)
     param = {
            'booster':'gbtree',  #default=gbtree
            #'objective':'binary:logistic', # default=reg:linear
            'eta':0.3,  #default=0.3, alias: learning_rate
            'max_depth':6, #[default=6
            'silent':1, #default=0
            'subsample':1, #default=1
            'max_delta_step':0, #default=0
            'scale_pos_weight':1,#[default=1
            'gamma':0, #default=0           
            'lambda':1, #default=1
           # 'eval_metric':"logloss",
            'min_child_weight':1,
            'nthread':2
            }
     num_round =100
     num_boost =100 #for cv
     num_fold = 5
     evallist  = [(dtrain,'train'),(dvalidate,'validate')] 
   #  print xgb.cv(param,dtrain,num_boost,num_fold)
     bst = xgb.train( param, dtrain, num_round, evallist )
     
#     gbm = xgb.XGBRegressor().fit(data.ix[:,25:], data['y'])  
     #predictions = gbm.predict(data.ix[:,25:])  
    # bst.save_model('./model/xgb.model') # 用于存储训练出的模型
     preds=bst.predict(dtest)
     #ptrain=bst.predict(dtrain)
     #print logloss(y_label,ptrain)
     return bst,preds
 
def generateTest(trainSet,testSet):

    label=pd.read_csv(SplitedTrainData)
    poiFeature=UF.getPoiFeature()   # id
    weatherFeature=UF.getWeatherFeature() # 0-37 + 0-23
    dateFeature=UF.getDateFeature() # 0-37
    avg=pd.read_csv('./31.csv')
    
    dateIndex=testSet['create_date'].unique()
    
    dir= './TrainDataForModel/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir)
      
    #for i in range(0):
    p=[]
    for j in range(len(testSet)):
        sample=[]
        x = testSet.iloc[j]
        startId = x['start_geo_id']
        endId = x['end_geo_id']
        hour = x['create_hour']
        date = x['create_date']
        
        i=list(dateIndex).index(date)+31
        dateHour = pd.read_csv('./getAvgHourFeature/'+str(i)+'.csv')
        
        
        
        #print (i)
        
        tmpLabel = label.iloc[j,i+1]
        
        '''
        if startId in poiFeature.keys():
            sample.append(1)
            sample.extend(poiFeature[startId])
        else:
            tmpPos=[x*0 for x in range(12)]
            sample.extend(tmpPos)
        
        if endId in poiFeature.keys():
            sample.append(1)
            sample.extend(poiFeature[endId])
        else:
            tmpPos=[x*0 for x in range(12)]
            sample.extend(tmpPos)
        '''    
        s = [0,0]
        if hour>0 :
            s[1] = dateHour.iloc[j,hour-1]
        if hour<23:
            s[0] = dateHour.iloc[j,hour+1]
            
        sample.extend(s)
        sample.extend(avg.iloc[j].tolist())
        sample.extend(dateFeature[i])
        sample.extend(weatherFeature[str(i)+'-'+str(hour)])
        sample.append((tmpLabel))
        #print (sample)
        p.append(sample)
    result = pd.DataFrame(p)
    result.to_csv(dir+'Test'+'.csv',encoding='utf-8',index = False)

   
    
def generateTrain(trainSet,testSet):

    label=pd.read_csv(SplitedTrainData)
    poiFeature=UF.getPoiFeature()   # id
    weatherFeature=UF.getWeatherFeature() # 0-37 + 0-23
    dateFeature=UF.getDateFeature() # 0-37
    avg=pd.read_csv('./31.csv')
    dateIndex=trainSet['create_date'].unique()
    dir= './TrainDataForModel/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir)
      
    for i in range(len(dateIndex)):
        print(i)
        p=[]
        for j in range(len(testSet)):
            sample=[]
            x = testSet.iloc[j]
            dateHour = pd.read_csv('./getAvgHourFeature/'+str(i)+'.csv')
            startId = x['start_geo_id']
            endId = x['end_geo_id']
            hour = x['create_hour']
            tmpLabel = label.iloc[j,i+1]
            
            '''
            if startId in poiFeature.keys():
                sample.append(1)
                sample.extend(poiFeature[startId])
            else:
                print (i,j)
                tmpPos=[x*0 for x in range(12)]
                sample.extend(tmpPos)
            
            if endId in poiFeature.keys():
                sample.append(1)
                sample.extend(poiFeature[endId])
            else:
                print (i,j)
                tmpPos=[x*0 for x in range(12)]
                sample.extend(tmpPos)
            ''' 
            s = [0,0]
            if hour>0 :
                s[1] = dateHour.iloc[j,hour-1]
            if hour<23:
                s[0] = dateHour.iloc[j,hour+1]
            
            sample.extend(s)
            sample.extend(avg.iloc[j].tolist())
            sample.extend(dateFeature[i])
            sample.extend(weatherFeature[str(i)+'-'+str(hour)])
            sample.append((tmpLabel))
            #print (sample)
            p.append(sample)
        result = pd.DataFrame(p)
        result.to_csv(dir+str(i)+'.csv',encoding='utf-8',index = False)
        #47    
def getTrain(startDate,endDate):
    dir= './TrainDataForModel/'
    for i in range(startDate,endDate+1):
        #print(i)
        if i == startDate:
            Train=pd.read_csv(dir+str(i)+'.csv').values
        else:
            tmp=pd.read_csv(dir+str(i)+'.csv').values
            Train=np.concatenate((Train,tmp))
    
    print (len(Train)) 
    
    return Train
    
def getTest():
    dir= './TrainDataForModel/'
    Test=pd.read_csv(dir+'Test.csv').values

    return Test


if __name__=="__main__":
    
    
    trainSet = UC.ReadTrain()
    dfTest = UC.ReadTest()
    
    
    generateTrain(trainSet,dfTest)
    generateTest(trainSet,dfTest)
    
   
    
    Train= getTrain(0,30)
    Validate = getTrain(21,30)
    Test = getTest()

    bst,result=Myxgb(Train[:,:-1],Train[:,-1],Validate[:,:-1],Validate[:,-1],Test[:,:-1])
    
    result = np.maximum(result,0)
    #result = np.around(result)
    
    df = pd.DataFrame()
    df['test_id'] = dfTest["test_id"].values.tolist()
    df['count'] = result.tolist()
    df.to_csv('./xgbmodel/prediction.csv',encoding='utf-8',index = False)
    
    #featureEvalution(bst,feats)
  