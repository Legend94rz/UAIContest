# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:04:29 2017

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:15:51 2017

@author: Dell
"""

import collections
import numpy as np
from datetime import datetime
import copy
import xgboost as xgb
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as gbr


def getDateFeature(): 
    #14
    #dateIndex=pd.Series(pd.read_csv(weatherPath)['date'].str.split(' ',expand=True)[0].unique())
    #print (dateIndex)
    NumofDays = 38
    #是否星期几 是否是工作日  是否是节假日  是否节前 是否节后
    
    weekFeature =[
                [0,1,0,0], # 星期六
                [0,1,0,0],
                [1,0,0,1],
                [1,0,0,0],
                [1,0,0,0],
                [1,0,0,0],
                [1,0,1,0]
             ]
    '''
    weekFeature =[
                [6,1], # 星期六
                [7,1],
                [1,0],
                [2,0],
                [3,0],
                [4,0],
                [5,0]
             ]
    '''
    #是否月初 1-3，是否月中14-16，是否月末 29-31
    result = {}
    for i in range(NumofDays):
        tmp =[]
        '''
        tmp2=[0,0,0]
        if i in range(1,3):
            tmp2[0]=1
        if i in range(14,16):
            tmp2[1]=1
        if i in range(29,31):
            tmp2[2]=1
        '''
        tmp.extend(weekFeature[i%7])
        #tmp.extend(tmp2)
        
        result[i]=tmp
    return result
        
def get_train_july_list():
    with open('./Data/train_July.csv') as f:
        a = f.readlines()
        Order=collections.namedtuple('order', ['key','date','day','hour','user','status'])
        train_list = []
        start_date = datetime.strptime('2017-7-1', '%Y-%m-%d')
        for x in a[1:]:
            x = x.strip().split(',')
            x.insert(4,datetime.strptime(x[3], '%Y-%m-%d').weekday())
            x[3] = (datetime.strptime(x[3], '%Y-%m-%d') - start_date).days
            x[5] = int(x[5])
            x[6] = int(x[6])
            order = Order._make([x[-2]+x[-1],x[3],x[4],x[5],x[2],x[6]])
            train_list += [order]
    return train_list

def get_train_aug_list():
    with open('./Data/train_Aug.csv') as f:
        a = f.readlines()
        Order=collections.namedtuple('order', ['key','date','day','hour','user','status'])
        train_list = []
        start_date = datetime.strptime('2017-7-1', '%Y-%m-%d')
        for x in a[1:]:
            x = x.strip().split(',')
            x.insert(4,datetime.strptime(x[3], '%Y-%m-%d').weekday())
            x[3] = (datetime.strptime(x[3], '%Y-%m-%d') - start_date).days
            x[5] = int(x[5])
            x[6] = int(x[6])
            order = Order._make([x[-2]+x[-1],x[3],x[4],x[5],x[2],x[6]])
            train_list += [order]
    return train_list

def get_test_list():
    with open('./Data/test_id_Aug_agg_public5k.csv') as f:
        a = f.readlines()
        names = ['key','day','hour','date','count']
        Order=collections.namedtuple('order', names)
        test_list = []
        start_date = datetime.strptime('2017-7-1', '%Y-%m-%d')
        for x in a[1:]:
            x = x.strip().split(',')
            order = Order._make([x[1]+x[2],datetime.strptime(x[3], '%Y-%m-%d').weekday(),int(x[4]),(datetime.strptime(x[3], '%Y-%m-%d') - start_date).days,0])
            test_list += [order]
    return test_list


def parse_list(order_list,mode=0):
    data_dict = dict()
    data_set_dict = dict()
    for x in order_list:
        key = x.key
        if key not in data_dict:
            data_dict[key] = [0.0]*24*38
            data_dict[key][x.date*24 + x.hour] = 1
        else:
            data_dict[key][x.date*24 + x.hour] += 1
    Sample=collections.namedtuple('sample', ['key','date','day','hour','count'])
    if mode == 2:
        return data_dict
    if mode == 1:
        for key in data_dict:
            for x in range(38*24):
                if(data_dict[key][x] != 0):
                    sample = Sample._make([key,x//24,(x//24-2)%7,x%24,int(data_dict[key][x])])
                    if key not in data_set_dict:
                        data_set_dict[key] = [sample]
                    else:
                        data_set_dict[key] += [sample]
        return data_set_dict
    data_set = []
    for key in data_dict:
        for x in range(38*24):
            if(data_dict[key][x] != 0):
                sample = Sample._make([key,x//24,(x//24-2)%7,x%24,int(data_dict[key][x])])
                data_set += [sample]
    return data_set


def calc_mean(data_set,full):
    hour_mean_dict = dict()
    mean_dict = dict()
    if full:
        day_num = [0.0]*7
        for x in range(37):
            day_num[(x-2)%7] += 1
        days = 34.0
    else:
        day_num = [0.0]*7
        for x in range(31):
            day_num[(x-2)%7] += 1
        days = 31.0
    for x in data_set:
        res = 0 if not full else (x.hour+1)%2
        if x.key not in hour_mean_dict:
            hour_mean_dict[x.key] = [0.0]*24
            hour_mean_dict[x.key][x.hour] += x.count/(days+res) 
        else:
            hour_mean_dict[x.key][x.hour] += x.count/(days+res)
        if x.key not in mean_dict:
            mean_dict[x.key] = [0.0]*24*7
            mean_dict[x.key][x.day*24+x.hour] += x[3]/day_num[x.day]
        else:
            mean_dict[x.key][x.day*24+x.hour] += x[3]/day_num[x.day]
    
    return mean_dict,hour_mean_dict
def get_mean_value(x, mean_dict, hour_mean_dict,mode=0):
    day = x.day
    key = x.key
    hour = x.hour
    if key not in hour_mean_dict:
        return 0
    if mode == 1:
        ret = mean_dict[key][day*24+hour]*0.6 + 0.2*(mean_dict[key][max((day*24+hour-1),0)]+mean_dict[key][min(day*24+hour+1,24*7-1)])
    else:
        ret = hour_mean_dict[key][hour]*0.6 + 0.2*(hour_mean_dict[key][(hour-1)%24]+hour_mean_dict[key][(hour+1)%24])
    return ret

def get_mean_hour(x, mean_dict, hour_mean_dict,mode=0):
    day = x.day
    key = x.key
    hour = x.hour
    if key not in hour_mean_dict:
        return 0
    if mode == 1:
        ret = mean_dict[key][day*24+hour]*0.6 + 0.2*(mean_dict[key][max((day*24+hour-1),0)]+mean_dict[key][min(day*24+hour+1,24*7-1)])
    else:
        ret = hour_mean_dict[key][hour]
    return ret

def get_mean_hourlater(x, mean_dict, hour_mean_dict,mode=0):
    day = x.day
    key = x.key
    hour = x.hour
    if key not in hour_mean_dict:
        return 0
    if mode == 1:
        ret = mean_dict[key][day*24+hour]*0.6 + 0.2*(mean_dict[key][max((day*24+hour-1),0)]+mean_dict[key][min(day*24+hour+1,24*7-1)])
    else:
        ret = mean_dict[key][min(day*24+hour+1,24*7-1)]
    return ret

def get_mean_hourearly(x, mean_dict, hour_mean_dict,mode=0):
    day = x.day
    key = x.key
    hour = x.hour
    if key not in hour_mean_dict:
        return 0
    if mode == 1:
        ret = mean_dict[key][day*24+hour]*0.6 + 0.2*(mean_dict[key][max((day*24+hour-1),0)]+mean_dict[key][min(day*24+hour+1,24*7-1)])
    else:
        ret =mean_dict[key][max((day*24+hour-1),0)]
    return ret

def get_mean_preds(data_set, mean_dict, hour_mean_dict,mode=0):
    ct = 0
    preds = []
    avghour = []
    avghourlater=[]
    avghourearly=[]
    
    for x in data_set:
        ct += 1
        preds += [get_mean_value(x, mean_dict, hour_mean_dict, mode)]
        avghour += [get_mean_hour(x, mean_dict, hour_mean_dict, mode)]
        avghourlater += [get_mean_hourlater(x, mean_dict, hour_mean_dict, mode)]
        avghourearly += [get_mean_hourearly(x, mean_dict, hour_mean_dict, mode)]
    return np.ceil(preds),avghour,avghourlater,avghourearly


def eval_preds(data_set,preds):
    gt = np.array([x.count for x in data_set])
    loss = np.abs(gt-np.array(preds)).sum()
    return loss


def get_poi_dict():
    with open('./Data/poi.csv') as f:
        b = f.readlines()
        poi_dict = dict()
        for x in b[1:]:
            x = x.strip().split(',')
            poi_dict[x[0]] = int(x[2])+int(x[4])+int(x[6])+int(x[8])+int(x[10])+int(x[12])+int(x[14])+int(x[16])+int(x[18])+int(x[20])
    return poi_dict

def train_set_generator(data_set, preds,preds2,avghour,avghourlater,avghourearly,poi_dict):
    dateFeature= getDateFeature() # 0-37
    res_train_set = copy.deepcopy(data_set)
    Y = [0]*len(res_train_set)
    for x in range(len(res_train_set)):
        Y[x] = res_train_set[x].count - preds[x]
    XX = []
    for x in range(len(res_train_set)):
        start_id = res_train_set[x].key[:len(res_train_set[x].key)//2]
        stop_id = res_train_set[x].key[len(res_train_set[x].key)//2:]
        
        if start_id not in poi_dict:
            start_poi = 0
        else:
            start_poi = poi_dict[start_id]
        if stop_id not in poi_dict:
            stop_poi = 0
        else:
            stop_poi = poi_dict[stop_id]
        stop_id = res_train_set[x][0][len(res_train_set[x][0])//2:]
        #XX += [[res_train_set[x].date,res_train_set[x].day,res_train_set[x].hour,preds2[x],avghour[x],avghourlater[x],avghourearly[x],start_poi,stop_poi] \
        #     + dateFeature[res_train_set[x].date]  \
        #    ]# \
        XX.append( [res_train_set[x].date,res_train_set[x].day,res_train_set[x].hour,preds2[x],avghour[x],start_poi,stop_poi]  )

    X = np.array(XX)
    return X,np.array(Y)





def MyRegression(xTrain,yTrain,xVal,yVal,xTest,learner='gdrt'):
    
    if (learner == 'gdrt'):
        
        clf = gbr(loss='lad',
            n_estimators=300, max_depth=300,
            learning_rate=0.1, min_samples_leaf=256,
            min_samples_split=256)
        '''
        clf = gbr(loss='lad')
        '''
        clf.fit(xTrain,yTrain)
        
        #val_preds = get_mean_preds(val_set, mean_dict, hour_mean_dict,mode=0)
        #X_val,Y_val = train_set_generator(val_set,val_preds, weather_dict, poi_dict)
        res_preds = clf.predict(xVal)
        test_res_preds = clf.predict(xTest)
    
    if (learner == 'xgboost'):
        param = {
            'booster':'gbtree',  #default=gbtree
            'objective':'reg:linear',
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
        num_round=500
        bst,res_preds =  Myxgb(xTrain,yTrain,xVal,param,num_round)
        dtest  = xgb.DMatrix(xTest)   
        test_res_preds=bst.predict(dtest)

    return res_preds,test_res_preds
    
    
def Myxgb(xTrain,yTrain,test,param,num_round):
     
     dtrain = xgb.DMatrix(xTrain, label=yTrain)
     dtest  = xgb.DMatrix(test)   
     bst = xgb.train( param, dtrain, num_round)
     preds=bst.predict(dtest)
     
     return bst,preds





if __name__=="__main__":
    
    '''
    print (len(get_weather_list()))
    print (len(UF.getWeatherFeature())) 
    print ((get_train_july_list()[:4]))
    '''
    
    
    train_list = get_train_july_list()
    train_set = parse_list(train_list)
    val_list = get_train_aug_list()
    val_set = parse_list(val_list)
    full_list = train_list + val_list
    full_set = train_set + val_set
    poi_dict = get_poi_dict()
    
    
   
    learner = 'gdrt'

    
    ####################      full
    mean_dict,hour_mean_dict = calc_mean(full_set,True)
    
    esemavghour,avghour,avghourlater,avghourearly = get_mean_preds(full_set, mean_dict, hour_mean_dict,mode=0)
    #preds= np.array([0]*len(esemavghour),dtype = np.float64)
    preds= esemavghour
    X,Y = train_set_generator(full_set,preds,esemavghour,avghour,avghourlater,avghourearly, poi_dict)
    
    esemavghour,avghour,avghourlater,avghourearly = get_mean_preds(val_set, mean_dict, hour_mean_dict,mode=0)
    #val_preds = np.array([0]*len(esemavghour),dtype = np.float64)
    val_preds= esemavghour
    
    X_val,Y_val = train_set_generator(val_set,val_preds,esemavghour,avghour,avghourlater,avghourearly, poi_dict)
        
    
    
    test_set = get_test_list()
    esemavghour,avghour,avghourlater,avghourearly = get_mean_preds(test_set, mean_dict, hour_mean_dict,mode=0)
    #test_preds = np.array([0]*len(esemavghour),dtype = np.float64)
    test_preds= esemavghour
    X_test,_ = train_set_generator(test_set,test_preds,esemavghour,avghour,avghourlater,avghourearly,poi_dict)
    

    
    res_preds,test_res_preds = MyRegression(X,Y,X_val,Y_val,X_test,learner=learner)
    
    val_preds += res_preds
    test_preds += test_res_preds
    
    print (eval_preds(val_set,val_preds)/len(val_set),len(val_set))
    
    
    
    
    res = pd.read_csv('w.csv')*0.4
    with open('outnores.csv','w') as f:
        f.write('test_id,count\n')
        i = 0
        for x in test_preds:
            f.write("%d,%d\n"%(i,np.ceil(test_preds[i]*0.6+res.iloc[i]['count'])))
            i+=1
    
    