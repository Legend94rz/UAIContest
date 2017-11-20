import pandas as pd
import datetime as dt
import numpy as np
import pickle
from multiprocessing.pool import Pool

train_july = '.\\Data\\train_July.csv'
train_aug = '.\\Data\\train_Aug.csv'
testFile = '.\\Data\\test_id_Aug_agg_public5k.csv'

julyset = pd.read_csv(train_july)
augset = pd.read_csv(train_aug)
testset = pd.read_csv(testFile)
trainset = pd.concat([julyset, augset])

X=[]
Y=[]
TX = []

def GetHistoryMean(trainSet):
    tmp = trainSet.groupby(['create_date','create_hour']).size().reset_index(name='count')
    return tmp[tmp['count']<80].groupby('create_hour').mean().reset_index()

def GetFeature(d, hur, tmpset):
    feature = []
    Y = []
    tmp = tmpset[tmpset['create_hour']==hur].groupby('create_date').size().reset_index(name='count')
    for i in range(len(tmp)):
        feature.append((dt.datetime.strptime( tmp.iloc[i,0],'%Y-%m-%d') - dt.datetime(2017,7,1)).days)
        Y.append(tmp.iloc[i,1])
    return feature,Y

def WorkerForTrain(*x):
    """
    x - [start_geo_id, end_geo_id, create_hour]
    """
    tmpset = trainset[(trainset['start_geo_id']==x[0]) & (trainset['end_geo_id']==x[1])]
    feature, y = GetFeature('notused', x[2], tmpset)
    return (feature,y)

def CbkForTrain(result):
    X.append(result[0])
    Y.append(result[1])
    if len(Y)%100==0:
        print("%s, gened train %d\n"%(dt.datetime.now(),len(Y)))

def GenTrainingSet():
    try:
        dic = pickle.load(open('train.pkl','rb'))
        return dic['X'],dic['Y']
    except IOError:
        pass
    print("Gening Training set...\n")
    pool = Pool()
    for i in range(len(testset)):
        pool.apply_async(WorkerForTrain,tuple(testset.loc[i,['start_geo_id','end_geo_id','create_hour']]),callback=CbkForTrain)

    pool.close()
    pool.join()
    pickle.dump({'X':X,'Y':Y},open('train.pkl','wb'))
    return X,Y

def WorkerForTest(*x):
    """
    x - ['start_geo_id','end_geo_id','create_date','create_hour']
    """
    tmpset = trainset[(trainset['start_geo_id']==x[0]) & (trainset['end_geo_id']==x[1])]
    feature,y = GetFeature(x[2],x[3],tmpset)
    return feature

def CbkForTest(result):
    TX.append(result)
    if len(TX)%100==0:
        print("%s, gened test %d\n"%(dt.datetime.now(),len(TX)))

def GenTestSet():
    try:
        tdic = pickle.load(open('test.pkl','rb'))
        return tdic['X']
    except IOError:
        pass
    print("Gening Test set...\n")
    pool = Pool()
    for i in range(len(testset)):
        pool.apply_async(WorkerForTest,tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']]),callback=CbkForTest)
    pool.close()
    pool.join()
    pickle.dump({'X':TX},open('test.pkl','wb'))
    return TX

