import pandas as pd
import datetime as dt
import numpy as np
import pickle
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

train_july = '.\\Data\\train_July.csv'
train_aug = '.\\Data\\train_Aug.csv'
testFile = '.\\Data\\test_id_Aug_agg_public5k.csv'

julyset = pd.read_csv(train_july)
augset = pd.read_csv(train_aug)
testset = pd.read_csv(testFile)
trainset = pd.concat([julyset, augset])

X = []
Y = []
TX = []

def WorkerForTrain(*x):
    """
    x - [start_geo_id, end_geo_id,create_date, create_hour]
    """
    feature = []
    y=[]
    time = dt.datetime.strptime(x[2],'%Y-%m-%d')+dt.timedelta(hours=int(x[3]))
    timeL = time+dt.timedelta(hours=-1)
    timeR = time+dt.timedelta(hours=1)
    for k in range(3):
        tmpset = trainset[(trainset['start_geo_id']==x[0]) &( trainset['end_geo_id']==x[1]) &  (trainset['status']==k)]
        feature.append(len(tmpset[ (tmpset['create_hour']==timeL.hour) & (tmpset['create_date']==timeL.strftime('%Y-%m-%d'))]))
        feature.append(len(tmpset[ (tmpset['create_hour']==timeR.hour)& (tmpset['create_date']==timeR.strftime('%Y-%m-%d'))]))
        tmp = tmpset[tmpset['create_hour']==x[3]].groupby('create_date').size().reset_index(name='count')
        tmp = tmp[tmp['count']<=20]
        if len(tmp)>0:
            feature.append(tmp[tmp['count']<=20]['count'].mean())
        else:
            feature.append(0)
    return (feature,Y)
def dummy():
    WorkerForTrain('c538ad66d710f99ad0ce951152da36a4','90bb1d035e403538d20b073aec57bea2','2017-08-01',21)
    pass
def CbkForTrain(result):
    X.append(result[0])
    Y.append(result[1])
    if len(Y) % 100 == 0:
        print("%s, gened train %d\n" % (dt.datetime.now(),len(Y)))

def GenTrainingSet():
    try:
        dic = pickle.load(open('train.pkl','rb'))
        return dic['X'],dic['Y']
    except IOError:
        pass
    print("Gening Training set...\n")
    pool = Pool(cpu_count()-1)
    for i in range(len(testset)):
        pool.apply_async(WorkerForTrain, tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']]), callback=CbkForTrain)

    pool.close()
    pool.join()
    pickle.dump({'X':X,'Y':Y},open('train.pkl','wb'))
    return X,Y

def WorkerForTest(*x):
    """
    x - ['start_geo_id','end_geo_id','create_date','create_hour']
    """
    feature = [(dt.datetime.strptime(x[2],'%Y-%m-%d') - dt.datetime(2017,7,1)).days*24 + x[3]]
    return feature

def CbkForTest(result):
    TX.append(result)
    if len(TX) % 100 == 0:
        print("%s, gened test %d\n" % (dt.datetime.now(),len(TX)))

def GenTestSet():
    try:
        tdic = pickle.load(open('test.pkl','rb'))
        return tdic['X']
    except IOError:
        pass
    print("Gening Test set...\n")
    pool = Pool(cpu_count()-1)
    for i in range(len(testset)):
        pool.apply_async(WorkerForTest,tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']]),callback=CbkForTest)
    pool.close()
    pool.join()
    pickle.dump({'X':TX},open('test.pkl','wb'))
    return TX

