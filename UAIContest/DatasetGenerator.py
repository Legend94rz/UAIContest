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

finished = 0

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
    global finished
    finished = finished+1
    if finished % 100 == 0:
        print("%s, gened train %d\n" % (dt.datetime.now(),finished))

def GenTrainingSet():
    try:
        dic = pickle.load(open('train.pkl','rb'))
        return dic['X'],dic['Y']
    except IOError:
        pass
    global finished
    print("Gening Training set...\n")
    pool = Pool(cpu_count()-1)
    finished = 0
    result = []
    for i in range(len(testset)):
        result.append( pool.apply_async(WorkerForTrain, tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']]), callback=CbkForTrain) )
    pool.close()
    pool.join()
    X = [result[i].get()[0] for i in range(len(result))]
    Y = [result[i].get()[1] for i in range(len(result))]
    pickle.dump({'X':X,'Y':Y},open('train.pkl','wb'))
    return X,Y
