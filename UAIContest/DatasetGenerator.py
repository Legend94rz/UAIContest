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
    x - [start_geo_id, end_geo_id, create_hour]
    """
    tmpset = trainset[(trainset['start_geo_id'] == x[0]) & (trainset['end_geo_id'] == x[1])].groupby(['create_date','create_hour']).size().reset_index(name='count')
    feature = [i for i in range(24*31)]
    Y = [0 for i in range(24*31)]
    for i in range(len(tmpset)):
        days = (dt.datetime.strptime( tmpset.loc[i,'create_date'] ,'%Y-%m-%d' )-dt.datetime(2017,7,1)).days
        if days<31:
            Y[ days*24 + tmpset.loc[i,'create_hour'] ] = tmpset.loc[i,'count']
        else:
            feature.append( days*24 + tmpset.loc[i,'create_hour'] )
            Y.append(tmpset.loc[i,'count'])
    return (feature,Y)
def dummy():
    #WorkerForTrain('c538ad66d710f99ad0ce951152da36a4','90bb1d035e403538d20b073aec57bea2',21)
    pass
def CbkForTrain(result):
    finished = finished+1
    if finished % 100 == 0:
        print("%s, gened train %d\n" % (dt.datetime.now(),finished))

def GenTrainingSet():
    try:
        dic = pickle.load(open('train.pkl','rb'))
        return dic['X'],dic['Y']
    except IOError:
        pass
    print("Gening Training set...\n")
    finished = 0
    result = []
    pool = Pool(cpu_count()-1)
    for i in range(len(testset)):
        resul.append(  pool.apply_async(WorkerForTrain, tuple(testset.loc[i,['start_geo_id','end_geo_id','create_hour']]), callback=CbkForTrain) )

    pool.close()
    pool.join()
    X=[result[i].get()[0]  for i in range(len(result)) ]
    Y=[result[i].get()[1]  for i in range(len(result)) ]
    pickle.dump({'X':X,'Y':Y},open('train.pkl','wb'))
    return X,Y

def WorkerForTest(*x):
    """
    x - ['start_geo_id','end_geo_id','create_date','create_hour']
    """
    feature = [(dt.datetime.strptime(x[2],'%Y-%m-%d') - dt.datetime(2017,7,1)).days*24 + x[3]]
    return feature

def CbkForTest(result):
    finished = finished + 1
    if finished % 100 == 0:
        print("%s, gened test %d\n" % (dt.datetime.now(),finished))

def GenTestSet():
    try:
        tdic = pickle.load(open('test.pkl','rb'))
        return tdic['X']
    except IOError:
        pass
    print("Gening Test set...\n")
    finished = 0
    result = []
    pool = Pool(cpu_count()-1)
    for i in range(len(testset)):
        result.append( pool.apply_async(WorkerForTest,tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']]),callback=CbkForTest))
    pool.close()
    pool.join()
    TX = [result[i].get() for i in range(len(result))]
    pickle.dump({'X':TX},open('test.pkl','wb'))
    return TX

