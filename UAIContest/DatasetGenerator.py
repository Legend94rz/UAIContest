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

def GetCurrentDateStat(trainSet, date):
    tmp = trainSet[  (trainSet['create_date']==date) ].groupby('create_hour').size().reset_index(name='count')
    return tmp

def GetHistoryMean(trainSet):
    tmp = trainSet.groupby(['create_date','create_hour']).size().reset_index(name='count')
    return tmp[tmp['count']<80].groupby('create_hour').mean().reset_index()

def GetFeature(d, hur, tmpset):
    cur = np.zeros(12)
    #todo : modify these start point and end point
    f = GetCurrentDateStat(tmpset,d)
    y=0
    for j in range(len(f)):
        if (hur%2==0 and f.iloc[j,0]%2!=0) or (hur%2!=0 and f.iloc[j,0]%2==0):
            cur[f.iloc[j,0]//2] = f.iloc[j,1]
        if f.iloc[j,0]==hur:
            y = f.iloc[j,1]
    if hur%2==0:
        p=(hur-1)//2
    else:
        p=hur//2
    cur = np.roll(cur,5-p)
    
    his=np.zeros(24)
    #todo : the same as above
    g = GetHistoryMean(tmpset)
    for j in range(len(g)):
        his[ int(g.iloc[j,0]) ] = g.iloc[j,1]
    feature=list(np.roll(his,11-hur))
    feature.extend(cur)
    return feature, y

def WorkerForTrain(*x):
    """
    x - [start_geo_id, end_geo_id, create_hour]
    """
    X = []
    Y = []
    dates = [ (dt.datetime(2017,7,1)+dt.timedelta(k)).strftime('%Y-%m-%d') for k in range(31) ]
    tmpset = trainset[(trainset['start_geo_id']==x[0]) & (trainset['end_geo_id']==x[1])]
    for d in dates:
        feature, y = GetFeature(d, x[2], tmpset)
        X.append(feature)
        Y.append(y)
    return (X,Y)

def CbkForTrain(result):
    X.extend(result[0])
    Y.extend(result[1])
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

