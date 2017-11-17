import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
import xgboost
from DatasetGenerator import DatasetGenerator


#plot A->B orders count by time
trainPath1 = '.\\Data\\train_July.csv'
trainPath2 = '.\\Data\\train_Aug.csv'
testPath = '.\\Data\\test_id_Aug_agg_public5k.csv'

trainJuly = pd.DataFrame()
trainAug = pd.DataFrame()
trainSet = pd.DataFrame()


def ReadTrain():
    global trainJuly, trainAug, trainSet
    trainJuly = pd.read_csv(trainPath1)
    trainAug = pd.read_csv(trainPath2)
    trainSet = pd.concat([trainJuly,trainAug])
    return trainJuly,trainAug

def ReadTest():
    return pd.read_csv(testPath)

def GetCurrentDateStat(trainSet, startGeo,endGeo,days):
    date = dt.datetime(2017,7,1) + dt.timedelta(days)
    tmp = trainSet[ (trainSet['start_geo_id']==startGeo) & (trainSet['end_geo_id']==endGeo) & (trainSet['create_date']==date.strftime('%Y-%m-%d')) ]\
            .groupby('create_hour').size().reset_index(name='count')
    return tmp

def GetHistoryMean(trainSet,startGeo,endGeo):
    tmp = trainSet[(trainSet['start_geo_id']==startGeo) & (trainSet['end_geo_id']==endGeo) ]\
        .groupby(['create_date','create_hour']).size().reset_index(name='count')
    return tmp.groupby('create_hour').mean().reset_index()
    
def deal(trainSet,testSet):
    trainSet = trainSet[['start_geo_id','end_geo_id','create_date','create_hour']]
    #####Gen All Training Set and Train:
    """
    X = []
    Y = []
    for d in range(31):
        print('%s:  at day %d\n'%(dt.datetime.now(),d))
        #for i in range(5000):
        for i in range(len(testSet)):
            if i%100==0:
                print('%s: day %d, gening %d\n'%(dt.datetime.now(),d,i))
            x = testSet.iloc[i]
            cur = np.zeros(12)
            #todo : modify these start point and end point
            f = GetCurrentDateStat(trainSet,x['start_geo_id'],x['end_geo_id'],d)
            hur = x['create_hour']
            for j in range(len(f)):
                if (hur%2==0 and j%2!=0) or (hur%2!=0 and j%2==0):
                    cur[f.iloc[j]['create_hour']/2] = f.iloc[j]['count']

            if x['create_hour']%2==0:
                p=(x['create_hour']-1)/2
            else:
                p=x['create_hour']/2
            cur = np.roll(cur,5-p)
            #todo : the same as above
            his = GetHistoryMean(trainSet, x['start_geo_id'],x['end_geo_id'])
            q=np.zeros(24)
            for j in range(len(his)):
                q[ int(his.iloc[j]['create_hour']) ]=his.iloc[j]['count']
            q=list(np.roll(q,11-x['create_hour']))
            q.extend(cur)
            X.append(q)
            if len(f[f['create_hour']==hur])>0:
                Y.append(f[f['create_hour']==hur]['count'].item())
            else:
                Y.append(0)
    pickle.dump({'X':X,'Y':Y},open('train.pkl','wb'))
    """
    #todo: split validation and train set:
    dic = pickle.load(open('train.pkl','rb'))
    X = dic['X']
    Y = dic['Y']
    m = xgboost.XGBRegressor()
    print("training...\n")
    m.fit(X ,Y)
    print("train over, score: %.5lf\n"%(m.score(X,Y)))

    """
    ###Gen All Test set and Gen Result:
    TX = []
    print('%s:  gen test set...\n'%dt.datetime.now())
    for i in range(5000):
        if i%100==0:
            print('%s:  gen %d\n'%(dt.datetime.now(),i))
        x = testSet.iloc[i]
        cur = np.zeros(12)
        days = (dt.datetime.strptime(x['create_date'],'%Y-%m-%d')-dt.datetime(2017,7,1)).days
        f = GetCurrentDateStat(trainSet,x['start_geo_id'],x['end_geo_id'],days)
        hur = x['create_hour']
        for j in range(len(f)):
            if (hur%2==0 and j%2!=0) or (hur%2!=0 and j%2==0):
                cur[f.iloc[j]['create_hour']/2] = f.iloc[j]['count']
        
        if x['create_hour']%2==0:
            p=(x['create_hour']-1)/2
        else:
            p=x['create_hour']/2
        cur = np.roll(cur,5-p)
        his = GetHistoryMean(trainSet, x['start_geo_id'],x['end_geo_id'])
        q=np.zeros(24)
        for j in range(len(his)):
            q[ int(his.iloc[j]['create_hour']) ]=his.iloc[j]['count']
        q=list(np.roll(q,11-x['create_hour']))
        q.extend(cur)
        TX.append(q)
    pickle.dump({'X':TX},open('test.pkl','wb'))
    """

    tdic = pickle.load(open('test.pkl','rb'))
    TX = tdic['X']
    YP = m.apply(TX)
    raw_input('wait')
    YP[YP<0] = 0

    #Gen Result:
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=YP
    result.to_csv('prediction2.csv',index=False,encoding='utf-8')


if __name__=="__main__":
    #trainJuly,trainAug = ReadTrain()
    #testSet = ReadTest()
    #deal(trainSet,testSet)
    dg = DatasetGenerator(trainPath1,trainPath2)
    dg.dummy()