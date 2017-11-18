import pandas as pd
import datetime as dt
import numpy as np
import pickle
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from threading import Lock

class DatasetGenerator(object):
    """description of class"""
    def __init__(self, train_july, train_aug,testFile):
        self.julyset = pd.read_csv(train_july)
        self.augset = pd.read_csv(train_aug)
        self.testset = pd.read_csv(testFile)
        self.trainset = pd.concat([self.julyset, self.augset])
        self.lock = Lock()

    def GetCurrentDateStat(self,trainSet, date):
        tmp = trainSet[  (trainSet['create_date']==date) ].groupby('create_hour').size().reset_index(name='count')
        return tmp
    
    def GetHistoryMean(self,trainSet):
        tmp = trainSet.groupby(['create_date','create_hour']).size().reset_index(name='count')
        return tmp[tmp['count']<80].groupby('create_hour').mean().reset_index()

    def GetFeature(self, d, hur, tmpset):
        cur = np.zeros(12)
        #todo : modify these start point and end point
        f = self.GetCurrentDateStat(tmpset,d)
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
        g = self.GetHistoryMean(tmpset)
        for j in range(len(g)):
            his[ int(g.iloc[j,0]) ] = g.iloc[j,1]
        his=list(np.roll(his,11-hur))
        his.extend(cur)
        return his, y

    def miniBatchForTrain(self,x):
        """
        x - [start_geo_id, end_geo_id, create_hour]
        """
        X = []
        Y = []
        dates = [ (dt.datetime(2017,7,1)+dt.timedelta(k)).strftime('%Y-%m-%d') for k in range(31) ]
        tmpset = self.trainset[(self.trainset['start_geo_id']==x[0]) & (self.trainset['end_geo_id']==x[1])]
        for d in dates:
            his, y = self.GetFeature(d, x[2], tmpset)
            X.append(his)
            Y.append(y)
        self.lock.acquire()
        self.finishCount=self.finishCount+1
        if self.finishCount % 100 ==0:
            print( self.finishCount )
        self.lock.release()
        return (X,Y)

    def GenTrainingSet(self):
        try:
            dic = pickle.load(open('train.pkl','rb'))
            return dic['X'],dic['Y']
        except IOError:
            pass
        print("Gening Training set...\n")
        pool = ThreadPool(cpu_count())
        self.finishCount=0
        X = []
        Y = []
        ind = list(self.augset[['start_geo_id','end_geo_id','create_hour']].values)
        Set = pool.map(self.miniBatchForTrain,ind)
        pool.close()
        pool.join()
        for ele in Set:
            X.extend(ele[0])
            Y.extend(ele[1])
        pickle.dump({'X':X,'Y':Y},open('train.pkl','wb'))
        return X,Y

    def OneSampleForTest(self,x):
        """
        x - ['start_geo_id','end_geo_id','create_date','create_hour']
        """
        tmpset = self.trainset[(self.trainset['start_geo_id']==x[0]) & (self.trainset['end_geo_id']==x[1])]
        his,y = self.GetFeature(x[2],x[3],tmpset)
        self.lock.acquire()
        self.finishCount=self.finishCount+1
        if self.finishCount % 100 ==0:
            print( self.finishCount )
        self.lock.release()
        return his

    def GenTestSet(self):
        try:
            tdic = pickle.load(open('test.pkl','rb'))
            return tdic['X']
        except IOError:
            pass
        print("Gening Test set...\n")
        self.finishCount=0
        pool = ThreadPool(cpu_count())
        ind = list(self.augset[['start_geo_id','end_geo_id','create_date','create_hour']].values)
        X = pool.map(self.OneSampleForTest,ind)
        pool.close()
        pool.join()
        pickle.dump({'X':X},open('test.pkl','wb'))
        return X

