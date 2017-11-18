import pandas as pd
import datetime as dt
import numpy as np
import pickle

class DatasetGenerator(object):
    """description of class"""
    def __init__(self, train_july, train_aug,testFile):
        self.julyset = pd.read_csv(train_july)
        self.augset = pd.read_csv(train_aug)
        self.testset = pd.read_csv(testFile)
        self.trainset = pd.concat([self.julyset, self.augset])

    def GetCurrentDateStat(self,trainSet, date):
        tmp = trainSet[  (trainSet['create_date']==date) ].groupby('create_hour').size().reset_index(name='count')
        return tmp
    
    def GetHistoryMean(self,trainSet):
        tmp = trainSet.groupby(['create_date','create_hour']).size().reset_index(name='count')
        return tmp[tmp['count']<80].groupby('create_hour').mean().reset_index()

    def GenTrainingSet(self):
        try:
            dic = pickle.load(open('train.pkl','rb'))
            return dic['X'],dic['Y']
        except IOError:
            pass
        X = []
        Y = []
        dates = [ (dt.datetime(2017,7,1)+dt.timedelta(i)).strftime('%Y-%m-%d') for i in range(31) ]
        for i in range(len(self.testset)):
            x = self.testset.iloc[i]
            start = x['start_geo_id']
            end = x['end_geo_id']
            hur = x['create_hour']
            tmpset = self.trainset[(self.trainset['start_geo_id']==start) & (self.trainset['end_geo_id']==end)]
            for d in dates:
                if i%100==0:
                    print('%s: day %s, gening %d\n'%(dt.datetime.now(),d,i))
                cur = np.zeros(12)
                #todo : modify these start point and end point
                f = self.GetCurrentDateStat(tmpset,d)
                y=0
                for j in range(len(f)):
                    if (hur%2==0 and j%2!=0) or (hur%2!=0 and j%2==0):
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

                X.append(his)
                Y.append(y)

        pickle.dump({'X':X,'Y':Y},open('train.pkl','wb'))
        return X,Y
    def GenTestSet(self):
        try:
            tdic = pickle.load(open('test.pkl','rb'))
            return tdic['X']
        except IOError:
            pass
        X = []
        for i in range(len(self.testset)):
            x = self.testset.iloc[i]
            start = x['start_geo_id']
            end = x['end_geo_id']
            d = x['create_date']
            hur = x['create_hour']
            tmpset = self.trainset[(self.trainset['start_geo_id']==start) & (self.trainset['end_geo_id']==end)]
            if i%100==0:
                print('%s: day %s, gening %d\n'%(dt.datetime.now(),d,i))
            cur = np.zeros(12)
            #todo : modify these start point and end point
            f = self.GetCurrentDateStat(tmpset,d)
            for j in range(len(f)):
                if (hur%2==0 and j%2!=0) or (hur%2!=0 and j%2==0):
                    cur[f.iloc[j,0]//2] = f.iloc[j,1]
                if f.iloc[j,0]==hur:
                    y = f.iloc[j,1]

            if hur%2==0:
                p=(hur-1)//2
            else:
                p=hur//2
            cur = np.roll(cur,5-p)
            #todo : the same as above
            g = self.GetHistoryMean(tmpset)
            his=np.zeros(24)
            for j in range(len(g)):
                his[ int(g.iloc[j,0]) ] = g.iloc[j,1]

            his=list(np.roll(his,11-hur))
            his.extend(cur)
            X.append(his)
        pickle.dump({'X':X},open('test.pkl','wb'))
        return X

