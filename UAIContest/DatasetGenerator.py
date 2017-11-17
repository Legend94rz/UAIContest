import pandas as pd
import sqlite3
from pandas.io import sql
import subprocess
import datetime as dt
import numpy as np
import pickle

class DatasetGenerator(object):
    """description of class"""
    def __init__(self, train_july, train_aug,testFile):
        self.julyset = pd.read_csv(train_july)
        self.augset = pd.read_csv(train_aug)
        self.testset = pd.read_csv(testFile)

        df = pd.concat([self.julyset, self.augset])
        self.con = sqlite3.connect(':memory:')
        sql.to_sql(df, name = 'train_ori', index = False, if_exists='append', con=self.con)

    def dummy(self):
        a = self.GetCurrentDateStat('c538ad66d710f99ad0ce951152da36a4','90bb1d035e403538d20b073aec57bea2','2017-08-01')
        b = self.GetHistoryMean('c538ad66d710f99ad0ce951152da36a4','90bb1d035e403538d20b073aec57bea2')
        l=[]

    def GetCurrentDateStat(self, startGeo, endGeo, date):
        c = self.con.cursor()
        res = c.execute("select create_hour,count(1) from train_ori where start_geo_id=? and end_geo_id=? and create_date=? group by create_hour order by create_hour",(startGeo,endGeo,date))
        return res

    def GetHistoryMean(self,startGeo,endGeo):
        c = self.con.cursor()
        res = c.execute("select create_hour, avg(cnt) from (select create_date,create_hour,count(1) as cnt from train_ori where start_geo_id=? and end_geo_id=? group by create_date,create_hour) where cnt<=100 group by create_hour order by create_hour",(startGeo,endGeo))
        return res

    def GenTrainingSet(self):
        try:
            dic = pickle.load(open('train.pkl','rb'))
            return dic['X'],dic['Y']
        except IOError:
            pass
        X = []
        Y = []
        dates = [ (dt.datetime(2017,7,1)+dt.timedelta(i)).strftime('%Y-%m-%d') for i in range(31) ]
        for d in dates:
            print('%s:  at day %s\n'%(dt.datetime.now(),d))
            for i in range(len(self.testset)):
                if i%100==0:
                    print('%s: day %s, gening %d\n'%(dt.datetime.now(),d,i))
                x = self.testset.iloc[i]
                start = x['start_geo_id']
                end = x['end_geo_id']
                cur = np.zeros(12)
                #todo : modify these start point and end point
                f = self.GetCurrentDateStat(start,end,d)
                hur = x['create_hour']
                y=0
                for row in f:
                    if (hur%2==0 and row[0]%2!=0) or (hur%2!=0 and row[0]%2==0):
                        cur[row[0]/2] = row[1]
                    if row[0]==hur:
                        y=row[1]

                if hur%2==0:
                    p=(hur-1)/2
                else:
                    p=hur/2
                cur = np.roll(cur,5-p)
                #todo : the same as above
                his = self.GetHistoryMean(start,end)
                q=np.zeros(24)
                for row in his:
                    q[ int( row[0] )] = row[1]

                q=list(np.roll(q,11-hur))
                q.extend(cur)
                X.append(q)
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
            if i%100==0:
                print('%s: day %s, gening %d\n'%(dt.datetime.now(),d,i))
            cur = np.zeros(12)
            #todo : modify these start point and end point
            f = self.GetCurrentDateStat(start,end,d)
            hur = x['create_hour']
            for row in f:
                if (hur%2==0 and row[0]%2!=0) or (hur%2!=0 and row[0]%2==0):
                    cur[row[0]/2] = row[1]

            if hur%2==0:
                p=(hur-1)/2
            else:
                p=hur/2
            cur = np.roll(cur,5-p)
            #todo : the same as above
            his = self.GetHistoryMean(start,end)
            q=np.zeros(24)
            for row in his:
                q[ int( row[0] )] = row[1]

            q=list(np.roll(q,11-hur))
            q.extend(cur)
            X.append(q)
        pickle.dump({'X':X},open('test.pkl','wb'))
        return X

