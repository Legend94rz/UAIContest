import pandas as pd
from DatasetGenerator import Synthe,testset,trainset, poi, weather,OutlierSet
import math
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing.pool import Pool
from multiprocessing import cpu_count


def calcSimilitude(y):
    '''
    Y - 4 by 7 of matrix
    '''
    sim = 0
    cnt = 0
    for j in range(1,4):
        a = np.sum(y[j] * y[j - 1])
        b = np.linalg.norm(y[j]) * np.linalg.norm(y[j - 1])
        if b == 0:
            continue
        sim = sim + (a / b)
        cnt = cnt + 1
    if cnt > 0:
        return sim / cnt
    else:
        return 0
    return sim

def makeOnePrediction(X,INDEX):
    start = testset.loc[INDEX,'start_geo_id']
    end = testset.loc[INDEX,'end_geo_id']
    date = testset.loc[INDEX,'create_date']
    DATE = dt.datetime.strptime(date,'%Y-%m-%d')
    hur = testset.loc[INDEX,'create_hour']
    XI = np.array(X[INDEX])
    s = XI.sum(0)

    days = (DATE-dt.datetime(2017,7,1)).days
    y = np.reshape([s[j] for j in range(hur,28 * 24,24)] , (4,7))
    #sim = calcSimilitude(y)
    sim = 0
    meanOfWeek = y[:,days%7].mean()

    preTm = DATE + dt.timedelta(hours=int(hur - 1))
    nxtTm = DATE + dt.timedelta(hours=int(hur + 1))
    L = 24 * 31 + 12 * (preTm - dt.datetime(2017,8,1)).days + preTm.hour // 2
    R = 24 * 31 + 12 * (nxtTm - dt.datetime(2017,8,1)).days + nxtTm.hour // 2
    #while L>0 and s[L]==0:
    #    L = L-1
    #while R<len(s) and s[R]==0:
    #    R = R+1

    pre = s[L]
    if R>=len(s):
        R = len(s)-1
    nxt = s[R]


    allHis = []
    for j in range(38):
        if j < 31:
            allHis.append(s[hur + 24 * j])
        else:
            if j%2!=hur%2:
                allHis.append(s[31 * 24 + (j - 31) * 12 + int(hur // 2)])
    allHis = np.array(allHis)
    meanOfHis = 0
    if len(allHis[(allHis>0)]) > 0:
       meanOfHis = allHis[(allHis>0)].mean()

    return sim,meanOfWeek,pre,nxt,L,R,meanOfHis,np.mean(s)

def work(X,i):
    try:
        poi1 = poi[ poi[0]==testset.loc[i,'start_geo_id'] ][22].item()
    except ValueError:
        poi1 = 0
    try:
        poi2 = poi[ poi[0]==testset.loc[i,'end_geo_id'] ][22].item()
    except ValueError:
        poi2 =0
    p = makeOnePrediction(X,i)
    return [poi1,poi2,testset.loc[i,'create_hour'],p[1],p[6],p[-1]]

Q = []
def log_result(res):
    Q.append(res)
    if len(Q)%100==0:
        print('%s %d'%(dt.datetime.now(),len(Q)))

def GenResult(X,Y,TX):
    col = ['sim','meanOfWeek','pre','nxt','L','R','meanOfHis','meanOfAll']
    f186 = pd.read_csv('1.86.csv')
    f197 = pd.read_csv('1.97.csv')
    for i in range(len(testset)):
        #t = makeOnePrediction(X,i)
        if testset.loc[i,'create_date']=='2017-08-02' and( testset.loc[i,'create_hour'] == 20 or testset.loc[i,'create_hour'] == 22) and f186.loc[i,'count']>=10:
            f186.loc[i,'count'] = (f186.loc[i,'count']*0.02 + 0.8)*f186.loc[i,'count']
    f186.to_csv('amp2022gt10.csv',index = False)



def analisis(X):
    R = pd.read_csv('1.86.csv')
    l = []

    for i in range(len(testset)):
        if testset.loc[i,'create_date']=='2017-08-02' and( testset.loc[i,'create_hour'] == 20 or testset.loc[i,'create_hour'] == 22):
            print(testset.loc[i])
            #f,axs = plt.subplots(3,sharey = False)
            #for k in range(3):
            x = list(range(31*24))
            for j in range(1,8):
                if j%2==0:
                    x.extend( range( (j+30)*24 + 1, (j+31)*24 ,2) )
                else:
                    x.extend( range( (j+30)*24,(j+31)*24,2))
            #s = X[i][k]
            s = np.sum(X[i],0)
            y = s
            plt.plot(x,y)
            days = (dt.datetime.strptime(testset.loc[i,'create_date'],'%Y-%m-%d')-dt.datetime(2017,7,1)).days
            plt.plot(days*24+testset.loc[i,'create_hour'], R.loc[i,'count'],'rx')
            x=range(testset.loc[i,'create_hour'],31*24,24)
            y=[s[j] for j in x]
            plt.plot(x,y,'bx')
            plt.show()
            l.append(R.loc[i,'count'])
    print(l)


if __name__ == "__main__":
    X,Y,TX = OutlierSet()
    #GenResult(X,Y,X)
    #analisis(X)
    l=[]