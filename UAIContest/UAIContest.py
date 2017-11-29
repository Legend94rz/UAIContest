import pandas as pd
from DatasetGenerator import Split012,Synthe
import math
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

#todo: remove this dependency
testFile = '.\\Data\\test_id_Aug_agg_public5k.csv'
testset = pd.read_csv(testFile)

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

def makeOnePrediction(X,INDEX):
    start = testset.loc[INDEX,'start_geo_id']
    end = testset.loc[INDEX,'end_geo_id']
    date = testset.loc[INDEX,'create_date']
    hur = testset.loc[INDEX,'create_hour']
    XI = np.array(X[INDEX])
    s = XI.sum(0)

    y = np.reshape([s[j] for j in range(hur,28 * 24,24)] , (4,7))
    sim = calcSimilitude(y)
    meanOfWeek = y[:,3].mean()


    preTm = dt.datetime.strptime(date,'%Y-%m-%d') + dt.timedelta(hours=int(hur - 1))
    nxtTm = dt.datetime.strptime(date,'%Y-%m-%d') + dt.timedelta(hours=int(hur + 1))
    pre = s[24 * 31 + 12 * (preTm - dt.datetime(2017,8,1)).days + preTm.hour // 2]
    if  date=='2017-08-07' and hur == 23 :
        nxt = pre
    else:
        nxt = s[24 * 31 + 12 * (nxtTm - dt.datetime(2017,8,1)).days + nxtTm.hour // 2]
    meanOfPN = (nxt+pre)/2

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

    return sim,meanOfWeek,pre,nxt,meanOfHis

def GenResult(X,Y,TX):
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    yp = []
    col = ['sim','meanOfWeek','pre','nxt','meanOfHis']
    for i in range(len(testset)):
        yp.append(makeOnePrediction(X,i))

    yp = np.array(yp)
    for i in range(len(col)):
        result[col[i]] = yp[:,i]

    result.to_csv('Fe.csv',index=False)

def analisis(X):
    R = pd.read_csv("1.97.csv")
    q = []
    for i in range(len(R)):
        print(testset.loc[i])
        x = range(testset.loc[i,'create_hour'],28 * 24,24)
        s = np.sum(X[i] ,0)
        y = [s[j] for j in x]
        q.append(calcSimilitude(y))
    r = pd.DataFrame()
    r['q'] = q
    r.to_csv('q.csv')

    '''
    ind = (R['count']<1.5)&(R['count']>=0.5)
    for i in range(len(R)):
   #for i in [0,664,665,666,680,687,736,1769]:
        #if ind[i]:
        if True:
            print(testset.loc[i])
            f,axs = plt.subplots(3,sharey = False)
            for k in range(3):
                x = list(range(31*24))
                for j in range(1,8):
                    if j%2==0:
                        x.extend( range( (j+30)*24 + 1, (j+31)*24 ,2) )
                    else:
                        x.extend( range( (j+30)*24,(j+31)*24,2))
                s = X[i][k]
                y = s
                axs[k].plot(x,y)
                days = (dt.datetime.strptime(testset.loc[i,'create_date'],'%Y-%m-%d')-dt.datetime(2017,7,1)).days
                axs[k].plot(days*24+testset.loc[i,'create_hour'], R.loc[i,'count'],'rx')
                x=range(testset.loc[i,'create_hour'],31*24,24)
                y=[s[j] for j in x]
                axs[k].plot(x,y,'bx')
            f.subplots_adjust(hspace=0)
            plt.show()
    '''


if __name__ == "__main__":
    X,Y,TX = Synthe()
    GenResult(X,Y,X)
    #analisis(X)
