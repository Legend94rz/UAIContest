import pandas as pd
from DatasetGenerator import Synthe,testset,trainset
import math
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

poi = pd.read_csv('Data\\poi.csv',encoding = 'ansi', header=None,names =list(range(21)))
poi[22] = poi[12]+poi[18]

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
    date = '2017-08-02'
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

    target = trainset[(trainset['create_date']==DATE)&(trainset['create_hour']==hur)&(trainset['start_geo_id']==start)&(trainset['end_geo_id']==end)].shape[0]

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

    return sim,meanOfWeek,pre,nxt,L,R,meanOfHis,np.mean(s),target

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
    return [poi1,poi2,p[1],testset.loc[i,'create_hour'],p[6],p[-1]]


def GenResult(X,Y,TX):
    #result = pd.DataFrame()
    #result['test_id'] = range(5000)
    #yp = []
    #col = ['sim','meanOfWeek','pre','nxt','L','R','meanOfHis','meanOfAll']
    #for i in range(len(testset)):
    #    yp.append(makeOnePrediction(X,i))
    #
    #yp = np.array(yp)
    #for i in range(len(col)):
    #    result[col[i]] = yp[:,i]
    #result.to_csv('Fe.csv',index=False)
    A = pd.DataFrame()
    B = pd.DataFrame()
    r = []
    s = []
    col = ['poi1','poi2','meanOfWeek','hur','meanOfHis','target']
    for i in range(len(testset)):
        if i%100==0:
            print('%s %d'%(dt.datetime.now(),i))
        try:
            poi1 = poi[ poi[0]==testset.loc[i,'start_geo_id'] ][22].item()
        except ValueError:
            poi1 = 0
        try:
            poi2 = poi[ poi[0]==testset.loc[i,'end_geo_id'] ][22].item()
        except ValueError:
            poi2 =0
        p = makeOnePrediction(X,i)
        if testset.loc[i,'create_hour']%2==0:
            r.append([poi1,poi2,p[1],testset.loc[i,'create_hour'],p[6],p[-1]])
        else:
            s.append([poi1,poi2,p[1],testset.loc[i,'create_hour'],p[6]])
    r = np.array(r)
    s = np.array(s)
    for i in range(len(col)):
        A[col[i]] = r[:,i]
        if i!=len(col-1):
            B[col[i]] = s[:,i]

    A.to_csv('A.csv',index =False)
    B.to_csv('B.csv',index =False)



def analisis(X):
    R =pd.read_csv('1.97.csv')
    for i in range(666,len(testset)):
        #if ind[i]:
        if True:
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
            #axs[k].plot(x,y)
            plt.plot(x,y)
            days = (dt.datetime.strptime(testset.loc[i,'create_date'],'%Y-%m-%d')-dt.datetime(2017,7,1)).days
            #axs[k].plot(days*24+testset.loc[i,'create_hour'], R.loc[i,'count'],'rx')
            plt.plot(days*24+testset.loc[i,'create_hour'], R.loc[i,'count'],'rx')
            x=range(testset.loc[i,'create_hour'],31*24,24)
            y=[s[j] for j in x]
            #axs[k].plot(x,y,'bx')
            plt.plot(x,y,'bx')
            #f.subplots_adjust(hspace=0)
            plt.show()


if __name__ == "__main__":
    X,Y,TX = Synthe()
    GenResult(X,Y,X)
    #analisis(X)
