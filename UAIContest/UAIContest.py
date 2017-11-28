import pandas as pd
from DatasetGenerator import Split012,Synthe
import math
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

#todo: remove this dependency
testFile = '.\\Data\\test_id_Aug_agg_public5k.csv'
testset = pd.read_csv(testFile)

def GenResult(X,Y,TX):
    result = pd.DataFrame()
    result['test_id']=range(5000)
    yp = []
    for i in range(len(testset)):
        x=testset.loc[i]
        days = (dt.datetime.strptime(x['create_date'],'%Y-%m-%d')-dt.datetime(2017,7,1)).days
        hur = x['create_hour']
        fe = np.array(X[i]).sum(0)
        s = []
        for j in range(38):
            if j<31:
                s.append(fe[hur+24*j])
            else:
                s.append( fe[31*24 + (j-31)*12 + int(hur//2)] )
        s = np.array(s)
        m = np.ceil( np.mean(s[s<=50]) )
        yp.append(m)
    result['count']=yp
    print(np.mean(yp))
    result.to_csv('opt_hist_ceil.csv',index=False)

def analisis(X):
    R = pd.read_csv("1.97.csv")
    ind = (R['count']<1.5)&(R['count']>=0.5)
    for i in range(len(ind)):
        if ind[i]:
            print(testset.loc[i])
            x = list(range(31*24))
            for j in range(1,8):
                if j%2==0:
                    x.extend( range( (j+30)*24 + 1, (j+31)*24 ,2) )
                else:
                    x.extend( range( (j+30)*24,(j+31)*24,2))
            s = np.sum( X[i],0)
            y = s
            plt.plot(x,y)
            days = (dt.datetime.strptime(testset.loc[i,'create_date'],'%Y-%m-%d')-dt.datetime(2017,7,1)).days
            plt.plot(days*24+testset.loc[i,'create_hour'], R.loc[i,'count'],'rx')
            x=range(testset.loc[i,'create_hour'],31*24,24)
            y=[s[j] for j in x]
            plt.plot(x,y,'bx')
            plt.show()


if __name__=="__main__":
    X,Y,TX = Synthe()
    #GenResult(X,Y,X)
    analisis(X)
