import pandas as pd
from DatasetGenerator import Synthe,testset,trainset, poi, weather,OutlierSet,SSSet
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from ILearner import ILearner,Xgb
from xgboost import XGBRegressor, plot_importance, plot_tree
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor, Ridge
from mlxtend.regressor  import StackingCVRegressor, StackingRegressor
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

Finished = 0
def log_result(res):
    global Finished
    Finished = Finished+1
    print('%s Finished %d'%(dt.datetime.now(),Finished))

def saveResult(filename, yp):
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=yp
    result.to_csv(filename+'.csv',index = False)

def work(*params):
    [id,X,Y,tx] = params
    #todo : how to do if have small number of train samples ?
    if len(Y)==0:
        return 0
    if len(Y)<=5:
        return Y.mean()

    optScore = 2*30
    for d in range(3,10):
        kf = KFold(5)
        m = XGBRegressor(max_depth = d)
        s = 0
        for trainId,testId in kf.split(X):
            m.fit( X.iloc[trainId,:].values,Y.iloc[trainId].values )
            s = s +  np.linalg.norm(m.predict(X.iloc[testId,:].values)-Y.iloc[testId].values , ord = 1)/len(testId)
        if s<optScore:
            optModel = m
            optScore = s
    print('%s %d opt depth is %d, optScore is %f'%(dt.datetime.now(), id, optModel.max_depth, s/5))
    return optModel.predict(tx.values.reshape(1,-1)).tolist()[0]


def GenResult(X,Y,TX):
    p = Pool(cpu_count()-1)
    mat = testset.values
    result = []
    for i in range(len(testset)):
        ind = (X['start_geo_id']==mat[i,1]) & (X['end_geo_id']==mat[i,2])
        x = X[ind].iloc[:,4:]
        y = Y[ind]
        tx = TX.iloc[i,4:]
        result.append( p.apply_async(work, (i,x,y,tx), callback = log_result) )
    p.close()
    p.join()
    yp = [it.get() for it in result]
    saveResult('5000xgb',yp)


def stratifiedSampling(group):
    if group.name==0:
        frac = 0.01
    else:
        frac = 1
    return group.sample(frac = frac)

def FeatureClean(df):
    df['spoi'] = df.iloc[:,4:14].sum(1)     #sum of start poi
    df['tpoi'] = df.iloc[:,14:24].sum(1)    #sum of end poi
    return df


if __name__ == "__main__":
    Train, Test = SSSet()
    #Train = Train.groupby('count').apply(stratifiedSampling)
    Train = FeatureClean(Train)
    Test = FeatureClean(Test)
    f = ['start_geo_id','end_geo_id','create_date','create_hour',\
        'feels_like0','wind_scale0','humidity0','feels_like1','wind_scale1','humidity1','spoi','tpoi','weekday','hour','-1','1','WeekMean','DayMean']
    GenResult(Train.loc[:,f],Train.loc[:,'count'], Test.loc[:,f])


    #mat = testset.values
    #import pickle
    #D = pickle.load(open('EveryPair.pkl','rb'))
    #x = []
    #for i in range(len(testset)):
    #    k = mat[i,1]+mat[i,2]
    #    if k in D:
    #        l = D[k]
    #    else:
    #        l = np.zeros(40*24)
    #    x.append( l[18*24 : 19*24].tolist() + l[32*24:39*24].tolist() )
    #st = pd.DataFrame(x,columns = [ '7-18 '+str(i) for i in range(24) ] + ['8-'+str(i)+' '+str(j) for i in range(1,8) for j in range(24)])
    #st.to_csv('718+8.csv')
