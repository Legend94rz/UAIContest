import pandas as pd
from DatasetGenerator import testset, trainset, SSSet
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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

def GenResult(X,TX):
    #X = X.groupby('count').apply(stratifiedSampling)
    X['spoi'] = X[['soil', 'smarket', 'suptown', 'ssubway', 'sbus', 'scaffee', 'schinese', 'satm', 'soffice', 'shotel']].sum(axis = 1)
    X['tpoi'] = X[['toil', 'tmarket', 'tuptown', 'tsubway', 'tbus', 'tcaffee', 'tchinese', 'tatm', 'toffice', 'thotel']].sum(axis = 1)

    TX['spoi'] = TX[['soil', 'smarket', 'suptown', 'ssubway', 'sbus', 'scaffee', 'schinese', 'satm', 'soffice', 'shotel']].sum(axis = 1)
    TX['tpoi'] = TX[['toil', 'tmarket', 'tuptown', 'tsubway', 'tbus', 'tcaffee', 'tchinese', 'tatm', 'toffice', 'thotel']].sum(axis = 1)

    featName = ['spoi','tpoi','feels_like0','wind_scale0','humidity0',
                 'estimate','hisMean','weekMean',\
                 'weekday','day','hour','-1','1']
    X['residual'] = X['count'] - X['estimate']
    m = GradientBoostingRegressor(loss='lad',n_estimators = 300,max_depth = 300, learning_rate = 0.1, verbose = 2)
    #m = XGBRegressor(n_estimators = 300, n_jobs = 3, max_depth = 10, learning_rate = 0.1)
    m.fit(X[featName],X['residual'])
    GBRresult = m.predict(TX[featName]) + TX['estimate']
    #w = pd.read_csv('w.csv')['count']
    #result = np.ceil(GBRresult.values * 0.6 + w.values * 0.4)
    saveResult('GBR_nomerge',GBRresult.values)
    #plot_importance(m)
    #plt.show()
    '''
    #use 7.18 to predict 8.2
    featName = ['spoi', 'tpoi', 'estimate','hisMean','hour','-1','1']
    x718 = X[(X['create_date'] == '2017-07-18')]
    tx82 = TX[(TX['create_date'] == '2017-08-02')]
    m2 = GradientBoostingRegressor(loss = 'lad',n_estimators = 100,max_depth=50,learning_rate =0.1,verbose = 2)
    m2.fit(x718[featName],x718['residual'])
    R82 = m2.predict(tx82[featName])+tx82['estimate']
    result[TX['create_date']=='2017-08-02'] =  result[TX['create_date']=='2017-08-02']*0.5 + R82*0.5
    saveResult('GBR',result)

    p = pd.DataFrame()
    p['r82']=R82
    p.to_csv('r82.csv')
    '''

def stratifiedSampling(group):
    if group.name==1:
        frac = 0.85
    else:
        frac = 1
    return group.sample(frac = frac)

## K-NN:
'''
    import pickle
    D = pickle.load(open('EveryPair.pkl','rb'))
    ans = []
    mat = testset.values
    K = 10
    for i in range(len(mat)):
        k = mat[i,1]+mat[i,2]
        if k not in D:
            ans.append(0)
            continue
        t = D[k]
        days = (dt.datetime.strptime( mat[i,3],'%Y-%m-%d' )-dt.datetime(2017,6,30)).days
        hur = mat[i,4]
        vec = np.array( [ t[days*24+hur-1],t[days*24+hur+1] ] )
        z = []
        for j in range(2,32*24):
            a1 = np.array( [t[j-2],t[j]] )
            z.append( [ np.linalg.norm( a1-vec ,2 ), t[j-1] ] )
        z = np.array( sorted(z, key = lambda x:x[0]) )

        ans.append(z[:K,1].mean())
    saveResult('1nn',ans)
'''

if __name__ == "__main__":
    Train,Test = SSSet()
    GenResult(Train,Test)

