import pandas as pd
from DatasetGenerator import testset, trainset, SSSet
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from ILearner import ILearner,Xgb, Voting
from xgboost import XGBRegressor, plot_importance, plot_tree
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor, Ridge
from mlxtend.regressor  import StackingCVRegressor, StackingRegressor
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import pickle

def saveModel(filename,m):
    pickle.dump({'model':m},open(filename+'.pkl','wb'))

def saveResult(filename, yp):
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=yp
    result.to_csv(filename+'.csv',index = False)

def GenResult(X,TX):
    #X = X.groupby('count').apply(stratifiedSampling)
   # X = X[X['day']<=31]
    X = X.replace(np.inf,0)
    TX = TX.replace(np.inf,0)
    X['spoi'] = X[['soil', 'smarket', 'suptown', 'ssubway', 'sbus', 'scaffee', 'schinese', 'satm', 'soffice', 'shotel']].sum(axis = 1)
    X['tpoi'] = X[['toil', 'tmarket', 'tuptown', 'tsubway', 'tbus', 'tcaffee', 'tchinese', 'tatm', 'toffice', 'thotel']].sum(axis = 1)

    TX['spoi'] = TX[['soil', 'smarket', 'suptown', 'ssubway', 'sbus', 'scaffee', 'schinese', 'satm', 'soffice', 'shotel']].sum(axis = 1)
    TX['tpoi'] = TX[['toil', 'tmarket', 'tuptown', 'tsubway', 'tbus', 'tcaffee', 'tchinese', 'tatm', 'toffice', 'thotel']].sum(axis = 1)

    featName = ['spoi','tpoi','dist','feels_like0','humidity0','humidity1',\
                 'hisMean','weekMean',\
                 'weekday','day','hour']
    X['residual'] = X['count'] - X['estimate']
    m = GradientBoostingRegressor(loss='lad',n_estimators = 300,max_depth = 300, learning_rate = 0.1, verbose = 2, min_samples_leaf = 256, min_samples_split = 256)
    m.fit(X[featName],X['residual'])
    FileName = 'gbr_100est_300dep_256min_useEstimate_nomerge'
    saveModel(FileName,m)
    modelResult = m.predict(TX[featName]) + TX['estimate']
    saveResult(FileName,modelResult.values)
    print(np.mean( modelResult.values ))

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
    Train,Test,Final = SSSet()
    GenResult(Train,Final)

    #v = Voting(['GBR_useNxthur_nomerge_256_100.csv',
    #            'GBR_usePrehour_nomerge_256.csv',
    #            'GBR_useHisMean_nomerge_256.csv',
    #            'GBR_useEstimate_nomerge_256.csv',
    #            'GBR_useEstimate_nomerge_256_300.csv'])
    #result = v.vote()
    #print(result.mean())
    #saveResult('voting', result)
