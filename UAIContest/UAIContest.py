import pandas as pd
from DatasetGenerator import testset, trainset, SSSet
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor, Ridge
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
    FileName = 'gbr_300est_300dep_256min_useEst_nomerge'
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

if __name__ == "__main__":
    Train,Test,Final = SSSet()
    GenResult(Train,Final)
