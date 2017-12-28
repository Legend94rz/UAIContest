import pandas as pd
from DatasetGenerator import Synthe,testset,trainset, poi, weather,OutlierSet,SSSet
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
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



def GenResult(X,Y,TX):
    featName = [ 'soil', 'smarket', 'suptown', 'ssubway', 'sbus', 'scaffee', 'schinese', 'satm', 'soffice', 'shotel',\
                 'toil', 'tmarket', 'tuptown', 'tsubway', 'tbus', 'tcaffee', 'tchinese', 'tatm', 'toffice', 'thotel',\
                 'MyCode0','feels_like0','wind_scale0','humidity0','MyCode1','feels_like1','wind_scale1','humidity1',\
                 'weekday','hour' ]
    estimate = X['estimate']
    residual = X['count']-X['estimate']
    m = GradientBoostingRegressor(loss='lad',n_estimators = 300,max_depth = 300, learning_rate = 0.1, min_samples_leaf = 256, min_samples_split=256,verbose = 2)
    m.fit(X[featName],residual)
    residualY =  m.predict(TX[featName])
    saveResult(residualY + TX['estimate'])




def stratifiedSampling(group):
    if group.name==0:
        frac = 0.01
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

