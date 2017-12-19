import pandas as pd
from DatasetGenerator import Synthe,testset,trainset, poi, weather,OutlierSet,SSSet
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold
from ILearner import ILearner,Xgb
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor, Ridge

from mlxtend.regressor import StackingRegressor

def saveResult(filename, yp):
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=yp
    result.to_csv(filename+'.csv',index = False)

def GenResult(X,Y,TX):
    L = ILearner()
    models = [
        XGBRegressor(),
        PassiveAggressiveRegressor(max_iter = 20)
        ]
    meta = XGBRegressor(max_depth = 20)
    stack = StackingRegressor(regressors = models,meta_regressor = meta, verbose = 4)
    stack.fit(X,Y)
    Final = stack.predict(TX)
    saveResult('stack',Final)


def stratifiedSampling(group):
    if group.name==1:
        frac = 0.7
    else:
        frac = 1
    return group.sample(frac = frac)

if __name__ == "__main__":
    Train, Test = SSSet()
    Train = Train.groupby('count').apply(stratifiedSampling)
    GenResult(Train.iloc[:,4:-1],Train.iloc[:,-1], Test.iloc[:,4:])
