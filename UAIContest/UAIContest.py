import pandas as pd
from DatasetGenerator import Synthe,testset,trainset, poi, weather,OutlierSet,SSSet
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold
from ILearner import ILearner,Xgb
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression

from mlxtend.regressor import StackingRegressor

def saveResult(filename, yp):
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=yp
    result.to_csv(filename+'.csv',index = False)

def GenResult(X,Y,VX,VY,TX):
    L = ILearner()
    models = [
        XGBRegressor(),
        LinearRegression(),
        ]
    meta = XGBRegressor(max_depth = 20)
    stack = StackingRegressor(regressors = models,meta_regressor = meta, verbose = 4)
    stack.fit(X,Y)
    yp = stack.predict(VX)
    print('validation socre: %f\n' % L.score(VY,yp))
    Final = stack.predict(TX)
    saveResult('stack',Final)


if __name__ == "__main__":
    X,Y,VX,VY,TX = SSSet()
    s = np.random.rand(len(Y))
    ind = ((Y==1)&(s>=0.5)) | (Y!=1)
    X = X[ind]
    Y = Y[ind]
    GenResult(X,Y,VX,VY,TX)