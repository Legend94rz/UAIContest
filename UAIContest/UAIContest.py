import pandas as pd
from DatasetGenerator import Synthe,testset,trainset, poi, weather,OutlierSet,SSSet
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from ILearner import ILearner,Xgb
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor, Ridge

from mlxtend.regressor  import StackingCVRegressor, StackingRegressor

def saveResult(filename, yp):
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=yp
    result.to_csv(filename+'.csv',index = False)

def GenResult(X,Y,TX):
    xgbr = XGBRegressor()
    pa = PassiveAggressiveRegressor(max_iter = 20)
    models = [
        xgbr,
        pa
        ]
    meta = XGBRegressor()
    stack = StackingRegressor(regressors = models,meta_regressor = meta,verbose = 4)

    params = {'xgbregressor__max_depth':range(4,40,3),
              'passiveaggressiveregressor__C':[0.001,0.01,0.1,0,10,100,1000],
              'meta-xgbregressor__max_depth':range(4,20,2)
        }

    grid = GridSearchCV(estimator = stack,param_grid = params,refit = True)
    grid.fit(X,Y)
    print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
    cv_keys = ('mean_test_score', 'std_test_score', 'params')
    
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))
        if r > 10:
            break
    print('...')
    
    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)

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
    #Train = Train.groupby('count').apply(stratifiedSampling)
    Train = Train.drop(['-9','-7','-5','-3','3','5','7','9'],axis = 1)
    Test = Test.drop(['-9','-7','-5','-3','3','5','7','9'],axis = 1)
    GenResult(Train.iloc[:,4:-1],Train.iloc[:,-1], Test.iloc[:,4:])
