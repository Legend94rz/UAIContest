import math
from xgboost import XGBRegressor
import numpy as np
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class ILearner(object):

    def score(self,YV,yp):
        s = 0.0
        for i in range(len(yp)):
            s = s + math.fabs(yp[i]-YV[i])
        return s/len(yp)

class Xgb(ILearner,XGBRegressor):
    def __init__(self, max_depth = 3, learning_rate = 0.1, n_estimators = 100, silent = True, objective = 'reg:linear', booster = 'gbtree', n_jobs = 1, nthread = None, gamma = 0, min_child_weight = 1, max_delta_step = 0, subsample = 1, colsample_bytree = 1, colsample_bylevel = 1, reg_alpha = 0, reg_lambda = 1, scale_pos_weight = 1, base_score = 0.5, random_state = 0, seed = None, missing = None, **kwargs):
        return super().__init__(max_depth, learning_rate, n_estimators, silent, objective, booster, n_jobs, nthread, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree, colsample_bylevel, reg_alpha, reg_lambda, scale_pos_weight, base_score, random_state, seed, missing, **kwargs)
       
    def predict(self, X):
        YP = super().predict(X)
        YP[YP<0] = 0
        return YP

class UseMean(ILearner,BaseEstimator):
    def __init__(self, cols ,**kwargs):
        self.cols = []
        return super().__init__(**kwargs)
    def predict(self, X):
        YP = []
        return list(YP)
    def set_params(self, **params):
        return super().set_params(**params)
    def get_params(self, deep = True):
        return super().get_params(deep)

class MXgb(ILearner,BaseEstimator):
    pass