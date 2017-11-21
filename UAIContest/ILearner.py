import math
from xgboost import XGBRegressor
import numpy as np
from sklearn.svm import SVR

class ILearner(object):
    def __init__(self):
        pass

    def train(self,X,Y):
        pass

    def train(args):
        args[0].train(*args[1:])

    def predict(self,X):
        pass

    def score(self,YV,yp):
        s = 0.0
        for i in range(len(yp)):
            s = s + math.fabs(yp[i]-YV[i])
        return s/len(yp)

class Xgb(ILearner):
    def __init__(self):
        self.m = XGBRegressor()
    
    def train(self, X, Y):
        self.m.fit(X,Y)

    def predict(self, X):
        YP = self.m.predict(X)
        YP[YP<0] = 0
        return YP

class UseMean(ILearner):
    def __init__(self):
        self.feind = [11]

    def predict(self, X):
        YP = np.array(X)[:,self.feind]
        return list(YP)

class Linear(ILearner):
    def __init__(self):
        self.m = SVR('linear')
    def train(self, X, Y):
        self.m.fit(X,Y)
    def predict(self,X):
        t = self.m.predict(X)
        t[t<0]=0
        return t
