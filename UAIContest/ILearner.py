import math

class ILearner(object):
    def __init__(self):
        pass

    def train(X,Y):
        pass

    def predict(X):
        pass

    def score(YV,yp):
        s = 0
        for i in range(len(yp)):
            s = s + math.fabs(yp[i]-YV[i])
        return s*1.0/len(yp)
