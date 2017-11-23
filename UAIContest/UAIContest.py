import pandas as pd
from DatasetGenerator import GenTrainingSet,dummy
from ILearner import UseMean,Xgb,Linear,GausProc
from sklearn.gaussian_process.kernels import ExpSineSquared,WhiteKernel
from multiprocessing.pool import Pool
import numpy as np
import matplotlib.pyplot as plt
import math

YP = []

def GenResult(X,Y,TX):
    for i in range(len(X)):
        y=[]
        for j in range(3):
            x = X[i][j*3:3+j*3]
            if (math.fabs(x[0]-x[1])<8):
                y.append((x[0]+x[1])/2)
            else:
                y.append(x[2])
        YP.append(sum(y))

    #Gen Result:
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=YP
    result.to_csv('prediction.csv',index=False,encoding='utf-8')

    #compare = pd.DataFrame()
    #compare['pred'] = YP
    #compare['real'] = Y
    #compare.to_csv('compare.csv',index=False)

if __name__=="__main__":
    #dummy()
    X,Y = GenTrainingSet()
    #TX = GenTestSet()
    GenResult(X,Y,X)
