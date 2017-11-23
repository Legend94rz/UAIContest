import pandas as pd
from DatasetGenerator import GenTestSet,GenTrainingSet
from ILearner import UseMean,Xgb,Linear
from multiprocessing.pool import Pool
import numpy as np
import matplotlib.pyplot as plt

YP = []

def GenResult(X,Y,TX):
    models = [Linear() for i in range(len(TX))]
    for i in range(len(TX)):
        XI = np.array(X[i]).reshape(-1,1)
        YI = np.array(Y[i]).reshape(-1,1)
        m = np.mean( YI )
        ind = np.abs(YI-m)<=2*m
        models[i].train(XI[ind].reshape(-1,1),YI[ind].reshape(-1,1))
        YP.append( float( models[i].predict( np.array(TX[i]).reshape(-1,1)  )  ) )

    #Gen Result:
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=YP
    result.to_csv('prediction.csv',index=False,encoding='utf-8')


if __name__=="__main__":
    X,Y = GenTrainingSet()
    TX = GenTestSet()
    GenResult(X,Y,TX)
