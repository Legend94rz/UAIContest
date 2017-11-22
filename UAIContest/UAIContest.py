import pandas as pd
from DatasetGenerator import GenTestSet,GenTrainingSet,dummy
from ILearner import UseMean,Xgb,Linear,GausProc
from sklearn.gaussian_process.kernels import ExpSineSquared,WhiteKernel
from multiprocessing.pool import Pool
import numpy as np
import matplotlib.pyplot as plt

YP = []
models = []
def trainAndPredict(XI,YI,TXI,modeli):
    modeli.train(XI,YI)
    return modeli.predict(TXI)

def log_result(yp):
    YP.append(float(yp))
    if len(YP)%100==0:
        print(len(YP))

def GenResult(X,Y,TX):
    global models
    models = [GausProc(kernel = ExpSineSquared(periodicity=24)) for i in range(len(TX))]
    p = Pool()
    for i in range(len(TX)):
        XI = np.array(X[i]).reshape(-1,1)
        YI = np.array(Y[i]).reshape(-1,1)
        TXI = np.array(TX[i]).reshape(-1,1)
        p.apply_async(trainAndPredict , (XI,YI,TXI,models[i]),callback = log_result )
    p.close()
    p.join()
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
    X,Y = GenTrainingSet()
    TX = GenTestSet()
    GenResult(X,Y,TX)
    #dummy()
