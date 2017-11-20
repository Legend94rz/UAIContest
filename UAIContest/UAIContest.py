import pandas as pd
from DatasetGenerator import GenTestSet,GenTrainingSet
from ILearner import UseMean,Xgb
from multiprocessing.pool import Pool

YP = []
def log_result(result):
    YP.append(int(result))

def GenResult(X,Y,TX):
    p = Pool()
    models = [Xgb() for i in range(len(TX))]
    for i in range(len(TX)):
        p.apply_async(Xgb.train , (models[i],X[i],Y[i]))
    p.close()
    p.join()

    p = Pool()
    for i in range(len(TX)):
        p.apply_async(Xgb.predict, (models[i],TX[i]),callback = log_result)
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
