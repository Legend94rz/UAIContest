import pandas as pd
from DatasetGenerator import GenTestSet,GenTrainingSet
from ILearner import UseMean,Xgb

def GenResult(X,Y,TX):
    m = UseMean()
    m.train(X,Y)
    YP = m.predict(X)
    print("score: %.5f\n"%m.score(YP,Y))

    #Gen Result:
    #result = pd.DataFrame()
    #result['test_id'] = range(5000)
    #result['count']=YP
    #result.to_csv('prediction.csv',index=False,encoding='utf-8')

    result = pd.DataFrame()
    result['pred'] = YP
    result['real'] = Y
    result.to_csv('compare.csv',index=False)

if __name__=="__main__":
    X,Y = GenTrainingSet()
    TX = GenTestSet()
    GenResult(X,Y,TX)
