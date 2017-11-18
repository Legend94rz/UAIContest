import pandas as pd
from DatasetGenerator import DatasetGenerator
import xgboost

#plot A->B orders count by time
trainPath1 = '.\\Data\\train_July.csv'
trainPath2 = '.\\Data\\train_Aug.csv'
testPath = '.\\Data\\test_id_Aug_agg_public5k.csv'

def GenResult(X,Y,TX):
    #todo: split validation and train set:
    m = xgboost.XGBRegressor()
    print("training...\n")
    m.fit(X ,Y)
    print("train over, score: %.5lf\n"%(m.score(X,Y)))

    YP = m.predict(TX)
    YP[YP<0] = 0

    #Gen Result:
    result = pd.DataFrame()
    result['test_id'] = range(5000)
    result['count']=YP
    result.to_csv('prediction.csv',index=False,encoding='utf-8')

if __name__=="__main__":
    dg = DatasetGenerator(trainPath1,trainPath2,testPath)
    X,Y = dg.GenTrainingSet()
    TX  = dg.GenTestSet()
    GenResult(X,Y,TX)
    