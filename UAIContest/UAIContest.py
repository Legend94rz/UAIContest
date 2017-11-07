import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

#plot A->B orders count by time
trainPath1 = '.\\Data\\train_July.csv'
trainPath2 = '.\\Data\\train_Aug.csv'
testPath = '.\\Data\\test_id_Aug_agg_public5k.csv'

def ReadTrain():
    train1 = pd.read_csv(trainPath1)
    train2 = pd.read_csv(trainPath2)
    return pd.concat([train1,train2])

def ReadTest():
    return pd.read_csv(testPath)

def analysis(trainSet,testSet):
    result = pd.DataFrame()
    result['test_id'] = testSet['test_id']
    p=[]
    for i in range(len(testSet)):
        if i%500==0:
            print('%s: %d'%(dt.datetime.now(), i))
        x = testSet.iloc[i]
        startId = x['start_geo_id']
        endId = x['end_geo_id']
        date = x['create_date']
        hur = x['create_hour']
        #Date = dt.datetime.strptime(date,'%Y-%m-%d')
        #Date.replace(hour=int(hur))
        tmp = trainSet[(trainSet['start_geo_id']==startId) & (trainSet['end_geo_id']==endId) & (trainSet['create_date'==date])]\
              .sort_values(['create_date','create_hour'])
        s = 0
        if(hour>0):
            s = s + tmp[tmp['create_hour']==hour-1].shape[0]
        if(hour<23):
            s = s + tmp[tmp['create_hour']==hour+1].shape[0]
        if(hour>0 and hour<23):
            s = s/2.0;
        p.append(s)
    result['count']=p
    result.to_csv('prediction.csv',encoding='utf-8',index = False)

if __name__=="__main__":
    trainSet = ReadTrain()
    testSet = ReadTest()
    analysis(trainSet,testSet)