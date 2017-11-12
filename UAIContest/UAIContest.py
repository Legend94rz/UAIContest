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
        tmp = trainSet[(trainSet['start_geo_id']==startId) & (trainSet['end_geo_id']==endId) & (trainSet['create_hour']==int(x['create_hour']))]\
              .groupby('create_date').size().reset_index(name='count')
        #X = range(24*38)
        #Y = [0 for j in range(24*38)]
        #for j in range(tmp.shape[0]):
        #    date = dt.datetime.strptime(tmp['create_date'][j],'%Y-%m-%d')
        #    date = date.replace(hour=int(tmp['create_hour'][j]))
        #    delta = date-dt.datetime(2017,7,1,0)
        #    Y[ delta.days*24 + int(delta.seconds/3600)  ] = tmp['count'][j]
        #
        #plt.plot(X,Y)
        #plt.show()
        if len(tmp['count']<100)>0:
            q = tmp[tmp['count']<100].mean().item()
            if q<3:
                q = tmp[tmp['count']<100]['count'].mode().mean()
        else:
            q=0
        p.append( q )

    result['count']=p
    result.to_csv('prediction.csv',index=False)


if __name__=="__main__":
    trainSet = ReadTrain()
    testSet = ReadTest()
    analysis(trainSet,testSet)