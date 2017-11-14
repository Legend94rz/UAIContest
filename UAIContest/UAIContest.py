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
    return train1,train2

def ReadTest():
    return pd.read_csv(testPath)


def analysis(trainJuly,trainAug,testSet):
    trainSet = pd.concat([trainJuly,trainAug])
    result = pd.DataFrame()
    result['test_id'] = testSet['test_id']
    p=[]
    for i in range(len(testSet)):
        if i%500==0:
            print('%s: %d'%(dt.datetime.now(), i))
        x = testSet.iloc[i]
        startId = x['start_geo_id']
        endId = x['end_geo_id']
        tmp = trainSet[(trainSet['start_geo_id']==startId) & (trainSet['end_geo_id']==endId) ]\
              .groupby(['create_date','create_hour']).size().reset_index(name='count')
        X = range(24*38)
        Y = [0 for j in range(24*38)]
        for j in range(tmp.shape[0]):
            date = dt.datetime.strptime(tmp['create_date'][j],'%Y-%m-%d')
            date = date.replace(hour=int(tmp['create_hour'][j]))
            delta = date-dt.datetime(2017,7,1,0)
            Y[ delta.days*24 + int(delta.seconds/3600)  ] = tmp['count'][j]
        
        plt.plot(X,Y)
        plt.show()

def GenResult(trainJuly,trainAug,testSet):
    trainSet = pd.concat([trainJuly,trainAug])
    result = pd.DataFrame()
    result['test_id'] = testSet['test_id']
    p=[]
    for i in range(len(testSet)):
        if i%500==0:
            print('%s: %d'%(dt.datetime.now(), i))
        x = testSet.iloc[i]
        startId = x['start_geo_id']
        endId = x['end_geo_id']
        hur = int(x['create_hour'])
        tmp = trainSet[(trainSet['start_geo_id']==startId) & (trainSet['end_geo_id']==endId) ]\
              .groupby(['create_date','create_hour']).size().reset_index(name='count')
        q=0
        w=0
        if len( tmp[(tmp['count']<50) & (tmp['create_hour']==hur)] )>0:
            q = tmp[(tmp['count']<50) & (tmp['create_hour']==hur)]['count'].mean()
            if q<3:
                q = tmp[(tmp['count']<50) & (tmp['create_hour']==hur)]['count'].mode().mean()
        if hur<23:
            try:
                w = w+tmp[(tmp['create_date']==x['create_date']) & (tmp['create_hour']==hur+1)]['count'].item()
            except :
                pass
        if hur>0:
            try:
                w = w+tmp[(tmp['create_date']==x['create_date']) & (tmp['create_hour']==hur-1)]['count'].item()
            except :
                pass
        if hur<23 and hur>0:
            w = w/2.0
        if w==0.5:
            w=1
        p.append( 0.6*q+0.4*w )

    result['count']=p
    result.to_csv('prediction.csv',index=False)

if __name__=="__main__":
    trainJuly,trainAug = ReadTrain()
    testSet = ReadTest()
    #analysis(trainJuly,trainAug,testSet)
    GenResult(trainJuly,trainAug,testSet)