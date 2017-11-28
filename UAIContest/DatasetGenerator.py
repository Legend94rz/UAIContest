import datetime as dt
import pickle
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import pandas as pd

train_july = '.\\Data\\train_July.csv'
train_aug = '.\\Data\\train_Aug.csv'
testFile = '.\\Data\\test_id_Aug_agg_public5k.csv'

julyset = pd.read_csv(train_july)
augset = pd.read_csv(train_aug)
testset = pd.read_csv(testFile)
trainset = pd.concat([julyset, augset])

finished = 0
def log_result(result):
    global finished
    finished = finished + 1
    if finished % 10 == 0:
        print("%s finished %d\n"%(dt.datetime.now(),finished))
def GenTrainingSet(filename, interface):
    try:
        dic = pickle.load(open(filename+'-train.pkl','rb'))
        return dic['X'],dic['Y']
    except IOError:
        pass
    print("Gening Training set...\n")
    global finished
    finished = 0
    result = []
    pool = Pool(cpu_count()-1)
    for i in range(len(testset)):
        result.append( interface(pool, tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']]) ) )
    pool.close()
    pool.join()
    #result = interface()
    X = [result[i].get()[0] for i in range(len(result))]
    Y = [result[i].get()[1] for i in range(len(result))]
    pickle.dump({'X':X,'Y':Y},open(filename+'-train.pkl','wb'))
    return X,Y
def GenTestSet(filename, interface):
    try:
        tdic = pickle.load(open(filename+'-test.pkl','rb'))
        return tdic['X']
    except IOError:
        pass
    print("Gening Test set...\n")
    global finished
    finished = 0
    result = []
    pool = Pool(cpu_count()-1)
    for i in range(len(testset)):
        result.append( interface( pool , tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']]) ) )
    pool.close()
    pool.join()
    #result = interface()
    pickle.dump({'X':TX},open(filename+'-test.pkl','wb'))
    TX = [result[i].get()[0] for i in range(len(result))]
    return TX

####    Gen for split 012       ###########
def Split012ForTrain(*x):
    feature = []
    Y=[]
    time = dt.datetime.strptime(x[2],'%Y-%m-%d')+dt.timedelta(hours=int(x[3]))
    timeL = time+dt.timedelta(hours=-1)
    timeR = time+dt.timedelta(hours=1)
    for k in range(3):
        tmpset = trainset[(trainset['start_geo_id']==x[0]) &( trainset['end_geo_id']==x[1]) &  (trainset['status']==k)]
        feature.append(len(tmpset[ (tmpset['create_hour']==timeL.hour) & (tmpset['create_date']==timeL.strftime('%Y-%m-%d'))]))
        feature.append(len(tmpset[ (tmpset['create_hour']==timeR.hour)& (tmpset['create_date']==timeR.strftime('%Y-%m-%d'))]))
        tmp = tmpset[tmpset['create_hour']==x[3]].groupby('create_date').size().reset_index(name='count')
        tmp = tmp[tmp['count']<=20]
        if len(tmp)>0:
            feature.append(tmp[tmp['count']<=20]['count'].mean())
        else:
            feature.append(0)
    return (feature,Y)
def Split012ForTest(*x):
    #x = p[1].loc[p[2],['start_geo_id','end_geo_id','create_date','create_hour']]
    feature = [(dt.datetime.strptime(x[2],'%Y-%m-%d') - dt.datetime(2017,7,1)).days*24 + x[3]]
    return feature
def Split012():
    def aplyTrain(pool,params):
        return pool.apply_async(Split012ForTrain,params,callback = log_result)
    def aplyTest(pool,params):
        return pool.apply_async(Split012ForTest,params,callback = log_result)
    X,Y = GenTrainingSet('split', aplyTrain)
    TX = GenTestSet('split',aplyTest)
    return X,Y,TX
###########################################

### feature is 3 vector of (31*24+7*12)
def SyntheSet(*x):
    feature = []
    Y = []
    tmpset = trainset[(trainset['start_geo_id']==x[0])&(trainset['end_geo_id']==x[1])]
    for k in range(3):
        t = [0 for i in range(31*24+7*12)]
        tmp = tmpset[tmpset['status']==k].groupby(['create_date','create_hour']).size().reset_index(name='count')
        for i in range(len(tmp)):
            days = (dt.datetime.strptime( tmp.loc[i,'create_date'],'%Y-%m-%d') - dt.datetime(2017,7,1)).days
            if days<31:
                t[days*24+tmp.loc[i,'create_hour']] = tmp.loc[i,'count']
            else:
                t[24*31+(days-31)*12+int( tmp.loc[i,'create_hour']//2 )]=tmp.loc[i,'count']
        feature.append(t)
    return (feature,Y)

def Synthe():
    def aply(pool,params):
        return pool.apply_async(SyntheSet,params,callback = log_result)
    X,Y = GenTrainingSet('synthe',aply)
    return X,Y,X


