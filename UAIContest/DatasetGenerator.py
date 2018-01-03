import datetime as dt
import pickle
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import pandas as pd
import numpy as np

julyset = pd.read_csv('.\\Data\\train_July.csv')
augset = pd.read_csv('.\\Data\\train_Aug.csv')
testset = pd.read_csv('.\\Data\\test_id_Aug_agg_public5k.csv')
poi = pd.read_csv('.\\Data\\poi.csv',encoding = 'ansi', header=None, names = list(range(21)))
weather = pd.read_csv('.\\Data\\weather.csv',encoding = 'gb2312')

weather['date'] = weather['date'].map(lambda x:dt.datetime.strptime(x,'%Y/%m/%d %H:%M'))
weather['MyCode'] = weather['text'].map({'晴':8,'多云':7,'阴':6,'阵雨':5,'雷阵雨':4,'小雨':3,'中雨':2,'大雨':1})
weather = weather.fillna(0)

trainset = pd.concat([julyset, augset])
### tool kits
finished = 0
def log_result(result):
    global finished
    finished = finished + 1
    if finished % 100 == 0:
        print("%s finished %d\n" % (dt.datetime.now(),finished))
def GenTrainingSet(filename, interface):
    try:
        dic = pickle.load(open(filename + '-train.pkl','rb'))
        return dic['X'],dic['Y']
    except IOError:
        pass
    print("Gening Training set...\n")
    global finished
    finished = 0
    result = []
    pool = Pool(cpu_count() - 1)
    for i in range(len(testset)):
        q = interface(pool, tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']])) 
        if q != None:
            result.append(q)
    pool.close()
    pool.join()
    X = [result[i].get()[0] for i in range(len(result))]
    Y = [result[i].get()[1] for i in range(len(result))]
    pickle.dump({'X':X,'Y':Y},open(filename + '-train.pkl','wb'))
    return X,Y
def GenTestSet(filename, interface):
    try:
        tdic = pickle.load(open(filename + '-test.pkl','rb'))
        return tdic['X']
    except IOError:
        pass
    print("Gening Test set...\n")
    global finished
    finished = 0
    result = []
    pool = Pool(cpu_count() - 1)
    for i in range(len(testset)):
        q = interface(pool , tuple(testset.loc[i,['start_geo_id','end_geo_id','create_date','create_hour']])) 
        if q != None:
            result.append(q)

    pool.close()
    pool.join()
    TX = [result[i].get()[0] for i in range(len(result))]
    pickle.dump({'X':TX},open(filename + '-test.pkl','wb'))
    return TX

#10 element
poiCatch = {}
def getPOI(id):
    if id in poiCatch:
        result = poiCatch[id]
    else:
        try:
            result = list(poi[poi[0] == id][[2,4,6,8,10,12,14,16,18,20]].values[0])
        except (IndexError,KeyError):
            result = [0 for i in range(10)]
        poiCatch[id] = result
    return result

# 4 element
weatherCatch = {}
def getWeather(time, howManyHalfHour=4):
    result = []
    for j in range(howManyHalfHour):
        if time in weatherCatch:
            result.extend(weatherCatch[time])
        else:
            tmp = weather[(weather['date'] == time)][['MyCode','feels_like','wind_scale','humidity']]
            t = [0 for i in range(4)]
            if tmp.shape[0] > 0:
                t = list(tmp.iloc[0])
            result.extend(t)
            weatherCatch[time] = t
        time = time + dt.timedelta(minutes = 30)
    return result

def flatten(l):    
    for el in l:    
        if hasattr(el, "__iter__") and not isinstance(el, str):    
            for sub in flatten(el):    
                yield sub    
        else:    
            yield el   


####    Gen for split 012 ###########
def Split012ForTrain(*x):
    feature = []
    Y = []
    time = dt.datetime.strptime(x[2],'%Y-%m-%d') + dt.timedelta(hours=int(x[3]))
    timeL = time + dt.timedelta(hours=-1)
    timeR = time + dt.timedelta(hours=1)
    for k in range(3):
        tmpset = trainset[(trainset['start_geo_id'] == x[0]) & (trainset['end_geo_id'] == x[1]) & (trainset['status'] == k)]
        feature.append(len(tmpset[(tmpset['create_hour'] == timeL.hour) & (tmpset['create_date'] == timeL.strftime('%Y-%m-%d'))]))
        feature.append(len(tmpset[(tmpset['create_hour'] == timeR.hour) & (tmpset['create_date'] == timeR.strftime('%Y-%m-%d'))]))
        tmp = tmpset[tmpset['create_hour'] == x[3]].groupby('create_date').size().reset_index(name='count')
        tmp = tmp[tmp['count'] <= 20]
        if len(tmp) > 0:
            feature.append(tmp[tmp['count'] <= 20]['count'].mean())
        else:
            feature.append(0)
    return (feature,Y)
def Split012ForTest(*x):
    #x =
    #p[1].loc[p[2],['start_geo_id','end_geo_id','create_date','create_hour']]
    feature = [(dt.datetime.strptime(x[2],'%Y-%m-%d') - dt.datetime(2017,7,1)).days * 24 + x[3]]
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
    tmpset = trainset[(trainset['start_geo_id'] == x[0]) & (trainset['end_geo_id'] == x[1])]
    for k in range(3):
        t = [0 for i in range(31 * 24 + 7 * 12)]
        tmp = tmpset[tmpset['status'] == k].groupby(['create_date','create_hour']).size().reset_index(name='count')
        for i in range(len(tmp)):
            days = (dt.datetime.strptime(tmp.loc[i,'create_date'],'%Y-%m-%d') - dt.datetime(2017,7,1)).days
            if days < 31:
                t[days * 24 + tmp.loc[i,'create_hour']] = tmp.loc[i,'count']
            else:
                t[24 * 31 + (days - 31) * 12 + int(tmp.loc[i,'create_hour'] // 2)] = tmp.loc[i,'count']
        feature.append(t)
    return (feature,Y)

def Synthe():
    def aply(pool,params):
        return pool.apply_async(SyntheSet,params,callback = log_result)
    X,Y = GenTrainingSet('synthe',aply)
    return X,Y,X


### for outlier ###
def OutlierForTraining(*x):
    start = x[0]
    end = x[1]
    tmp = trainset[(trainset['start_geo_id'] == start) & (trainset['end_geo_id'] == end)].groupby(['create_date','create_hour']).size().reset_index(name='count')
    left,right,mid,time = [],[],[],[]
    for i in range(1,len(tmp) - 1):
        if tmp.loc[i,'count'] > 3 * tmp.loc[i - 1,'count'] and tmp.loc[i,'count'] > 3 * tmp.loc[i + 1,'count']:
            left.append(tmp.loc[i - 1,'count'])
            right.append(tmp.loc[i + 1,'count'])
            mid.append(tmp.loc[i,'count'])
            D = dt.datetime.strptime(tmp.loc[i,'create_date'],'%Y-%m-%d') + dt.timedelta(hours = int(tmp.loc[i,'create_hour']))
            time.append(D)

    feature = []
    Y = []
    for i in range(len(time)):
        feature.append([e for e in flatten([getPOI(start), getPOI(end), -1,  getWeather(time[i] + dt.timedelta(hours=-1),1),  mid[i]])])
        feature.append([e for e in flatten([getPOI(start), getPOI(end), +1,  getWeather(time[i] + dt.timedelta(hours=+1),1),  mid[i]])])
        Y.append(left[i])
        Y.append(right[i])
    return (feature,Y)

def OutlierForTest(*x):
    start = x[0]
    end = x[1]
    feature = []
    m = trainset[(trainset['start_geo_id'] == start) & (trainset['end_geo_id'] == end) & (trainset['create_date'] == x[2]) & (trainset['create_hour'] == 21)].shape[0]
    if(x[3] == 20):
        feature.append([e for e in flatten([getPOI(start), getPOI(end), -1,  getWeather('2017-08-02 20:00',1),  m])])
    else:
        feature.append([e for e in flatten([getPOI(start), getPOI(end), +1,  getWeather('2017-08-02 22:00',1),  m])])
    return feature

def OutlierSet():
    def aplyForTrain(pool,params):
        return pool.apply_async(OutlierForTraining,params,callback=log_result)
    def aplyForTest(pool,params):
        if params[2] == '2017-08-02' and (params[3] == 20 or params[3] == 22):
            return pool.apply_async(OutlierForTest,params,callback = log_result)
        else:
            return None
    X,Y = GenTrainingSet('Outlier',aplyForTrain)
    X = [X[i][j] for i in range(len(X)) for j in range(len(X[i]))]
    Y = [y for y in flatten(Y)]
    TX = GenTestSet('Outlier',aplyForTest)
    return X,Y,TX


#[ poi of start, poi of end, weather of next 2 hour, weekday, hour ]
#| [count]
# 10 + 10 + 4*4 + 1 + 1 = 38 ;

# 不再生成验证集，用7月份训练，8月份测试
def GetEveryPairData(df):
    try:
        return pickle.load(open('EveryPair.pkl','rb'))
    except FileNotFoundError:
        pass
    D = {}
    X = []
    g = df.groupby(['start_geo_id','end_geo_id','create_date','create_hour']).size().reset_index(name='count')
    mat = g.values
    i=0
    while i < len(mat):
        s,t = mat[i,0],mat[i,1]
        if i%10000==0:
            print('%s gen %d neighboors\n'%(dt.datetime.now(),i))
        tmp = np.zeros(40*24)
        j=i
        while j<len(mat) and mat[j,0]==s and mat[j,1]==t:
            tmp[ (dt.datetime.strptime( mat[j,2] ,'%Y-%m-%d' )-dt.datetime(2017,6,30)).days * 24 + mat[j,3] ] = mat[j,-1]
            j = j + 1
        tmp[0:24] = tmp[24:48]  #6.30 <- 7.1
        tmp[-24:] = tmp[-48:-24]# 8.8 <- 8.7
        D[s+t] = tmp
        i=j
    pickle.dump(D,open('EveryPair.pkl','wb'))
    return D

def GetEstimate(ep,start,end,date,hour,rng = 2):
    key = start+end
    if key in ep:
        tmp = ep[key]
        date = dt.datetime.strptime( date ,'%Y-%m-%d' )
        days = (date-dt.datetime(2017,6,30)).days
        pos = days * 24 + hour
        ind = pos + 2*np.array(range(-rng//2,rng//2))+1
        nei = list(tmp[ind])

        r = tmp.reshape((-1,24))
        meanOnHurByDay = r[1:-1,hour].sum()/(34 + (hour+1)%2)

        meanOnPreByDay = r[1:-1,(hour-1)%24].sum()/(34 + (hour)%2)

        meanOnNxtByDay = r[1:-1,(hour+1)%24].sum()/(34 + (hour)%2)

        meanOnHurByWeek = r[(days-1)%7::7].sum() / ( 5+( days%7>=1 and days%7<=3 ) - (date.day%2 == hour%2) )

        return [np.ceil( meanOnHurByDay*0.6 + (meanOnNxtByDay+meanOnPreByDay)*0.2 ), meanOnHurByDay, meanOnHurByWeek] + nei
    else:
        return [0,0,0]+[0]*rng

def GenENSet(df,ep):
    #估计值、近邻特征
    estimate = []
    mat = df.values
    for i in range(len(mat)):
        estimate.append(GetEstimate(ep,mat[i,0],mat[i,1],mat[i,2],mat[i,3]))
    return pd.DataFrame(np.array(estimate),columns = ['estimate','hisMean','weekMean','-1','1'])

def GenBasicSet(df):
    #基本特征
    df['datetime'] = pd.to_datetime(df['create_date'] + ' ' + df['create_hour'].astype('str') + ':00')
    df['weekday'] = df['datetime'].map(lambda x: x.isoweekday())
    df['week'] = df['weekday'].map(lambda x: x//7+1)
    df['day'] = df['datetime'].map(lambda x: (x-dt.datetime(2017,6,30)).days)
    poiOfStart = np.array(df['start_geo_id'].apply(getPOI))
    poiOfEnd = np.array(df['end_geo_id'].apply(getPOI))
    wthr = np.array(df['datetime'].apply(getWeather))
    dayAndHur = np.array(df[['weekday','day','week','create_hour']])
    X=[]
    for i in range(len(df)):
        if i % 10000 == 0:
            print('%s proceed %i Samples\n' % (dt.datetime.now(),i))
        t = []
        t.extend(poiOfStart[i])
        t.extend(poiOfEnd[i])
        t.extend(wthr[i])
        t.extend(dayAndHur[i])
        X.append(t)
    Tset = pd.DataFrame(columns = ['soil', 'smarket', 'suptown', 'ssubway', 'sbus', 'scaffee', 'schinese', 'satm', 'soffice', 'shotel',\
                                   'toil', 'tmarket', 'tuptown', 'tsubway', 'tbus', 'tcaffee', 'tchinese', 'tatm', 'toffice', 'thotel',\
                                   'MyCode0','feels_like0','wind_scale0','humidity0','MyCode1','feels_like1','wind_scale1','humidity1',\
                                   'MyCode2','feels_like2','wind_scale2','humidity2','MyCode3','feels_like3','wind_scale3','humidity3',\
                                   'week','weekday','day','hour'\
                                   ],data = np.array(X))
    return Tset

def GenSSData(df,filename,ep,forTrain = True):
    try:
        return pd.read_csv(filename+'.csv')
    except FileNotFoundError:
        pass

    if forTrain:
        zeros = 0;  month = 7;
        df = df.groupby(['start_geo_id','end_geo_id','create_date','create_hour']).size().reset_index(name='count')
        #生成0数据
        '''
        zeroNum = 80000
        zeroData = []
        while zeroNum < zeros:
            if (zeroNum + 1) % 5000 == 0:
                print('%s gened %d rand Samples\n' % (dt.datetime.now(),zeroNum + 1))
            if month == 7:
                randDate = dt.datetime(2017,7,np.random.randint(1,32)).strftime('%Y-%m-%d')
                randHur = np.random.randint(0,24)
            else:
                d = np.random.randint(1,8)
                randHur = np.random.randint(0,12) * 2 + (d%2==0)
                randDate = dt.datetime(2017,8,d).strftime('%Y-%m-%d')
            zeroData.append([np.random.randint(0,len(df)),np.random.randint(0,len(df)),randDate,randHur  ])
            zeroNum = zeroNum + 1
        zeroData = np.array(zeroData)
        Zset = pd.DataFrame(columns = ['start_geo_id','end_geo_id','create_date','create_hour'])
        Zset['start_geo_id'] = df.loc[zeroData[:,0].astype(int), 'start_geo_id'].values
        Zset['end_geo_id'] = df.loc[zeroData[:,1].astype(int), 'end_geo_id'].values
        Zset['create_date'] = zeroData[:,2]
        Zset['create_hour'] = zeroData[:,3].astype(int)
        df = df.merge(how='outer',right = Zset,on = ['start_geo_id','end_geo_id','create_date','create_hour']).fillna(0) #合并随机0数据
        '''
        #now df is : start, end, date, hour, count
    else:
        df = df.drop(['test_id'],axis = 1)
    ENset = GenENSet(df,ep)
    Tset = GenBasicSet(df)
    if forTrain:
        Tset = pd.concat([ df[['start_geo_id','end_geo_id','create_date','create_hour']], Tset, ENset, df['count'] ],axis = 1)
    else:
        Tset = pd.concat([df[['start_geo_id','end_geo_id','create_date','create_hour']], Tset, ENset], axis = 1)
    Tset.to_csv(filename + '.csv',index = False)
    return Tset

def SSSet():
    ep = GetEveryPairData(trainset)
    Train = GenSSData(trainset,'SSTrain',ep,True)
    Test = GenSSData(testset,'SSTest',ep,False)
    return Train,Test

