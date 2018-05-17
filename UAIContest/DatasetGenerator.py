import datetime as dt
import pickle
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import pandas as pd
import numpy as np

julyset = pd.read_csv('.\\Data\\train_July.csv')
augset = pd.read_csv('.\\Data\\train_Aug.csv')
testset = pd.read_csv('.\\Data\\test_id_Aug_agg_public5k.csv')
finalset = pd.read_csv('.\\Data\\test_id_Aug_agg_private5k.csv')

poi = pd.read_csv('.\\Data\\poi.csv',encoding = 'ansi', header=None, names = list(range(21)))
weather = pd.read_csv('.\\Data\\weather.csv',encoding = 'gb2312')

weather['date'] = weather['date'].map(lambda x:dt.datetime.strptime(x,'%Y/%m/%d %H:%M'))
weather['MyCode'] = weather['text'].map({'晴':8,'多云':7,'阴':6,'阵雨':5,'雷阵雨':4,'小雨':3,'中雨':2,'大雨':1})
#weather = weather.fillna(0)

trainset = pd.concat([julyset, augset])

#10 element
poiCatch = {}
def getPOI(id):
    if id in poiCatch:
        result = poiCatch[id]
    else:
        try:
            result = poi[poi[0] == id][[2,4,6,8,10,12,14,16,18,20]].values[0]
        except (IndexError,KeyError):
            result = np.zeros((1,10))
        poiCatch[id] = result
    return result

# 4 element
weatherCatch = {}
def getWeather(time, howManyHalfHour=4):
    result = np.array([])
    for j in range(howManyHalfHour):
        if time in weatherCatch:
            result.extend(weatherCatch[time])
        else:
            tmp = weather[(weather['date'] == time)][['MyCode','feels_like','wind_scale','humidity']]
            t = np.zeros((1,4))
            if tmp.shape[0] > 0:
                t = tmp.iloc[0]
            result = np.vstack((result,t))
            weatherCatch[time] = t
        time = time + dt.timedelta(minutes = 30)
    return result

ids={}
def hash(id,get=False):
    if id not in ids:
        if not get:
            ids[id]=len(ids)+1
        else:
            return -1
    return ids[id]

#[ poi of start, poi of end, weather of next 2 hour, weekday, hour ]
#| [count]
# 10 + 10 + 4*4 + 1 + 1 = 38 ;

# 不再生成验证集，用7月份训练，8月份测试
def GetEveryPairData(df):
    try:
        D = pickle.load(open('EveryPair.pkl','rb'))
    except FileNotFoundError:
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
    global ids
    try:
        f = pickle.load(open('Distance.pkl','rb'))
        ids = f['ids']
        dist = f['dist']
    except FileNotFoundError:
        g = df.groupby(['start_geo_id','end_geo_id'])['estimate_distance'].mean().reset_index(name='meanDist')
        L = len( pd.concat( [df['start_geo_id'],df['end_geo_id']], axis=0).unique() )
        dist = np.empty((L+1,L+1))
        dist.fill( np.inf )
        mat = g.values
        for i in range(len(mat)):
            s,t = hash(mat[i,0]),hash(mat[i,1])
            if i%10000==0:
                print('%s gen %d distance\n'%(dt.datetime.now(),i))
            dist[s,t] = dist[t,s] = mat[i,2]

        print('Running Floyd. total %d...\n'%L)
        for k in range(L):
            if k%100==0:
                print('%s floyd %d \n'%(dt.datetime.now(),k))
            for i in range(L):
                for j in range(L):
                    if dist[i,k]+dist[k,j]<dist[i,j]:
                        dist[i,j] = dist[i,k]+dist[k,j]
        pickle.dump({'dist':dist,'ids':ids},open('Distance.pkl','wb'))
    return D, dist

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

def GenBasicSet(df,dist):
    #基本特征
    df['dist'] = df[['start_geo_id','end_geo_id']].apply(lambda x: dist[hash(x[0],True),hash(x[1],True)] if hash(x[0],True)>0 and hash(x[1],True)>0 else np.inf , axis = 1)
    df['datetime'] = pd.to_datetime(df['create_date'] + ' ' + df['create_hour'].astype('str') + ':00')
    df['weekday'] = df['datetime'].map(lambda x: x.isoweekday())
    df['week'] = df['weekday'].map(lambda x: x//7+1)
    df['day'] = df['datetime'].map(lambda x: (x-dt.datetime(2017,6,30)).days)
    poiOfStart = df['start_geo_id'].apply(getPOI)
    poiOfEnd = df['end_geo_id'].apply(getPOI)
    wthr = df['datetime'].apply(getWeather)
    others = df[['weekday','day','week','create_hour','dist']]
    X=[]
    for i in range(len(df)):
        if i % 10000 == 0:
            print('%s proceed %i Samples\n' % (dt.datetime.now(),i))
        t = []
        t.extend(poiOfStart[i])
        t.extend(poiOfEnd[i])
        t.extend(wthr[i])
        t.extend(others[i])
        X.append(t)
    Tset = pd.DataFrame(columns = ['soil', 'smarket', 'suptown', 'ssubway', 'sbus', 'scaffee', 'schinese', 'satm', 'soffice', 'shotel',\
                                   'toil', 'tmarket', 'tuptown', 'tsubway', 'tbus', 'tcaffee', 'tchinese', 'tatm', 'toffice', 'thotel',\
                                   'MyCode0','feels_like0','wind_scale0','humidity0','MyCode1','feels_like1','wind_scale1','humidity1',\
                                   'MyCode2','feels_like2','wind_scale2','humidity2','MyCode3','feels_like3','wind_scale3','humidity3',\
                                   'weekday','day','week','hour','dist'\
                                   ],data = np.array(X))
    return Tset

def GenSSData(df,filename,ep,dist,forTrain = True):
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
    Tset = GenBasicSet(df,dist)
    ENset = GenENSet(df,ep)
    if forTrain:
        Tset = pd.concat([ df[['start_geo_id','end_geo_id','create_date','create_hour']], Tset, ENset, df['count'] ],axis = 1)
    else:
        Tset = pd.concat([df[['start_geo_id','end_geo_id','create_date','create_hour']], Tset, ENset], axis = 1)
    Tset.to_csv(filename + '.csv',index = False)
    return Tset

def SSSet():
    ep, dist = GetEveryPairData(trainset)
    Train = GenSSData(trainset,'SSTrain',ep,dist,True)
    Test = GenSSData(testset,'SSTest',ep,dist,False)
    FinalTest = GenSSData(finalset,'SSFinal',ep,dist,False)
    return Train,Test,FinalTest

