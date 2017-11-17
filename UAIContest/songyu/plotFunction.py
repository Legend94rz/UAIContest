# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:21:21 2017

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import UAIGetFeature as UF
import UAIContest as UC
import numpy as np

meanPath = './xgbmodel/2.08.csv'
modelPath = './xgbmodel/prediction.csv'



def plotTest():
    
    a=[3,5,6,7,8,1]
    plt.figure()
    #plt.plot(result[num])
    plt.boxplot(a)
    #plt.savefig(dir+str(num)+'result.jpg')



def plotgetHourFeature():
    
    dir= './getAvgHourFeature/'
    result = UF.getHourFeature(31).values  
   
    #res.append(getridofOutlier(pd.Series(tmp[j])).tolist())
    
    print (len(result))
    #for num in range(len(result)):
    for num in range(10):
        plt.figure()
        plt.plot(result[num])
        print('\n',np.mean(result[num]))
        #print (result[num])
        res = []
        for i in range (24):
            #print (result[num][24*i:24*i+24])
            res.extend(UF.getridofOutlier(pd.Series(result[num][31*i:31*i+31])).tolist())
        print (np.mean(res))
        plt.plot(res,color = 'red')
        #plt.boxplot(result[num])
        plt.savefig(dir+str(num)+'result.jpg')



def plotResult():
    
    num=100
    dir= './FigResult/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for num in range (50):
        plt.figure()
        meanResult = pd.read_csv(meanPath)
        modelResult = pd.read_csv(modelPath)
        plt.plot(meanResult['count'].tolist()[num*100:num*100+100],color='red')     
        plt.plot(modelResult['count'].tolist()[num*100:num*100+100])
        plt.xlabel("count")
        plt.ylabel("test")
        plt.title(str(num))
        plt.savefig(dir+str(num)+'result.jpg')


if __name__=="__main__":
    #plotResult()
    #dfTest = UC.ReadTest()
    #plotgetHourFeature(dfTest)
    #plotTest()
    plotgetHourFeature()