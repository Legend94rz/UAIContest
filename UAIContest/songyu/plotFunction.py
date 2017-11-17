# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:21:21 2017

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

meanPath = './xgbmodel/2.08.csv'
modelPath = './xgbmodel/prediction.csv'

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
    plotResult()