import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import numpy as np


#################################################################################################
# https://www.kaggle.com/unsdsn/world-happiness/version/2
def createHappyData():
    happiness2017 = pd.read_csv('2017.csv')
    countries = pd.read_csv('countries.csv')
    result = happiness2017.set_index('Country').join(countries.set_index('Country'))
    result.to_csv('HappyLatLong.csv')

def plotH(displayNames=False):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    print('hi')
    HappyCountries = pd.read_csv('HappyLatLong.csv')
    # HappyCountries.plot.scatter(x='lat', y='long', s=HappyCountries['Happiness.Rank']*1);
    HappyCountries.plot.scatter(x='long', y='lat', c='Happiness.Score', s=50);
    # HappyCountries.plot.scatter(x='long', y='lat');
    if displayNames:
        for index, row in HappyCountries.iterrows():
            plt.annotate(HappyCountries['Country'][index], (row['long'], row['lat']))
    plt.show()

def plotHB():
    HappyCountries = pd.read_csv('HappyLatLong.csv')
    plt.scatter(HappyCountries['long'], HappyCountries['lat'], c='b')
    idx = np.random.choice(len(HappyCountries),int(len(HappyCountries) * 0.2), replace=False)
    plt.scatter(HappyCountries['long'][idx], HappyCountries['lat'][idx],c='r')
    plt.show()


def plotDenseCircle():
    x=np.linspace(-2, 2, num=100)
    y = np.linspace(-2, 2, num=100)
    # noise = np.random.normal(0, 0.2, x.shape)
    # x=x + noise
    # noise = np.random.normal(0, 0.2, x.shape)
    # y=y+noise

    x_s = []
    x_c = []
    y_s = []
    y_c = []
    for t in x:
        for k in y:
            if t*t + k*k < 1:
                x_c.append(t)
                y_c.append(k)
            else:
                x_s.append(t)
                y_s.append(k)
    plt.scatter(x_c, y_c, c='b',s=0.1)
    plt.scatter(x_s, y_s, c='r',s=0.1)
    plt.show()

def plotDenseCircleNonBalance():
    x=np.append(np.linspace(-2, 2, num=50), np.linspace(0.1, 1.84, num=100))
    y = np.append(np.linspace(-2, 2, num=50), np.linspace(0.1, 1.84, num=100))
    # noise = np.random.normal(0, 0.2, x.shape)
    # x=x + noise
    # noise = np.random.normal(0, 0.2, x.shape)
    # y=y+noise

    x_s = []
    x_c = []
    y_s = []
    y_c = []
    for t in x:
        for k in y:
            if t*t + k*k < 1:
                x_c.append(t)
                y_c.append(k)
            else:
                x_s.append(t)
                y_s.append(k)
    plt.scatter(x_c, y_c, c='b',s=0.1)
    plt.scatter(x_s, y_s, c='r',s=0.1)
    plt.show()
#################################################################################################