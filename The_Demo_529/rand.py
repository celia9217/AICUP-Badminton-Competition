import os
import cv2
import csv
import numpy as np
import pandas as pd

# 無警告指令
"""
$ python -W ignore rand.py
"""

csvFile = './test.csv'
df = pd.read_csv(csvFile)

hitframe = df['HitFrame'].tolist()

hitterList = ['A', 'B']
roundheadList = [1, 2]
backhandList = [1, 2]
ballheightList = [1, 2]
balltypeList = [i for i in range(1,10)]

videoname = df['VideoName'].tolist()
for i in range(len(videoname)-1):
    name = videoname[i]
    if i == 0:
        h = np.random.choice(hitterList)    
    
    df['Hitter'][i] = h
    r, b1, b2, b3 = np.random.choice(roundheadList), np.random.choice(backhandList), np.random.choice(ballheightList), np.random.choice(balltypeList)
    df['RoundHead'][i] = r
    df['Backhand'][i] = b1
    df['BallHeight'][i] = b2
    df['BallType'][i] = b3


    # 打擊者交替
    if videoname[i+1] == name:
        hset = {'A', 'B'}
        hset.remove(h)
        h = list(hset)[0]
    else:
        h = np.random.choice(hitterList)
        wset = {'A', 'B'}
        wset.remove(h)
        df['Winner'][i] = list(wset)[0]

    if i == len(videoname) - 2:
        df['Hitter'][i+1] = h

    print(h, r, b1, b2, b3)

    # 羽球軌跡
    ballcsv = './predict/' + name.split('.')[0] + '.csv'
    dfBall = pd.read_csv(ballcsv)
    try:
        df['LandingX'][i] = round(dfBall['X'].tolist()[hitframe[i]])
        df['LandingY'][i] = round(dfBall['Y'].tolist()[hitframe[i]])
    except:
        pass

# 最後一筆資料
h = df['Hitter'][len(videoname)-1]
wset = {'A', 'B'}
wset.remove(h)
df['Winner'][len(videoname)-1] = list(wset)[0]
df['LandingX'][len(videoname)-1] = round(dfBall['X'].tolist()[hitframe[len(videoname)-1]])
df['LandingY'][len(videoname)-1] = round(dfBall['Y'].tolist()[hitframe[len(videoname)-1]])

df.to_csv('./test_rand.csv', index=False)