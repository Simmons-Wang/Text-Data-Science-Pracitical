from tqdm import tqdm
from stemming.porter2 import stem
import os
import math
import datetime
import calendar

os.chdir(r'C:\Users\Simmons\PycharmProjects\ttds')

with open('./input/tweetsclassification/tweets.test.txt', 'r', encoding='utf-8') as f:
    tws = f.readlines()


ops = {"+": (lambda x,y: x-y),"-": (lambda x,y: x+y)}

t1 = 'Sat 02 May 2015 19:54:36 +0530'
t2 = 'Fri 01 May 2015 13:54:36 -0000'

def transform_date(t1):
    t1_list = t1.split(' ')
    t1_list[2] = str(list(calendar.month_abbr).index(t1_list[2]))
    time_1 = '-'.join(t1_list[ 1:-1])
    tz = t1_list[-1]
    tzd = ops[tz[0]](datetime.timedelta(0), datetime.timedelta(hours=int(str(tz[1:3])), minutes=int(str(tz[3:]))))
    result = datetime.datetime.strptime(time_1, '%d-%m-%Y-%H:%M:%S') + tzd
    return result

def time_delta(t1, t2):
    time1 = transform_date(t1)
    time2 = transform_date(t2)
    delta = (time1 - time2).total_seconds()
    return abs(int(delta))


time_delta(t1, t2)


a1 = 1
a2 = 1


def feb(i):
    if i == 0:
        result = a1
    elif i == 1:
        result = a2

    else:
        result = feb(i-1) + feb(i-2)

    return result


lines = ['2015-07-19,1.08,Louisville',
'2015-08-16,1.05,RichmondNorfolk',
'2015-04-05,1.1,Orlando',
'2015-07-26,1.12,GrandRapids',
'2015-05-31,1.1,Atlanta'
]

import pandas as pd
import numpy as np

dataset = []

for line in lines:
    index = ['date', 'price', 'region']
    ser = pd.Series(line.split(','), index)
    dataset.append(ser)

dataset = pd.concat(dataset, axis=1)