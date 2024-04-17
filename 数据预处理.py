# -*- coding: utf-8 -*-
"""DM-2.ipynb

数据挖掘互评作业二: 频繁模式与关联规则挖掘
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/content/data/MyDrive/data/DM-2/'

"""1.数据集“Amazon product co-purchasing network”的预处理

（https://snap.stanford.edu/data/amazon0302.html）

"""

# 将txt文件转为csv文件
with open(path+'amazon-meta.txt', 'r', encoding='utf-8') as F:
  content=F.readlines()
content=[x.strip()for x in content]

file= open(path+'amazon_clean_meta.txt','w',encoding='utf-8')
all_row=['Id','title','group','salesrank','categories','totalreviews','avgrating'] # Column names

for line in content:
  lines = line.split(':')
  if lines[0]=='Id':
    if (len(all_row)==7):
      for comp in all_row[0:6]:
        file.write(comp)
        file.write(',')

      file.write(all_row[6])
      file.write('\n')
      all_row=[]
      all_row.append(lines[1].strip())
  if lines[0]=='title':
    title=':'.join(lines[1:]).strip().replace(',',' ').replace('\n',' ').strip()
    all_row.append(title)
  if lines[0]=='group':
    all_row.append(lines[1].strip())
  if lines[0]=='salesrank':
    all_row.append(lines[1].strip())
  if lines[0]=='categories':
    all_row.append(lines[1].strip())
  if lines[0]=='reviews' and lines[1].strip()=='total':
    all_row.append(lines[2].split(' ')[1])
    all_row.append(lines[4].strip())

file.close()

meta = pd.read_csv(path+'amazon_clean_meta.txt',sep=',')
meta['Id'].iloc[0] = 1
meta.to_csv(path+'amazon_meta.csv',index=False)

#数据简要展示
df = pd.read_csv(path+'amazon_meta.csv')
print(df.head(5))
print(df.dtypes)

"""此处,Id为产品编号；title为产品名字；group为产品种类，包括Book、DVD、Video、Music或其他；salesrank为商品在Amazon的销售排名；categories为产品所属的产品类别层次结构中的位置；totalreviews为产品被购买数；avgrating为产品评价评分。"""

#检查数据缺失值,发现无缺失值
print(df.isnull().any())
#删去重复数据
print(f'原始数据长度：{len(df)}')
df_nodup = df.drop_duplicates()
print(f'去重后数据长度：{len(df_nodup)}')

"""对数据进行一系列探索性分析："""

numeric = ['salesrank','categories','totalreviews','avgrating']
print(df[numeric].describe().loc[['max', '75%', '50%', '25%', 'min']])

"""为方便后续挖掘关联规则，将数值型属性根据数值划分范围："""

def trans_avgrating(row):
  new = []
  if 0 <= row['avgrating'] < 1:
    new.append('[0,1)')
  elif 1 <= row['avgrating'] < 2:
    new.append('[1,2)')
  elif 2 <= row['avgrating'] < 3:
    new.append('[2,3)')
  elif 3 <= row['avgrating'] < 4:
    new.append('[3,4)')
  elif 4 <= row['avgrating']:
    new.append('[4,5]')
  return new

def trans_reviews(row):
  new = []
  if 0 <= row['totalreviews'] < 5:
    new.append('[0,5)')
  elif 5 <= row['totalreviews'] < 10:
    new.append('[5,10)')
  elif 10 <= row['totalreviews'] < 1000:
    new.append('[10,1000)')
  elif 1000 <= row['totalreviews'] < 3000:
    new.append('[1000,3000)')
  elif 3000 <= row['totalreviews']:
    new.append('[3000,-)')
  return new

def trans_categories(row):
  new = []
  if 0 <= row['categories'] < 2:
    new.append('[0,2)')
  elif 2 <= row['categories'] < 5:
    new.append('[2,5)')
  elif 5 <= row['categories'] < 10:
    new.append('[5,10)')
  elif 10 <= row['categories'] < 50:
    new.append('[10,50)')
  elif 50 <= row['categories']:
    new.append('[50,-)')
  return new

def trans_salesrank(row):
  new = []
  if row['salesrank'] < 500000:
    new.append('[-,500000)')
  elif 500000 <= row['salesrank'] < 1000000:
    new.append('[500000,1000000)')
  elif 1000000 <= row['salesrank'] < 1500000:
    new.append('[1000000,1500000)')
  elif 1500000 <= row['salesrank'] < 2000000:
    new.append('[1500000,2000000)')
  elif 2000000 <= row['salesrank']:
    new.append('[2000000,-)')
  return new

new_rating = []
new_reviews = []
new_rank = []
new_cat = []
for _, row in df.iterrows():
  new_rating.extend(trans_avgrating(row))
  new_reviews.extend(trans_reviews(row))
  new_rank.extend(trans_salesrank(row))
  new_cat.extend(trans_categories(row))
df['avgrating'] = new_rating
df['totalreviews'] = new_reviews
df['categories'] = new_cat
df['salesrank'] = new_rank
df.head(5)

df.to_csv(path+'processed_data1.csv',index=False)


"""2.数据集“MOOC User Action Dataset”的预处理

（https://snap.stanford.edu/data/act-mooc.html）

"""

actions = pd.read_csv(path+'mooc_actions.tsv',sep='\t')
features = pd.read_csv(path+'mooc_action_features.tsv',sep='\t')
labels = pd.read_csv(path+'mooc_action_labels.tsv',sep='\t')
print(actions.head(10))
print(features.head(10))
print(labels.head(10))

"""此处变量的意义为：

ACTIONID:每个操作的唯一id。

USERID:每个用户的唯一id。

TARGETID:每个目标活动的唯一id。

TIMESTAMP:动作的时间戳，以秒为单位。

FEATUREx:与动作相关联的特征值。总共四个，使其成为一个四维特征向量。

LABEL:一个二进制标签，指示学生在操作后是否退出。退出操作的值为1，否则为0。
"""

# 将三个表合并
act = pd.merge(actions,features,on='ACTIONID')
act = pd.merge(act,labels,on='ACTIONID')
print(act.head())
print(act.dtypes)

# 展示数据情况
for col in ['ACTIONID','USERID','TARGETID','LABEL']:
  print(act[col].value_counts())

print(act[['TIMESTAMP','FEATURE0','FEATURE1','FEATURE2','FEATURE3']].describe().loc[['max', '75%', '50%', '25%', 'min']])

"""将TIMESTAMP,ACTIONID,TARGETID,FEATUREx,LABEL转化为适合用于频繁项集与关联规则挖掘的格式"""

# 将TIMESTAMP特征根据数值划分范围
def trans_timestamp(row):
  new = []
  if row['TIMESTAMP'] < 700000:
    new.append('[-,500000)')
  elif 500000 <= row['TIMESTAMP'] < 1000000:
    new.append('[500000,1000000)')
  elif 1000000 <= row['TIMESTAMP'] < 1500000:
    new.append('[1000000,1500000)')
  elif 1500000 <= row['TIMESTAMP'] < 2000000:
    new.append('[1500000,2000000)')
  elif 2000000 <= row['TIMESTAMP']:
    new.append('[2000000,-)')
  return new
new = []
for _,row in act.iterrows():
  trans = trans_timestamp(row)
  new.extend(trans)
act['TIMESTAMP'] = new

# 将属性值转化为字符串
act['ACTIONID'] = act['ACTIONID'].astype('string')
act['TARGETID'] = act['TARGETID'].astype('string')
act['LABEL'] = act['LABEL'].astype('string')
act['FEATURE0'] = act['FEATURE0'].astype('string')
act['FEATURE1'] = act['FEATURE1'].astype('string')
act['FEATURE2'] = act['FEATURE2'].astype('string')
act['FEATURE3'] = act['FEATURE3'].astype('string')
act['USERID'] = act['USERID'].astype('string')
print(act.dtypes)

# 将FEATURE0,FEATURE1,FEATURE2,FEATURE3合并
act['FEATURES'] = act['FEATURE0'] + ',' + act['FEATURE1'] + ',' + act['FEATURE2'] + ',' + act['FEATURE3']

act['new_TARGETID'] = 'TARGETID_' + act['TARGETID']
act['new_ACTIONID'] = 'ACTIONID_' + act['ACTIONID']
act['new_LABEL'] = 'LABEL_' + act['LABEL']

print(act.head())

"""为方便后续频繁项集与关联规则挖掘，在经过初步实验后，发现LABEL属性值的分布有着巨大差异，所以将原数据集根据LABEL属性的值（0或1）划分为两个数据集，不干扰后续分析"""

labels_group = act.groupby('new_LABEL')
group_0 = labels_group.get_group('LABEL_0')
#group_0

group_1 = labels_group.get_group('LABEL_1')
#group_1

group_1.to_csv(path+'processed_group1.csv',index=False)
group_0.to_csv(path+'processed_group0.csv',index=False)