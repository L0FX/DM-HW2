# -*- coding: utf-8 -*-
"""DM-2.ipynb
数据挖掘互评作业二: 频繁模式与关联规则挖掘
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/content/data/MyDrive/data/DM-2/'

"""1.数据集“Amazon product co-purchasing network”的关联规则挖掘

（https://snap.stanford.edu/data/amazon0302.html）

"""

df = pd.read_csv(path+'processed_data1.csv')

"""为进行后续的频繁模式挖掘，现将数据中的group，salesrank，categories，totalreviews，avgrating提取出来为子数据集并转化为list"""

data = df[['avgrating','totalreviews','categories','salesrank','group']]
print(data.head(5))
arr = np.array(data)
d = arr.tolist()

"""

使用Apriori算法或FP-Growth算法，根据处理后的数据集计算频繁项集和关联规则。关联规则挖掘是一种常用的数据挖掘技术，可以从数据集中发现项集之间的关联关系。
"""

from apyori import apriori

result = list(apriori(transactions=d, min_support=0.1, min_confidence = 0.5))

def show_result(result):
  lastItems = []
  for i in result:
    tempItem = ';'.join(i.items)
    sup = round(i.support,4)
    for j in i.ordered_statistics:
      col = []
      tempItemBase = ';'.join(j.items_base)
      tempItemAdd = ';'.join(j.items_add)
      col.append(tempItem)
      col.append(sup)
      col.append(tempItemBase)
      col.append(tempItemAdd)
      col.append(round(j.confidence,4))
      col.append(round(j.lift,2))
      lastItems.append(col)
  lastItems = pd.DataFrame(lastItems)
  lastItems.columns=['LastItem','Support','ItemBase','ItemAdd','Confidence','Lift']
  lastItems.index = range(len(lastItems))
  print(len(lastItems))
  lastItems = lastItems.sort_values('Confidence',ascending = False).reset_index()
  print(lastItems.head(10))

  return lastItems

lastItems = show_result(result)

"""模式命名与分析

分析：根据上面的输出结果，以index为19的数据为例，存在“Music->[-,500000)”，即“音乐类别的产品->销售排名高于500000”，可以得出，音乐类别的产品的销售排名有约97%的概率高于500000，且这种情况的发生比例约为18.35%。其余数据同理可进行解释。

模式命名：关于产品的类别与评分、销售排名、被购买次数的关系，我们可以将其命名为产品推销模式。

可视化
"""

## 绘制支持度和置信度的散点图
def scatter(items,size):
  items.plot(kind="scatter",x = "Support",c = "r",
            y = "Confidence",s = size,figsize=(8,5))
  plt.grid("on")
  plt.xlabel("Support",size = 12)
  plt.ylabel("Confidence",size = 12)
  plt.title(f"Scatter plot of {size} rules")
  plt.show()

import networkx as nx

def network(items,size):
  plt.figure(figsize=(8,8))
  ## 生成社交网络图
  G=nx.DiGraph()

  draw_df = items[0:size]

  ## 为图像添加边
  for ii in draw_df.index:
    G.add_edge(draw_df.ItemBase[ii],draw_df.ItemAdd[ii],weight = draw_df.Confidence[ii])

  ## 定义2种边
  elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.6]
  emidle=[(u,v) for (u,v,d) in G.edges(data=True) if (d['weight'] <= 0.6)&(d['weight'] >= 0.45)]
  esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <= 0.45]

  ## 图的布局方式
  pos=nx.circular_layout(G)

  # 根据规则的置信度节点的大小
  nx.draw_networkx_nodes(G,pos,alpha=0.4)

  # 设置边的形式
  nx.draw_networkx_edges(G,pos,edgelist=elarge,
                      width=2,alpha=0.6,edge_color='r')
  nx.draw_networkx_edges(G,pos,edgelist=emidle,
                      width=2,alpha=0.6,edge_color='g',style='dashdot')
  nx.draw_networkx_edges(G,pos,edgelist=esmall,
                      width=2,alpha=0.6,edge_color='b',style='dashed')

  # 为节点添加标签
  nx.draw_networkx_labels(G,pos,font_size=10)

  plt.axis('off')
  plt.title(f"Sample network of {size} rules")
  plt.show()

scatter(lastItems,128)

network(lastItems,30)

"""2.数据集“MOOC User Action Dataset”的关联规则挖掘

（https://snap.stanford.edu/data/act-mooc.html）

"""

group_0 = pd.read_csv(path+'processed_group0.csv')
group_1 = pd.read_csv(path+'processed_group1.csv')
"""将数据转化为用于频繁项集挖掘的格式"""

g1 = group_1[['new_ACTIONID','new_TARGETID','TIMESTAMP']]
g0 = group_0[['new_ACTIONID','new_TARGETID','TIMESTAMP']]

arr1 = np.array(g1)
arr0 = np.array(g0)
d1 = arr1.tolist()
d0 = arr0.tolist()

"""频繁模式挖掘

a.针对LABEL=1部分数据
"""

result1 = list(apriori(transactions=d1, min_support=0.01, min_confidence = 0.2))

lastItems1 = show_result(result1)

"""a.针对LABEL=0部分数据"""

result0 = list(apriori(transactions=d0, min_support=0.01, min_confidence = 0.2))

lastItems0 = show_result(result0)

"""模式命名与分析

**分析：**

a.LABEL=1

根据上面的输出结果，可以发现，存在“TIMESTAMP[-,500000]->TARGETID_7,8,9,13,15,16,19”，即“在500000的时间节点前->目标网页为7,8，9,13,15,16,19”，其中，有达到53.16%的概率，在时间点500000前，用户的目标网页为9。其余数据同理可进行解释。

a.LABEL=0

根据上面的输出结果，可以发现，存在“TIMESTAMP[2000000,-]->TARGETID_79”，即“在2000000的时间节点后->目标网页为79”，可解释为，有达到82.78%的概率在2000000的时间节点后，用户的目标网页为79；以及，存在“TIMESTAMP[-,500000]->TARGETID_1,3,4,5,7,8,9，13,14”，即有高于45%的概率在时间点500000前，用户的目标网页为1,3,4,5，7，8,9,13，14。其余数据同理可进行解释。

联系两种label的结果，可以发现，无论用户是否在操作后的退出，在时间点500000前，用户均有可观的概率目标网页为7,8,9和13。

**模式命名：**

根据上面的分析，我们可以得出不同时间点用户的需求不同，即存在“时间偏好”的关系模式。

可视化
"""

# LABEL=1
# 绘制支持度和置信度的散点图
scatter(lastItems1,9)

network(lastItems1,9)

# LABEL=0
scatter(lastItems0,15)

network(lastItems0,15)