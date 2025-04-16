#!/usr/bin/env python
# coding: utf-8

# In[51]:


from IPython.display import Image
from igraph import *
import pandas as pd
import numpy as np


# In[52]:


#打开文件
r = pd.read_pickle("D:/STUDY/互联网与社会研究/作业/互联网与社会2020.pkl")
dat = r.loc[:, [3,6]]
dat.rename(columns = {3:'user',6:'generator'},inplace = True)
dat.loc[(dat.generator.isnull()),'generator'] = 1


# In[53]:


#建立从转发者到生产者的网络链接元组
reposter = dat['user'].values.tolist()
producer = dat['generator'].values.tolist()
link = []
for i in range(0,92636):
    if producer[i] != 1:
        tuple1 = (producer[i],reposter[i])
        link.append(tuple1)
len(link)


# In[54]:


newlink = list(set(link))
len(newlink)


# In[55]:


#所需数据转换为列表,newreposter100个
tuplereposter = [link[i][1] for i in range(len(link))]
tupleproducer = [link[i][0] for i in range(len(link))]
newreposter = []
newproducer = []
for each in reposter:
    if each in tuplereposter:
        newreposter.append(each)
for each in producer:
    if each in tupleproducer:
        newproducer.append(each)
newreposter = list(set(newreposter))
newproducer = list(set(newproducer))


# In[56]:


len(newreposter)


# In[57]:


g = Graph(directed=True) ## this is a directed network


# In[58]:


#添加节点、链接
g.add_vertices(newreposter)
g.add_vertices(newproducer)


# In[59]:


g.add_edges(newlink)
print(g.summary())


# In[62]:


#删除度小于等于10的节点,g.delete_vertices(list of index)
for v in g.vs:
        if g.degree(v) <= 10:
            g.delete_vertices(v)


# In[63]:


sorted(g.degree())


# In[64]:


print(g.summary())


# In[65]:


#添加属性，建立节点/链接与节点/链接性质一对一的字典

#节点属性1：转发者or被转发者,先生成节点：属性对应的字典，再get节点对应的属性并加入网络
dicV_character = {}
for i in newreposter:
    dicV_character[i] = 0
for i in newproducer:
    dicV_character[i] = 1
        
listV_character = [dicV_character.get(v) for v in g.vs['name']]
g.vs['character'] = listV_character
for v in g.vs:
    print(v)
    break


# In[66]:


#节点属性2：节点大小与节点中心性相关
vsize = utils.rescale(g.degree(), [1, 5])
g.vs['size'] = [vsize[i]*6 for i in range(len(vsize))]


# In[67]:


#链接属性1：链接名字
namelist = []
dicE_name = {}
for e in g.es:
    ename = (g.vs[e.source]['name'], g.vs[e.target]['name'])
    namelist.append(ename)
    for i in namelist:
        dicE_name[i] = i
listE_name = [dicE_name.get((g.vs[e.source]['name'], g.vs[e.target]['name'])) for e in g.es]
g.es['name'] = listE_name


# In[68]:


#链接属性2：以转发次数作为关系强度（设置为weight属性）
dicE_weight = {}
for each in dicE_name:
    dicE_weight[each] = namelist.count(each)
listE_weight = [dicE_weight.get(e['name']) for e in g.es]
g.es['weight'] = listE_weight


# In[69]:


#链接属性3：链接宽度与关系强度相关
dicE_width = {}
ewidth = utils.rescale(listE_weight, [1, 2])
for e in g.es:
    for i in range(len(ewidth)):
        dicE_width[e['name']] = ewidth[i]
listE_width = [dicE_width.get(e['name']) for e in g.es]
g.es['width'] = listE_width


# In[70]:


#设置箭头大小
g.es['arrow_size'] = 0.5
g.es['arrow_width'] = 0.5


# In[79]:


for v in g.vs:
     if g.degree(v) == 0:
        g.delete_vertices(v)


# In[80]:


print(g.summary())


# In[81]:


g['layout'] = g.layout_fruchterman_reingold()


# In[82]:


plot(g).show()


# In[102]:


cps = g.components('WEAK')#弱成分的分析
g = cps.giant() # 各成分的规模
print(cps.summary())


# In[103]:


#社群分析，一共21个社群
sgcom = g.community_spinglass()
print(sgcom.summary()) ## summary of the community structure
sgcom.membership ## membership of each node
g.vs['community'] = sgcom.membership


# In[104]:


#设置颜色透明度函数
def col_to_rgba(col_name, alpha=0.5):
    return color_name_to_rgb(col_name) + (alpha,) 
## 注意alpha控制透明程度，alpha后面带有逗号，保证是一个tuple，从而可以连接
## 必须使用return 命令返回结果


# In[105]:


ncluster = max(g.vs['community']) + 1 #共多少社群

c4c = ClusterColoringPalette(ncluster)

for v in g.vs:
    v['color'] = c4c.get(v['community'])
    v['frame_color'] = c4c.get(v['community'])
    
for e in g.es:
    e['color'] = g.vs[e.source]['color']
    
g.vs['color'] = [col_to_rgba(v, 0.5) for v in g.vs['color']]


# In[106]:


#应用颜色透明度函数
g.vs['color'] = [col_to_rgba(v, 0.5) for v in g.vs['color']]
g.vs['frame_color'] = [col_to_rgba(v, 0.5) for v in g.vs['frame_color']]
g.es['color'] = [col_to_rgba(e, 0.5) for e in g.es['color']]


# In[107]:


#创建标记每个社群中心性最高的节点的函数
def gettop(n,graph):
    dicrank = {}
    toplist = []
    for v in graph.vs:
        if v['community'] == n:
            tuple2 = (v.degree(mode = 'OUT'),v['name'],v['community'])
            toplist.append(tuple2)
    return toplist


# In[112]:


toplist = []
for i in range(23):
    topinfo = sorted(gettop(i,g),reverse = True)
    toplist.append(topinfo[0:3])

topnamelist = []
for each in toplist:
    for i in each:
        topnamelist.append(i[1])
print(toplist)


# In[109]:


for v in g.vs:
    if v['name'] in topnamelist:
        v['label'] = v['name']
        v['label_size'] = 10


# In[111]:


plot(g,'socialnetwork.pdf')


# In[ ]:




