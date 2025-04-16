#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入包
from IPython.display import Image
from igraph import *
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import jieba
import zhon.hanzi as hanzi
import re
import csv
import os
from gensim import corpora
from gensim import models
import itertools
import logging
from collections import defaultdict
from tmtoolkit.topicmod import tm_gensim
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


# In[193]:


#打开文件
r = pd.read_pickle("D:/STUDY/互联网与社会研究/作业/互联网与社会2020.pkl")
r.head(20)


# In[194]:


#df1：用于社群分析。筛选有转发行为的用户、博文、时间并重新设置索引
r.dropna(subset=[6],inplace=True) #删除没有转发行为的用户
df = r.loc[:, [0,1,3,4,6]].reset_index().drop(['index'],axis = 1)
df.rename(columns = {0:'year',1:'repo_cont',3:'reposter',4:'orig_cont',6:'origin'},inplace = True)
df.head()


# In[195]:


#博文基本信息1：总体数量（数据量级）
len(df)


# In[196]:


#博文基本信息2：数量随时间的变化（后续有时效变化分析）

#构建获取年份信的函数
def getyear(a):
    if len(a) == 10:
        return a[0:4]
    else:
        return '2019'
    
#重写年份栏
df['year'] = df.apply(lambda x:getyear(x.year), axis = 1)
df.head()


# In[197]:


#微博数量的年份分布可视化
a = df['year'].value_counts().sort_index()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['font.family']='sans-serif'
a.plot(kind = "bar",title = '微博数量年份分布')

#删除2010、2019年的微博数据
df=df[~(df['year'].isin(['2010','2019']))]


# In[7]:


#构建从原创者到转发者的网络链接元组
reposter = df['reposter'].values.tolist()
origin = df['origin'].values.tolist()
link = []
for i in range(len(df)):
    repotuple = (origin[i],reposter[i])
    link.append(repotuple)

#网络链接去重
newlink = list(set(link))
len(newlink)


# In[8]:


#构建转发者、原创者列表
tuplereposter = [link[i][1] for i in range(len(link))]
tupleorigin = [link[i][0] for i in range(len(link))]
newreposter = []
neworigin = []
for each in reposter:
    if each in tuplereposter:
        newreposter.append(each)
for each in origin:
    if each in tupleorigin:
        neworigin.append(each)
newreposter = list(set(newreposter))
neworigin = list(set(neworigin))
print(len(newreposter),len(neworigin))


# In[130]:


#创建网络
g = Graph(directed=True)


# In[131]:


#添加节点、链接
g.add_vertices(newreposter)
g.add_vertices(neworigin)
g.add_edges(newlink)


# In[137]:


print(g.summary())


# In[136]:


#按度过滤节点

#删除度数小的节点
for v in g.vs:
        if g.degree(v) <= 5:
            g.delete_vertices(v)


# In[138]:


#网络基本信息：度分布的可视化
deg = g.degree()
degpd = pd.Series(deg)

degpd.plot(kind='hist',logx=True,logy=True,title = '节点度分布')
plt.show()


# In[139]:


#度分布统计
print(g.degree_distribution())


# In[140]:


#按转发、原创角色设置形状

#节点属性1：转发者or原创者,先生成{节点：属性}对应的字典，再get节点对应的属性并加入网络
dicV_character = {}
for i in newreposter:
    dicV_character[i] = 0
for i in neworigin:
    dicV_character[i] = 1
        
listV_character = [dicV_character.get(v) for v in g.vs['name']]
g.vs['character'] = listV_character
    
for v in g.vs:
    if g.vs['character'] == 1:
        g.vs['shape'] = 'square'


# In[141]:


#按度设置节点大小

#节点属性2：节点大小size，与节点中心性相关
vsize = utils.rescale(g.degree(), [1, 4])
g.vs['size'] = [vsize[i]*6 for i in range(len(vsize))]


# In[142]:


#链接属性1：链接名字,与来源-目标节点相关
namelist = []
dicE_name = {}
for e in g.es:
    ename = (g.vs[e.source]['name'], g.vs[e.target]['name'])
    namelist.append(ename)
    for i in range(len(namelist)):
        dicE_name[i] = namelist[i]

listE_name = [dicE_name.get(e.index) for e in g.es]
g.es['name'] = listE_name


# In[143]:


#按链接强度设置链接粗细

#链接属性2：关系强度weight,与转发次数相关
dicE_weight = {}
for each in listE_name:
        dicE_weight[each] = link.count(each)
listE_weight = [dicE_weight.get(e['name']) for e in g.es]
g.es['weight'] = listE_weight


# In[144]:


#链接属性3：链接宽度width,与关系强度相关
dicE_width = {}
ewidth = utils.rescale(listE_weight, [1, 50])
for e in g.es:
    dicE_width[e.index] = ewidth[e.index]

listE_width = [dicE_width.get(e.index) for e in g.es]
g.es['width'] = listE_width


# In[145]:


#设置箭头大小
g.es['arrow_size'] = 0.5
g.es['arrow_width'] = 0.5


# In[146]:


#作图前，删除孤立节点
for v in g.vs:
     if g.degree(v) == 0:
        g.delete_vertices(v)


# In[147]:


print(g.summary())


# In[148]:


g['layout'] = g.layout_fruchterman_reingold()
plot(g).show()


# In[149]:


#社群分析基本设置
cps = g.components('WEAK')#弱成分的分析
g = cps.giant() # 最大成分的规模

#社群分析，一共18个社群
sgcom = g.community_spinglass()
print(sgcom.summary()) ## summary of the community structure
sgcom.membership ## membership of each node
g.vs['community'] = sgcom.membership


# In[150]:


#社群基本信息1：各个社群的规模（只选择规模大于一定数量的）
dicV_count = {}
for v in g.vs:
    list1 = g.vs['community']
    dicV_count[v['community']] = list1.count(v['community'])

print(sorted(dicV_count.items(), key=lambda d:d[0]))#字典排序 key=lambda d:d[0]


# In[48]:


#设置颜色透明度函数
def col_to_rgba(col_name, alpha=0.5):
    return color_name_to_rgb(col_name) + (alpha,) 


# In[49]:


#按社群设置颜色
ncluster = max(g.vs['community']) + 1

c4c = ClusterColoringPalette(ncluster)

for v in g.vs:
    v['color'] = c4c.get(v['community'])
    v['frame_color'] = c4c.get(v['community'])
    
for e in g.es:
    e['color'] = g.vs[e.source]['color']
    
g.vs['color'] = [col_to_rgba(v, 0.5) for v in g.vs['color']]

g.vs['color'] = [col_to_rgba(v, 0.5) for v in g.vs['color']]
g.vs['frame_color'] = [col_to_rgba(v, 0.5) for v in g.vs['frame_color']]
g.es['color'] = [col_to_rgba(e, 0.5) for e in g.es['color']]


# In[37]:


#显示所属社群
for v in g.vs:
    v['label'] = v['community']
    v['label_size'] = 5


# In[46]:


#作图前，删除孤立节点
for v in g.vs:
     if g.degree(v) == 0:
        g.delete_vertices(v)


# In[42]:


print(g.summary())


# In[127]:


g['layout'] = g.layout_fruchterman_reingold()
plot(g,'socialnetwork.png')


# In[151]:


#社群基本信息2：各个社群的影响力核心成员

#创建标记每个社群中心性最高的节点的函数
def gettop(n,graph):
    dicrank = {}
    toplist = []
    for v in graph.vs:
        if v['community'] == n:
            tuple2 = (v.degree(mode = 'OUT'),v['name'],v['community'])
            toplist.append(tuple2)
    return toplist

toplist = []
for i in range(ncluster):
    topinfo = sorted(gettop(i,g),reverse = True)
    toplist.append(topinfo[0:3])

topnamelist = []
for each in toplist:
    for i in each:
        topnamelist.append(i[1])
print(toplist)


# In[ ]:


#各个社群命名和概述：通过核心影响力成员的博文


# In[ ]:


#各个社群的互动特征：社群内的度数、社群外的度数


# In[ ]:


#按年份显示网络图，随时间变化的趋势，用于描述社群互动结构的演进


# In[ ]:


#按社群互动特征将社群归类为群体，获取节点（转发者）名单


# In[ ]:


#df2-3：用于话题模型。筛选每个群体中的转发者、转发博文、原创者、原创博文、时间并各建df


# In[154]:


#构建取词函数
def get_words(file):
    with open(file, encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = [w.strip() for w in stopwords]
    return stopwords

#构建分词函数，过滤停用词
def jieba_tokenizer(doc,stopwords=None):
    tokens = jieba.cut(doc)
    tokens = [el for el in tokens if len(el) > 1]
    tokens = [el for el in tokens if el not in hanzi.punctuation] ## remove Chinese punctuation
    if stopwords is not None:
        tokens = [w for w in tokens if w not in stopwords]
    return tokens

#构建文本清理函数
def cleandata(text_name,file_name):
    cleantext = []
    for each in text_name:
        each = str(each)
        each = re.sub('@[\u4e00-\u9fa5a-zA-Z0-9_-]{2,30}|[+-]?(0|([1-9]\d*))(\.\d+)?', '', each)
        cleantext.append(each)
    name = ['content']
    file = pd.DataFrame(columns = name,data = cleantext)
    return  file.to_csv('%s.csv' % file_name)


# In[200]:


#删除没有出现在网络里的行
netnamelist = []
for v in g.vs:
    if v['character'] == 0:
        netnamelist.append(v['name'])
deletelist = []
for each in reposter:
    if each not in netnamelist:
        deletelist.append(each)
deletelist = list(set(deletelist))        
print(deletelist)


# In[205]:


for i in range(len(netnamelist)):
    df2 = df[~(df['reposter'].isin([netnamelist[i]]))]

df2.head(50)   


# In[155]:


#合并文本
df['all_cont'] = df['repo_cont']+df['orig_cont']
df.head()


# In[156]:


#文本清理
text = df['all_cont'].values.tolist()
cleandata(text_name=text,file_name='content')

#去停用词
r = csv.reader(open('content.csv', encoding="UTF8"))
stp = get_words('baidu_stopwords.txt')
text_seg = [jieba_tokenizer(line[1],stp) for line in r if len(jieba_tokenizer(line[1],stp)) > 5 ]  


# In[157]:


#建构词典和语料库
dictionary = corpora.Dictionary()
dictionary.add_documents(text_seg)
dictionary.filter_extremes(no_below=2, no_above=0.9)
weibo = [dictionary.doc2bow(doc) for doc in text_seg]


# In[158]:


#话题数
ks = list(range(1, 11, 1)) + list(range(11, 30, 5))  

varying_params = [dict(num_topics=k) for k in ks]
print(varying_params)


# In[159]:


# model comparison and selection
eval_results =     tm_gensim.evaluate_topic_models(data=(dictionary, weibo),
                                    varying_parameters=varying_params,
                                    coherence_gensim_texts=text_seg,
                                    )  # set coherence_gensim_texts to get more coherence measures


# In[160]:


plt.style.use('ggplot')
results_by_n_topics = results_by_parameter(eval_results, 'num_topics')
plot_eval_results(results_by_n_topics,
                  xaxislabel='num. topics k',
                  title='Evaluation results',
                  figsize=(8, 6))


# In[162]:


# choose the final model
mod_final1 = models.LdaModel(corpus=weibo, num_topics=9, id2word=dictionary,chunksize=100,passes=5)
mod_final1.print_topics


# In[163]:


# choose the final model
mod_final2 = models.LdaModel(corpus=weibo, num_topics=5, id2word=dictionary,chunksize=100,passes=5)
mod_final2.print_topics


# In[164]:


mod_final1.save("mod_final1.gensim")
mod_final1.show_topics(9,50 )


# In[165]:


mod_final2.save("mod_final2.gensim")
mod_final2.show_topics(5,50 )


# In[206]:


def get_topics(lda, text):
    ps = defaultdict(int,lda[text])
    return [ps[k] for k in range(lda.num_topics)]


# In[207]:


tp = [get_topics(mod_final2,_) for _ in weibo]


# In[208]:


tpdf = pd.DataFrame(tp, columns = [f'topics{_}' for _ in range(1,6)])
tpdf.head()


# In[209]:


ans = pd.concat([df,tpdf], axis = 1)#use pd.merge if there are doc_id variable
ans.head(20)


# In[223]:


vcommunity = g.vs['community']
vname = g.vs['name']


c ={'vname':vname,'vcommunity':vcommunity}#合并成一个新的字典c

new = pd.DataFrame(c)
new.head(50)


# In[220]:


ans2 = pd.merge(ans,new,how = 'inner', left_on = 'reposter',right_on = 'vname')


# In[222]:


communityavg = ans2.groupby('vcommunity')['topics1','topics2','topics3','topics4','topics5'].mean()
communityavg.head(20)


# In[224]:


communityavg.plot(kind = "bar")

