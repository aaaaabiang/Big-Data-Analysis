#!/usr/bin/env python
# coding: utf-8

# In[6]:


import jieba
import os
import re
from gensim import corpora
from gensim import models
import itertools
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
import csv
from tmtoolkit.topicmod import tm_gensim
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results
import matplotlib.pylab as plt

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


# In[7]:


def get_words(file):
    with open(file, encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = [w.strip() for w in stopwords]
    return stopwords


# In[8]:


#分词，过滤停用词
def jieba_tokenizer(doc,stopwords=None):
    tokens = jieba.cut(doc)
    tokens = [el for el in tokens if len(el) > 1]
    import zhon.hanzi as hanzi
    tokens = [el for el in tokens if el not in hanzi.punctuation] ## remove Chinese punctuation
    if stopwords is not None:
        tokens = [w for w in tokens if w not in stopwords]
    return tokens


# In[9]:


#文本清理
def cleandata(text_name,file_name):
    cleantext = []
    for each in text_name:
        each = str(each)
        each = re.sub('@[\u4e00-\u9fa5a-zA-Z0-9_-]{2,30}|[+-]?(0|([1-9]\d*))(\.\d+)?', '', each)
        cleantext.append(each)
    name = ["content"]
    file = pd.DataFrame(columns = name,data = cleantext)
    return  file.to_csv('%s.csv' % file_name)


# In[10]:


data = pd.read_pickle("D:/fudan/大四下/互联网与社会/互联网与社会2020.pkl")
data['content'] = data[1].astype(str)+data[4].astype(str)
content = data['content']
text = content.values.tolist()
cleandata(text_name=text,file_name='content')


# In[11]:


#读入数据，生成列表
r = csv.reader(open("content.csv", encoding="UTF8"))
stp = get_words("stopwords.txt")
text_seg = [jieba_tokenizer(line[1],stp) for line in r if len(jieba_tokenizer(line[1],stp)) > 3 ]  
text_seg


# In[12]:


#建构词典和语料库
dictionary = corpora.Dictionary()
dictionary.add_documents(text_seg)
dictionary.filter_extremes(no_below=2, no_above=0.9)
weibo = [dictionary.doc2bow(doc) for doc in text_seg]


# In[13]:


#模型一致性函数
def get_umass(corpus, num_topics, dictionary):
    mod = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
    cm = models.CoherenceModel(model=mod, corpus=corpus, dictionary=dictionary, coherence="u_mass")
    umass = cm.get_coherence()
    return umass


# In[14]:


#话题数
ks = list(range(1, 10, 1)) + list(range(11, 50, 5))  


# In[99]:


#一致性指标
umass = [get_umass(weibo, k, dictionary) for k in ks]
plt.plot(ks, umass, "k-o")
plt.xlabel("number of topics")
plt.ylabel("umass")
plt.show()


# In[15]:


varying_params = [dict(num_topics=k) for k in ks]
print(varying_params)


# In[44]:


# model comparison and selection
eval_results =     tm_gensim.evaluate_topic_models(data=(dictionary, weibo),
                                    varying_parameters=varying_params,
                                    coherence_gensim_texts=text_seg,
                                    workers = 0
                                    )  # set coherence_gensim_texts to get more coherence measures


# In[45]:


plt.style.use('ggplot')
results_by_n_topics = results_by_parameter(eval_results, 'num_topics')
plot_eval_results(results_by_n_topics,
                  xaxislabel='num. topics k',
                  title='Evaluation results',
                  figsize=(8, 6))


# In[49]:


# choose the final model
mod_final1 = models.LdaModel(corpus=weibo, num_topics=12, id2word=dictionary,chunksize=100,passes=10)
mod_final1.print_topics


# In[50]:


mod_final1.save("mod_final1.gensim")


# In[51]:


mod_final1.show_topics(12,50 )

