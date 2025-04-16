#!/usr/bin/env python
# coding: utf-8

# In[102]:


from wordcloud import WordCloud
import pandas as pd
import collections
import matplotlib.pylab as plt #词云转换为图片的库
import jieba
import argparse
import re


# In[103]:


def get_words(file):
    with open(file, encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = [w.strip() for w in stopwords]
    return stopwords


# In[341]:


def text2wordcloud(text_path, word_cloud_path, stopwords, new_words, font_path):
    if new_words is not None:
        for w in new_words:
            jieba.add_word(w)
    freq = collections.Counter()
    texts = open(text_path, encoding="utf-8", errors="replace")
    for text in texts:
        tokens = jieba.lcut(text)
        tokens = [w for w in tokens if len(w) > 1]
        if stopwords is not None:
            tokens = [w for w in tokens if w not in stopwords]
        for w in tokens:
            freq[w] += 1
    print(freq.most_common(20))
    
    cloud = WordCloud(font_path=font_path,
                    width=1600, height=800, background_color='white')
    cloud.generate_from_frequencies(freq)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig("%s.pdf" % word_cloud_path)
    
    return freq


# In[342]:


#构建清理数据函数
def cleandata(text_name,file_name):
    cleantext = []
    for each in text_name:
        each = str(each)
        each = re.sub('@[\u4e00-\u9fa5a-zA-Z0-9_-]{2,30}|[+-]?(0|([1-9]\d*))(\.\d+)?', '', each)
        cleantext.append(each)
    name = ["content"]
    file = pd.DataFrame(columns = name,data = cleantext)
    return  file.to_csv('%s.csv' % file_name)


# In[343]:


#打开和重命名数据
data = pd.read_pickle("D:\python\互联网与社会2020.pkl")
data.rename(columns={0:'time'},inplace=True)
data['content'] = data[1].astype(str)+data[4].astype(str)


# In[344]:


#生成整体词云
content = data['content']
text = content.values.tolist()
cleandata(text_name=text,file_name='allcontent')
text2wordcloud('allcontent.csv', 'word_cloud_path.pdf', stopwords=get_words('stopwords.txt'), new_words=None, font_path='D:\python\SIMSUN.ttf')


# In[351]:


#整合分段处理相关数据
time = data['time'].values.tolist()
content = data['content'].values.tolist()
list1 = list(zip(time,content))    
del list1[0:8]
list1[0:5]
listafter = []
listbefore = []
list4 = []
for each in list1:
    each = list(each)
    list4.append(each)
#划分前后时间段内容,生成对应内容列表
for each in list4:
    if each[0].startswith('2015') or each[0].startswith('2016') or each[0].startswith('2017') or each[0].startswith('2018'):
            listafter.append(each[1])
    else:
            listbefore.append(each[1])


# In[352]:


#生成15-18年词云
cleandata(text_name=listafter,file_name='aftercontent')
text2wordcloud('aftercontent.csv', 'word_cloud_path.pdf', stopwords=get_words('stopwords.txt'), new_words=None, font_path='D:\python\SIMSUN.ttf')


# In[353]:


#生成11-15前词云
cleandata(text_name=listbefore,file_name='beforecontent')
text2wordcloud('beforecontent.csv', 'word_cloud_path.pdf', stopwords=get_words('stopwords.txt'), new_words=None, font_path='D:\python\SIMSUN.ttf')


# In[ ]:




