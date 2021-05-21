import json
import pickle as pkl
import time,sys
import numpy as np
import nltk

#All dics needed
Author = []
Body = [] #content+summary
NormBody = [] #remove extra \n from Body
Content = [] 
Content_len = []
Summary = [] 
Summary_len = []
ID = []
Subreddit = []
Sub_id = []
Title = []

Len = 3848330


def is_number(token):
    token = token.replace(",","")
    try:
        float(token)
        return True
    except ValueError:
        return False


def token2dic(tokens,doc_ID): #构建倒叙索引 如需词频则应再构建一个doc内的字典
    for term in tokens:
        term = term.replace("'","")
        term = term.replace(".","")  #先做一个暴力的，将所有' . 删去
        term = term.lower() #转为小写
        term = stemmer.stem(term)
        if(len(term)==1 or len(term) == 0):  #ignore single word
            continue
        if(is_number(term)): #ignore numbers 或者不该省略？
            continue
        if(term in stopwords):
            continue
        if(term not in dic): #字典中没有该term 向字典添加
            dic[term] = [doc_ID]
        else:
            if(dic[term][-1] != doc_ID):
                dic[term].append(doc_ID)


with open('./corpus-webis-tldr-17.json','r',encoding='utf-8') as f:
    cnt = 0
    for _ in f:
        info = json.loads(_)
        per = cnt/Len*100
        print('\rPercentage Completed: {:.6f}%'.format(per), end = '') 

        #The key is nullable, set default to None in case of KeyError
        info.setdefault('author',None)
        info.setdefault('body',None)
        info.setdefault('normalizedBody',None)
        info.setdefault('content',None)
        info.setdefault('content_len',None)
        info.setdefault('summary',None)
        info.setdefault('summary_len',None)
        info.setdefault('id',None)
        info.setdefault('subreddit',None)
        info.setdefault('subreddit_id',None) 
        info.setdefault('title',None)

        Body.append(info['body'])
        '''
        NormBody.append(info['normalizedBody'])
        Content.append(info['content'])
        Content_len.append(info['content_len'])
        Summary.append(info['summary'])
        Summary_len.append(info['summary_len'])
        ID.append(info['id'])
        Subreddit.append(info['subreddit'])
        Sub_id.append(info['subreddit_id'])
        Title.append(info['title'])
        '''
        cnt+=1
        #if(cnt==10): #提取前十个看看效果
        #    break


stemmer = nltk.stem.porter.PorterStemmer()  #还原成词根
stopwords = nltk.corpus.stopwords.words('english')   #ignore stopwords
dic = {}
doc_ID = 0

for text in Body:
    tokens = nltk.word_tokenize(text) #分词
    token2dic(tokens,doc_ID)
    doc_ID+=1
    per = doc_ID/Len*100
    print('\rPercentage Completed: {:.6f}%'.format(per), end = '')

np.save('Body_dic.npy',dic)

#Author = np.array(Author)
#np.savez_compressed("Author",Author)

'''
Body = np.array(Body)
np.save("Body.npy",Body)

NormBody = np.array(NormBody)
np.save("NormBody.npy",NormBody)

Content = np.array(Content)
np.save("Content.npy",Content)

Content_len = np.array(Content_len)
np.save("Content_len.npy",Content_len)

Summary = np.array(Summary)
np.save("Summary.npy",Summary)

Summary_len = np.array(Summary_len)
np.save("Summary_len.npy",Summary_len)

ID = np.array(ID)
np.save("ID.npy",ID)

Subreddit = np.array(Subreddit)
np.save("Subreddit.npy",Subreddit)

Sub_id = np.array(Sub_id)
np.save("Sub_id.npy",Sub_id)
'''
