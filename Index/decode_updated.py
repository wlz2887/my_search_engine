import json
import pickle as pkl
import time,sys
import numpy as np
import nltk
#from Index_Extraction import *  #导入is_number和token2dic方法


Author = []
Body = []
NormBody = []
Content = []
Content_len = []
Summary = []
Summary_len = []
ID = []
Subreddit = []
Sub_id = []
Title = []

Len = 3848330

with open('./corpus-webis-tldr-17.json','r',encoding='utf-8') as f:
    cnt = 0
    for _ in f:
        info = json.loads(_)
        per = cnt/Len*100
        print('\rPercentage Completed: {:.6f}%'.format(per), end = '')
        #sys.stdout.flush()
        #time.sleep(1)        
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

        #Author.append(info['author'])
        #Summary.append(info['summary'])
        Content.append(info['content'])

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

#Summary = np.array(Summary)
#np.save('Summary.npy',Summary)

def is_number(token):
    token = token.replace(",","")
    try:
        float(token)
        return True
    except ValueError:
        return False


def token2dic(tokens,doc_ID):
    for term in tokens:
        dic_doc={}
        dic_doc[doc_ID]=1
        term = term.replace("'","")
        term = term.replace(".","")  #先做一个暴力的，将所有' . 删去
        term = term.lower() #转为小写
        term = stemmer.stem(term)
        if(len(term)==1 or len(term) == 0):  #ignore singel word
            continue
        if(is_number(term)): #ignore numbers
            continue
        if(term in stopwords):
            continue
        if(term not in dic): #term is not in dic
            dic[term] = dic_doc 
        else:
            if(doc_ID in dic[term].keys()): #已经存在
                dic[term][doc_ID]+=1
            else:
                dic[term][doc_ID]=1


stemmer = nltk.stem.porter.PorterStemmer()  #return to stemm word
stopwords = nltk.corpus.stopwords.words('english')   #ignore stopwords
dic = {}
doc_ID = 0

for text in Content:
    tokens = nltk.word_tokenize(text)
    token2dic(tokens,doc_ID)
    doc_ID+=1
    per = doc_ID/Len*100
    #if(doc_ID==100):
    #    break
    print('\rPercentage Completed: {:.6f}%'.format(per), end = '')

#np.save('Body_dic.npy',dic)
#np.save('Sum_dic.npy',dic)
np.save('Content_dic.npy',dic)

#Author = np.array(Author)
#np.savez_compressed("Author",Author)

'''
Body = np.array(Body)
np.save("Body.npy",Body)

NormBody = np.array(NormBody)
np.savez_compressed("NormBody",NormBody)

Content = np.array(Content)
np.savez_compressed("Content",Content)

Content_len = np.array(Content_len)
np.savez_compressed("Content_len",Content_len)

Summary = np.array(Summary)
np.savez_compressed("Summary",Summary)

Summary_len = np.array(Summary_len)
np.savez_compressed("Summary_len",Summary_len)

ID = np.array(ID)
np.savez_compressed("ID",ID)

Subreddit = np.array(Subreddit)
np.savez_compressed("Subreddit",Subreddit)

Sub_id = np.array(Sub_id)
np.savez_compressed("Sub_id",Sub_id)
'''
