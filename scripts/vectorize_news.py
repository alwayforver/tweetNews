import sys,os
import codecs,unidecode
import re,string
import requests,json

import nltk
from datetime import datetime as dt
from datetime import timedelta as tdelta
from utils import dbpedia,parseEntity,my_tokenizer,grep_ent,grep_ent_with_context,MAX_BODY_LEN,rep1,rep2
import backendDefs as bk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from time import time

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
tokenizer = 'reg'


class newsStream(object):
    def __init__(self,newsDIR,s_dt,e_dt,per,loc,org,other):
        self.newsDIR=newsDIR
        self.s_dt=s_dt
        self.e_dt=e_dt
        self.ind2obj={} # redundant storage for second stream
        self.per=per
        self.loc=loc
        self.org=org
        self.other=other
    def __iter__(self):
        numDays = (self.e_dt - self.s_dt).days
        count = 0
        sys.stderr.write("iterator started... \n" )
        for x in range(numDays+1):
            fileDate = self.s_dt + tdelta(days = x)
            filename = fileDate.strftime("%Y-%m-%d")+'.txt'
            if os.path.isfile(newsPre+filename):
                sys.stderr.write("file processing: " + filename +'\r'),
                for item in parsefile(filename,self.newsDIR,self.per,self.loc,self.org,self.other):
                    self.ind2obj[count] = item[1]
                    yield item[0]
                    count +=1
        sys.stderr.write('\n')
        sys.stderr.write("final count = " + str(count) + '\n')

def parsefile(f,inPre,per,loc,org,other):
    fin = codecs.open(inPre+f, encoding = 'utf-8')
    for line in fin:
        if len(line.strip().split("\t")) != 13:
            continue
        ID,url,title,source,created_at,authors,key_word,snippets,raw_text,\
                h_tokens,b_tokens,h_tokens_ent,b_tokens_ent = line.strip().split("\t")
        if len(b_tokens.split()) > MAX_BODY_LEN:            
            continue
        h_tokens_ent = unidecode.unidecode(h_tokens_ent.strip())
        b_tokens_ent = unidecode.unidecode(b_tokens_ent.strip())
        #h = grep_ent_with_context(h_tokens_ent,per,loc,org,other)  # fds_per_| asked me about ...
        #b = grep_ent_with_context(b_tokens_ent,per,loc,org,other)
        h = grep_ent(h_tokens_ent,per,loc,org,other) # fsd_per_| oregon_loc_| ...
        b = grep_ent(b_tokens_ent,per,loc,org,other)
        h = rep2.sub('', h)
        b = rep2.sub('', b)
        h = my_tokenizer(h, tokenizer)
        b = my_tokenizer(b, tokenizer)
        tokens = h+' '+h+' '+b  # title twice
        yield tokens.lower(),bk.News(ID,title,raw_text,snippets,key_word,source,created_at,f.split('.')[0]) # can also leave lowercase to scikit
    fin.close()

def getVec(stream, vocab, min_df):
    #countVect = CountVectorizer(max_df=1.0, min_df=2, max_features=None, vocabulary=None)
    tfidfVect = TfidfVectorizer(analyzer=string.split, max_df=1.0, min_df=min_df, max_features=None, vocabulary=vocab)
    #Xcount = countVectorizer.fit_transform(nstream)
    t0 = time()
    X = tfidfVect.fit_transform(stream)
    print
    print "done in (sec): ", time() - t0
    print "n_samples, n_features: ", str(X.shape)
    return X,tfidfVect

# python vectorize_news.py ../data/news_tokenized/ 2016-01-07 2016-01-23 ../data/20160107_0123.pickle
if __name__ == "__main__":
    newsPre = sys.argv[1]
    s_dt = dt.strptime(sys.argv[2],"%Y-%m-%d")
    e_dt = dt.strptime(sys.argv[3],"%Y-%m-%d")
    
    nstream = newsStream(newsPre,s_dt,e_dt,False,False,False,False)
    nstream_per = newsStream(newsPre,s_dt,e_dt,True,False,False,False)
    nstream_loc = newsStream(newsPre,s_dt,e_dt,False,True,False,False)
    nstream_org = newsStream(newsPre,s_dt,e_dt,False,False,True,False)
    nstream_all = newsStream(newsPre,s_dt,e_dt,True,True,True,True)
    
    
    X, vect = getVec(nstream, None, 2)
    Xp, vectp = getVec(nstream_per, None, 2)
    Xl, vectl = getVec(nstream_loc, None, 2)
    Xo, vecto = getVec(nstream_org, None, 2)
    X_all, vect_all = getVec(nstream_all, None, 2)
    
    # time vector
    DT = []
    for i in nstream.ind2obj:
        curr_t = (nstream.ind2obj[i].created_at - s_dt).total_seconds()
        DT.append(curr_t)

    import pickle
    with open(sys.argv[4], 'w') as f:
        pickle.dump([X,Xp,Xl,Xo,X_all,vect,vectp,vectl,vecto,vect_all,DT,nstream.ind2obj],f)
    #for item in nstream:
    #    print item    
    
    #for item in nstream_per:
    #for item in nstream_loc:
    #for item in nstream_org:
    #for item in nstream_all:
