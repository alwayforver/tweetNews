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
    def __init__(self,filename,per,loc,org,other):
        self.ind2obj={} # redundant storage for second stream
        self.filename = filename
        self.per=per
        self.loc=loc
        self.org=org
        self.other=other
    def __iter__(self):
        count = 0
        sys.stderr.write("iterator started... \n" )
        if count % 10 == 1:
            sys.stderr.write("processing: " + str(count) +'\r'),
        for item in parsefile(filename,self.per,self.loc,self.org,self.other):
            if not self.per and not self.loc and not self.org and not self.other:
                self.ind2obj[count] = item[1]
            yield item[0]
            count +=1
        sys.stderr.write('\n')
        sys.stderr.write("final count = " + str(count) + '\n')

def parsefile(f,per,loc,org,other):
    fin = codecs.open(f, encoding = 'utf-8')
    for line in fin:
        #if len(b_tokens.split()) > MAX_BODY_LEN:            
        #    continue
        b_tokens_ent = unidecode.unidecode(line.strip())
        b = grep_ent(b_tokens_ent,per,loc,org,other)
        b = rep2.sub('', b)
        b = my_tokenizer(b, tokenizer)
        yield b.lower(),bk.News(raw_text=line.strip()) # can also leave lowercase to scikit
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
    filename = sys.argv[1] 
    nstream = newsStream(filename,False,False,False,False)
    nstream_per = newsStream(filename,True,False,False,False)
    nstream_loc = newsStream(filename,False,True,False,False)
    nstream_org = newsStream(filename,False,False,True,False)
    nstream_plo = newsStream(filename,True,True,True,False)
    nstream_all = newsStream(filename,True,True,True,True)
    
    
    X, vect = getVec(nstream, None, 5)
    Xp, vectp = getVec(nstream_per, None, 5)
    Xl, vectl = getVec(nstream_loc, None, 5)
    Xo, vecto = getVec(nstream_org, None, 5)
    Xplo, vectplo = getVec(nstream_plo, None, 5)
    X_all, vect_all = getVec(nstream_all, None, 5)
    
    # time vector
    DT = []

    import pickle
    with open(sys.argv[2], 'w') as f:
        pickle.dump([ [X,Xp,Xl,Xo,Xplo,X_all],[vect,vectp,vectl,vecto,vectplo,vect_all],DT,nstream.ind2obj],f)
    #for item in nstream:
    #    print item    
    
    #for item in nstream_per:
    #for item in nstream_loc:
    #for item in nstream_org:
    #for item in nstream_all:
