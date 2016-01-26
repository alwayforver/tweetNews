from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import sys,os
import codecs
import glob
import re

import os.path
import backendDefs as bk
from datetime import datetime as dt
from datetime import timedelta as tdelta
import operator
import scipy.io as sio
from scipy.sparse import coo_matrix,hstack

def readfile(file,dataop,count,ind2obj,dtpure,lines):
#    lines = []
#    ind2obj = {}
#    count = 0
    for line in file:
        if len(line.strip().split("\t")) != 9:
            continue
        ID,url,title,source,created_at,authors,key_word,snippets,raw_text = line.strip().split("\t")
        ID = int(ID)
        ind2obj[count] = bk.News(ID,title,raw_text,snippets,key_word,source,created_at,dtpure)
        if dataop == "all":
            lines.append(line)
        if dataop == "text":
            lines.append('\t'.join([title,key_word,snippets,raw_text]))
        if dataop == "snippets":
            lines.append('\t'.join([title,key_word,snippets]))
        count += 1
    return count
def getVec(lines,vocab):
    if vocab:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                min_df=2, stop_words='english',
                use_idf=opts.use_idf,vocabulary=vocab)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                min_df=2, stop_words='english',
                use_idf=opts.use_idf)
    print("Extracting features using a sparse vectorizer")
    t0 = time()
    X = vectorizer.fit_transform(lines)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()
    return (X,vectorizer)
def getCluster(X,k,opts):
    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=50,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=50,
                    verbose=opts.verbose)
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()
    
    labels = km.labels_
    clus2doc = {}
    for i in range(len(labels)):
        clus2doc[labels[i]] = clus2doc.get(labels[i],set())
        clus2doc[labels[i]].add(i)    
    return (km,clus2doc)
def getRelTweets(newsID,dtpure,tweetPre,tweetIDset,tweetSet):
    #n = News.objects.filter(ID=newsID)
    #if n.count() > 0:
    #    #return News.objects.get(ID=newsID).tweet_set.all()
    #    return list(n[0].tweet_set.all())
    #else:
    #    return []
    t_path = glob.glob(tweetPre+dtpure+"/"+str(newsID)+"_*")
    if len(t_path) != 1:
        print('no tweets for news ',newsID,'len(t_path)',len(t_path))
        return ([],[])
    if os.path.exists(t_path[0]):
        t = codecs.open(t_path[0], encoding = 'utf-8') 
    
    #tweets = set()
    tweets = []
    tweetsObj = []
# stupid redundancy
    for line in t:
        fields = line.strip().split("\t")
        if len(fields) < 24:
        #    tweets_log.write("not 27:"+line.strip()+"\n")
            continue
        ID, raw_text, created_at, contained_url, hash_tags, retw_id_str, retw_favorited, retw_favorite_count, is_retweet, retweet_count, \
        tw_favorited, tw_favorite_count, tw_retweeted, tw_retweet_count, user_id_str, verified, follower_count, statuses_count, friends_count, \
    favorites_count, user_created_at= fields[:21]
        try:
            ID = int(ID)
        except:
            continue
        try:
            is_retweet=bool(is_retweet)
        except:
            is_retweet=False
        try:
            retweet_count = int(retweet_count)
        except:
            retweet_count = -1
        #if len(tag_text) > 100:
        #    tweets_log.write("hashtag too long: "+line.strip()+"\n")
        #    continue
        #if len(tw_text) > 200:
        #    tweets_log.write("tweet too long: "+line.strip()+"\n")
        #    continue
        #
        ## convert user_created time
        ## Fri Nov 07 22:20:38 +0000 2014
        #tw_created_at_tz = parse(tw_created_at) # utc time with tz information
        #tw_local_timezone = tw_created_at[len(tw_created_at)-10:len(tw_created_at)-5] # +0000

#        tweets.append(tw_text)
        #s = tw_text.find("http://")
        #if s == -1:
        #    s = tw_text.find("https://")
        #if s != -1:
        #    tmp = tw_text[s:] 
        #    e = tmp.find(" ")
        #    if e == -1:
        #        e = len(tmp)
        #    tw_text = (tw_text[:s].strip()+ " " + tmp[e:].strip()).strip()
            #if s != 0 and tw_text[s-1] != " " and tw_text[s-1] != "\t":
            #    tw_text = tw_text[:s] + tmp[e:]
            #else:
            #    tw_text = tw_text[:s] + tmp[e+1:]
        # remove url    
        #tw_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tw_text)
        if "http" not in raw_text and "RT @" not in raw_text \
            and ID not in tweetIDset and raw_text not in tweetSet:
            tweet = bk.Tweet(ID,raw_text,created_at,is_retweet,retweet_count,hash_tags)
            tweetsObj.append(tweet)
            tweets.append(raw_text)
            tweetIDset.add(ID)
            tweetSet.add(raw_text)
#        tweets.add(tw_text)
        #tweet = Tweet(ID=int(tw_id_str), user=int(user_id_str) ,raw_text = tw_text,created_at = tw_created_at_tz, local_time_zone = tw_local_timezone, retweet_count = tw_retweet_count,\
        #hash_tags = tag_text)
    t.close()
#    if tweets:
    return (tweets,tweetsObj)
 #   else:
  #      return None
def getNewsCenter(X,indList):
    return X[np.array(list(indList)),:].mean(0)
    
def rankTweets(tweets, tweetsObj, newsVec, vocab, t_topK):
    tweetVectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf,vocabulary=vocab)
    X = tweetVectorizer.fit_transform(tweets)
    scores = X.dot(newsVec)
    tweetsInd = scores.argsort()[::-1][:t_topK]
    topTweetsObj = [tweetsObj[i] for i in tweetsInd]
    topTweetsScore = {}
    for i in tweetsInd:
        topTweetsScore[tweetsObj[i].ID] = scores[i]
    return topTweetsObj,topTweetsScore
def printCluster(X,i,lines,order_centroids,centers,terms,t_topK, opts):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
        center = centers[i,:]
        Ind = X.dot(center).argsort()[::-1][:t_topK] 
        for tweet in Ind:
            print(lines[tweet])

#################################################################################        
def getEntMat(ind2obj,resind,ent_count,thresh):
    row = []
    col = []
    data = []
    terms = {}
    termList = []
    count = 0
    for i in ind2obj:
        res = ind2obj[i].entities
                                                                                                 
        for e in res[resind]:
            if ent_count[e] <thresh:
                continue
            if e not in terms:
                terms[e] = count
                termList.append(e)
                count += 1
                
            row.append(i)
            col.append(terms[e])
            data.append(res[resind][e])
    row = np.array(row)
    col = np.array(col)
    data = np.array(data,dtype=float)
    entMat = coo_matrix((data,(row,col)),shape=(len(ind2obj),count))
    return entMat,termList

def getSurface(ind2obj):
    ent_surf = {}
    for i in ind2obj:
        res = ind2obj[i].entities
        entity_surface = res[4]
        for e in entity_surface:
            if e not in ent_surf:
                ent_surf[e] = {}
            surfs = entity_surface[e]
            for sf in surfs:
                ent_surf[e][sf] = ent_surf[e].get(sf,0) + surfs[sf]
    return ent_surf


if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO,
     #                   format='%(asctime)s %(levelname)s %(message)s')    
    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")
    
    (opts, args) = op.parse_args()
   
# tweetfile loadmat starttime savemat outfile4tweets
lines = []
ts = []
rts = []
tweetfile = codecs.open(sys.argv[1], encoding = 'utf-8')
mat = sio.loadmat(sys.argv[2])
s_dt = dt.strptime(sys.argv[3],"%Y-%m-%d")
count = 0

terms = mat['terms']
terms = [x.strip() for x in terms]
#
for line in tweetfile:
    try:
        (tweet,created_at,rt) = line.strip().split('\t')
    except:
        continue
    try:
        dtts = dt.strptime(created_at[4:],"%b %d %H:%M:%S %Y")
    except:
        continue
    lines.append(tweet)
    ts.append((dtts-s_dt).total_seconds())
    rts.append(int(rt)) 
    count+=1
    if count%10000 == 1:
        print(count)
#X,vectorizer = getVec(lines,terms)
#

#
tweetmat = sio.loadmat(sys.argv[4])
X = tweetmat['Xt']
(clusModel,clus2doc) = getCluster(X,6,opts) 
order_centroids = clusModel.cluster_centers_.argsort()[:, ::-1]
for i in range(6):
    printCluster(X,i,lines,order_centroids,clusModel.cluster_centers_,terms,50, opts)

#
exit(0)
sio.savemat(sys.argv[4],dict(Xt=X,ts=ts,rts=rts))
outfile=codecs.open(sys.argv[5], 'w', encoding = 'utf-8')
for line in lines:
    outfile.write(line+"\n")

