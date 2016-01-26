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
def getCluster(X,k,M,opts):
    # M: knnNum
#    t0 = time()
#    print("knn graph")
    knn_graph = None
    #    knn_graph = kneighbors_graph(X, M)
#    print("knn graph done in %0.3fs" % (time() - t0))
#    outfile.write("knn graph done in %0.3fs\n" % (time() - t0))
#    aggl = AgglomerativeClustering(linkage='ward', connectivity=knn_graph, n_clusters=k)
    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=50,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=50,
                    verbose=opts.verbose)
    #aggl = AgglomerativeClustering(linkage='ward', n_clusters=k)
    print("Clustering sparse data with %s" % km)
#    outfile.write("Clustering sparse data with %s\n" % aggl)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
#    outfile.write("clustering done in %0.3fs\n" % (time() - t0))
    print()
    
    labels = km.labels_
    clus2doc = {}
    for i in range(len(labels)):
        clus2doc[labels[i]] = clus2doc.get(labels[i],set())
        clus2doc[labels[i]].add(i)    
    return (km,clus2doc,knn_graph)
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
def printCluster(X,i,lines,clus2doc,clusModel,order_centroids,terms,outfile,ind2obj,t_topK,vectorizer,tweetPre,opts):
    if not (opts.n_components or opts.use_hashing):
        print("Cluster %d:" % i, end='')
        outfile.write("Cluster %d:" % i)
    #    for ind in order_centroids[i, :10]:
    #        print(' %s' % terms[ind], end='')
    #        outfile.write(' %s' % terms[ind])
        print()
        outfile.write('\n')
        tweets = []
        tweetsObj = []
    #    tweets = set()
        newsList = [ind2obj[ind] for ind in clus2doc[i]]
        newsEntityDict = {}
        tweetsEntityDict = {}
        for news in sorted(newsList, key=operator.attrgetter('created_at')):
#        for ind in clus2doc[i]:
#            news = ind2obj[ind]

            print(str(news.created_at)+"\t"+news.title)
            print(news.entities())
            outfile.write(str(news.created_at)+"\t"+news.title+"\n")
            outfile.write(news.entities()+"\n")
            #print(lines[ind].split('\t')[2])
            #outfile.write(lines[doc].split('\t')[2])
            #outfile.write('\n')
            print("-------")
            outfile.write("-------\n")
            newsID = news.ID
            dtpure = news.dtpure
            tweetIDset = set()
            tweetSet = set()
            #if getRelTweets(newsID,dtpure,tweetPre, tweetIDset,tweetSet):
            addtweets,addtweetsObj = getRelTweets(newsID,dtpure,tweetPre,tweetIDset,tweetSet)           
            tweets = tweets + addtweets
            tweetsObj = tweetsObj + addtweetsObj
    #            tweets = tweets | getRelTweets(newsID)
    #    tweets = list(tweets)
        if tweets:
            newsCenter = np.squeeze(np.asarray(getNewsCenter(X,clus2doc[i])))
            for term in newsCenter.argsort()[::-1][:20]:
                print(' %s' % terms[term], end='')
                outfile.write(' %s' % terms[term])
            #topTweets = rankTweets(tweets, clusModel.cluster_centers_[i,:], vectorizer.vocabulary_,t_topK)
            topTweetsObj,topTweetsScore = rankTweets(tweets,tweetsObj, newsCenter, vectorizer.vocabulary_,t_topK)
            print("*******total tweets: "+str(len(tweets)))
            outfile.write("\n*******top tweets:********total tweets: " + str(len(tweets))+"\n")
            print("top tweets:")
            for t in sorted(topTweetsObj, key=operator.attrgetter('created_at')):
                print(str(topTweetsScore[t.ID])+"\t"+str(t.created_at)+"\t" + t.raw_text )
                print(t.entities())
                outfile.write(str(topTweetsScore[t.ID])+"\t"+str(t.created_at)+"\t" + t.raw_text+"\n")
                outfile.write(t.entities())
                outfile.write('\n-------\n')
                print("-------")
        else:
            print("no tweets retrieved")
            outfile.write("no tweets retrieved\n")
        print("=========")
        outfile.write("=========\n\n")
        print()

#def process(X,k,lines,vectorizer,outfile,ind2obj,t_topK,opts):
#    (clusModel,clus2doc) = getCluster(X,k,opts)
##    order_centroids = clusModel.cluster_centers_.argsort()[:, ::-1]
##    terms = vectorizer.get_feature_names()
#    order_centroids=None
#    terms=None
#    for i in range(k):
#        #outfile.write('%s' % maxDist)
#        printCluster(i,lines,clus2doc,clusModel,order_centroids,terms,outfile,ind2obj,t_topK,vectorizer,tweets_input_prefix,opts)
#        outfile.write('*****************\n')
#    print(clusModel.n_leaves_)
#    print(clusModel.n_components_)
#    print(clusModel.children_)

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
    if len(args) != 9:
        op.error("this script takes 9 arguments: news_input_prefix start_date end_date tweets_input_prefix num_clus t_topK all/text/snippets outfile thresh")
        sys.exit(1)
    
    newsPre = sys.argv[1]
    s_dt = dt.strptime(sys.argv[2],"%Y-%m-%d")
    e_dt = dt.strptime(sys.argv[3],"%Y-%m-%d")
    tweetPre = sys.argv[4]
    K = int(sys.argv[5])
    t_topK = int(sys.argv[6])
    dataop = sys.argv[7]
    outfile = codecs.open(sys.argv[8], 'w', encoding = 'utf-8')
    thresh = int(sys.argv[9])
    knnNum = 30

############## readfile news. Two copies: lines (for scikit-learn's convenience) and objects (for later print use)
    numDays = (e_dt - s_dt).days
#    K = 20*(numDays+1)
    count = 0
    lines = []
    ind2obj = {}
    for x in range(numDays+1):
            fileDate = s_dt + tdelta(days = x)
            dtpure = fileDate.strftime("%Y-%m-%d")
            filename = newsPre + dtpure +".txt"
            if os.path.isfile(filename):
                file = codecs.open(filename, encoding = 'utf-8')
                count = readfile(file,dataop,count,ind2obj,dtpure,lines)
    vocab = None
############# clustering on news
    ent_surf = getSurface(ind2obj)
    ent_count = {}
    for e in ent_surf:
        surfs = ent_surf[e]
        count = sum(surfs.values())
        ent_count[e] = count
        outfile.write(str(count))
        outfile.write("\t"+e)
        for sf in surfs:
            outfile.write("\t"+sf+":\t"+str(surfs[sf]))
        outfile.write("\n")
    X,vectorizer = getVec(lines,vocab)
    terms = vectorizer.get_feature_names()
    DT = []
    titles = []

    for i in ind2obj:
#        curr_t = (ind2obj[i].created_at - s_dt).days+2
        curr_t = (ind2obj[i].created_at - s_dt).total_seconds()
        DT.append(curr_t)
        titles.append(ind2obj[i].title)

    Xp,termsp = getEntMat(ind2obj,0,ent_count,thresh)
    print("per entity done")
    Xl,termsl = getEntMat(ind2obj,1,ent_count,thresh)
    print("loc entity done")
    Xo,termso = getEntMat(ind2obj,2,ent_count,thresh)
    print("org entity done")
    X_all,terms_all = getEntMat(ind2obj,3,ent_count,thresh)
    print("all entity done")
            
    #clus2doc index of newsID
    clusModel,clus2doc,knn_graph = getCluster(X,K,knnNum,opts)
    labels = clusModel.labels_
    centers = clusModel.cluster_centers_
    for i in clus2doc:
        outfile.write(str(len(clus2doc[i]))+"\t")
 
    nDocs,nWords = X.shape
    Pz_d_km = np.zeros((K,nDocs))
    for i in range(nDocs):
        Pz_d_km[labels[i],i] = 1
    Pz_d_km = Pz_d_km +0.01;
    Pz_d_km = Pz_d_km / np.tile(sum(Pz_d_km),(K,1))
    C = centers.T+1/nWords/nWords
    Pw_z_km = C/np.tile(sum(C),(nWords,1))
    
    Learn = (1,10)
    X = X.tocoo()
    t0 = time()
    Pw_z,Pz_d,Pd,Li,Learn = pLSA.pLSA(X,K,Learn,Pz_d_km,Pw_z_km)
    print("pLSA done in %fs" % (time() - t0))
# print topics
    M = 50
    N = 10
    wordInd = Pw_z.argsort(axis=0)[::-1,:]
    docInd = Pz_d.argsort()[:,::-1]
    for i in range(K):
        sys.stdout.write("topic "+str(i))
        for j in range(M):
            sys.stdout.write('\t'+terms[wordInd[j,i]])
        sys.stdout.write('\n')
        for k in range(N):
            print(ind2obj[docInd[i,k]].title)


##############################################
    import pickle
    with open('test30.pickle', 'w') as f:
        pickle.dump([X,Xp,Xl,Xo,X_all,K,Learn,Pz_d_km,Pw_z_km,Pw_z,Pz_d,Pd,Li,\
                labels,terms,termsp,termsl,termso,terms_all,DT,ind2obj,clusModel],f)
    exit(0)
#######################
    order_centroids = clusModel.cluster_centers_.argsort()[:, ::-1]
    termsXX = terms +terms_all
    for i in clus2doc:
        indList = np.array(list(clus2doc[i]))
        ki = len(indList)/20
        if ki<2:          
          outfile.write("ki<2\n")
          for ind in order_centroids[i, :20]:
            outfile.write(' %s' % termsXX[ind])
          outfile.write("========\n")
          for ind in indList:
            outfile.write(ind2obj[ind].title+"\n")
            ents = ind2obj[ind].entities[3]
            for e in ents:
                outfile.write(e+":"+str(ents[e])+"\t")
            outfile.write("\n")
          outfile.write("***********************************************************\n")
          continue
        Xi = X[indList,:]
        clusModeli,clus2doci,knn_graphi = getCluster(Xi,ki,knnNum,opts) 
        order_centroidsi = clusModeli.cluster_centers_.argsort()[:, ::-1] 
        
        for j in range(ki):
            for ind in order_centroidsi[j, :20]:
                outfile.write(' %s' % termsXX[ind])
            outfile.write("========\n")
            indListj = indList[np.array(list(clus2doci[j]))]
            for ind in indListj:
                outfile.write(ind2obj[ind].title+"\n")
                ents = ind2obj[ind].entities[3]
                for e in ents:
                    outfile.write(e+":"+str(ents[e])+"\t")
                outfile.write("\n")
            outfile.write("---------------------------------------\n")
        outfile.write("***********************************************************\n")
    exit(0)



############# print cluster + rank tweets    
    # place holder
    order_centroids=None
    terms=vectorizer.get_feature_names()

    for i in range(k):
        #outfile.write('%s' % maxDist)
        printCluster(X,i,lines,clus2doc,clusModel,order_centroids,terms,outfile,ind2obj,t_topK,vectorizer,tweetPre,opts)
        outfile.write('*****************\n')
