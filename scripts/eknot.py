from __future__ import print_function
import operator
import codecs
import glob
import backendDefs as bk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time

import numpy as np
import sys,os

import pLSABet_reduceK as pLSABet
import pickle
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getwriter('utf-8')(sys.stdin)

def calc_Pw_z(labels, X, K):
#    X = X.tocsr()
    nWords = X.shape[1]
    Pw_z = np.zeros((nWords,K))
    for i in range(K):
        Pw_z[:,i] = X.tocsr()[labels==i,:].mean(0)
    C = Pw_z+1e-7 #1.0/nEnt/nEnt
    Pw_z = C/np.tile(sum(C),(nWords,1))
    return Pw_z
def init_all(K,Xs,DT):
    km = MiniBatchKMeans(n_clusters=K, init='k-means++', n_init=10,init_size=1000,
            batch_size=1000,verbose=True)

#    km = KMeans(n_clusters=K, init='k-means++', max_iter=100, n_init=50)
    km.fit(Xs[0])
    labels = km.labels_
    centers = km.cluster_centers_
    # print number of doc in each cluster
    clus2doc = {}
    for i in range(len(labels)):
        clus2doc[labels[i]] = clus2doc.get(labels[i],set())
        clus2doc[labels[i]].add(i)    
    if len(clus2doc) < K:
        K_ = len(clus2doc)
        print (str(K_)+" clusters")
        print("kmeans reduce K to "+str(K_))
        exit(0)
#####        return init_all(K_,X,DT)
    for i in clus2doc:
        print (str(i+1)+"\t"+str(len(clus2doc[i])))        
    # init                                                                                
    inits = []
    nDocs,nWords = Xs[0].shape
    Pz_d_km = np.zeros((K,nDocs))
    for i in range(nDocs):
        Pz_d_km[labels[i],i] = 1
    Pz_d_km = Pz_d_km +0.01;
    Pz_d_km = Pz_d_km / np.tile(sum(Pz_d_km),(K,1))
    inits.append(Pz_d_km)
    C = centers.T+1/nWords/nWords
    Pw_z_km = C/np.tile(sum(C),(nWords,1))
    inits.append(Pw_z_km)
    for i in range(1,len(Xs)):
        inits.append( calc_Pw_z(labels, Xs[i], K) )
    mu_km, sigma_km= inittime(DT,K,labels)
    inits.append(mu_km)
    inits.append(sigma_km)
    # return (K,[Pz_d_km,Pw_z_km,mu_km, sigma_km])
    return inits


def getRelTweets(newsID,dtpure,tweetPre,tweetIDset,tweetSet):
    t_path = glob.glob(tweetPre+dtpure+"/"+str(newsID)+"_*")
    if len(t_path) != 1:
        print('no tweets for news ',newsID,'len(t_path)',len(t_path))
        return ([],[])
    if os.path.exists(t_path[0]):
        t = codecs.open(t_path[0], encoding = 'utf-8') 
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
        if "http" not in raw_text and "RT @" not in raw_text \
            and ID not in tweetIDset and raw_text not in tweetSet:
            tweet = bk.Tweet(ID,raw_text,created_at,is_retweet,retweet_count,hash_tags)
            tweetsObj.append(tweet)
            tweets.append(raw_text)
            tweetIDset.add(ID)
            tweetSet.add(raw_text)
    t.close()
    return (tweets,tweetsObj)
def rankTweets(tweets, tweetsObj, newsVec, vocab, t_topK):
    tweetVectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=2, stop_words='english',
                                 vocabulary=vocab)
    X = tweetVectorizer.fit_transform(tweets)
    scores = X.dot(newsVec)
    tweetsInd = scores.argsort()[::-1][:t_topK]
    topTweetsObj = [tweetsObj[i] for i in tweetsInd]
    topTweetsScore = {}
    for i in tweetsInd:
        topTweetsScore[tweetsObj[i].ID] = scores[i]
    return topTweetsObj,topTweetsScore

def printTerms(M,terms,wordInd,i,outfilen):
    for j in range(min(M,wordInd.shape[0])):
        sys.stdout.write('\t'+terms[wordInd[j,i]])
        outfilen.write('\t'+terms[wordInd[j,i]])
    sys.stdout.write('\n')
    outfilen.write('\n')

def printCluster(X,i,termsList,outfile,ind2obj,t_topK,tweetPre,Pw_z,Pz_d,wordIndList,docInd):
        outfilen = codecs.open(outfile+"news_"+str(i), 'w', encoding = 'utf-8')
        outfilet = codecs.open(outfile+"tweets_"+str(i), 'w', encoding = 'utf-8')
        print("Cluster %d:" % i, end='')
        print()
        tweets = []
        tweetsObj = []
        M = 50
        N = 1000
        for dim in range(len(termsList)):            
            print("dim "+str(dim)+": ")
            outfilen.write("dim "+str(dim)+": ")
            printTerms(M,termsList[dim],wordIndList[dim],i,outfilen)
#        for j in range(M):
#            sys.stdout.write('\t'+terms[wordInd[j,i]])
#        sys.stdout.write('\n')
        tweetIDset = set()
        tweetSet = set()
        for k in range(N):
            docIDinX = docInd[i,k]
            news_cosine=np.dot(Pw_z[:,i],X.toarray()[docIDinX,:])
            news_score=Pz_d[i,docIDinX]
            if news_score < 0.4:
                break
            news = ind2obj[docIDinX]

            print(str(news_cosine)+"\t"+str(news_score)+"\t"+news.title)


#        newsList = [ind2obj[ind] for ind in clus2doc[i]]
#        for news in sorted(newsList, key=operator.attrgetter('created_at')):
#        for ind in clus2doc[i]:
#            news = ind2obj[ind]

#            print(str(news.created_at)+"\t"+news.title) #+"\t"+news.raw_text+"\t"+news.source)
            #outfilen.write(str(news_cosine)+"\t"+str(news_score)+"\t"+str(news.created_at)+"\t"+news.title+"\n")
            outfilen.write(str(news.created_at)+"\t"+news.title+"\t"+news.raw_text+"\t"+news.source+"\n")
            print("-------")
            newsID = news.ID
            dtpure = news.dtpure
            #if getRelTweets(newsID,dtpure,tweetPre, tweetIDset,tweetSet):
            addtweets,addtweetsObj = getRelTweets(newsID,dtpure,tweetPre,tweetIDset,tweetSet)           
            tweets = tweets + addtweets
            tweetsObj = tweetsObj + addtweetsObj
    #            tweets = tweets | getRelTweets(newsID)
    #    tweets = list(tweets)
        if tweets:
            newsCenter = Pw_z[:,i]
            #newsCenter = np.squeeze(np.asarray(getNewsCenter(X,clus2doc[i])))
            for term in newsCenter.argsort()[::-1][:20]:
                print(' %s' % terms[term], end='')
            #topTweets = rankTweets(tweets, clusModel.cluster_centers_[i,:], vectorizer.vocabulary_,t_topK)
            topTweetsObj,topTweetsScore = rankTweets(tweets,tweetsObj, newsCenter, terms,t_topK)
            print("*******total tweets: "+str(len(tweets)))
            print("top tweets:")
            for t in sorted(topTweetsObj, key=operator.attrgetter('created_at')):
                print(str(topTweetsScore[t.ID])+"\t"+str(t.created_at)+"\t" + t.raw_text )
                #outfilet.write(str(topTweetsScore[t.ID])+"\t"+str(t.created_at)+"\t" + t.raw_text+"\n")
                outfilet.write(str(t.created_at)+"\t" + t.raw_text +"\t" + str(t.retweet_count) + "\t"+t.hash_tags +"\n")
                print("-------")
        else:
            print("no tweets retrieved")
            outfilet.write("no tweets retrieved\n")
        print("=========")
        print()
def inittime(DT,K,labels):
    mu = np.zeros(K)
    sigma = np.zeros(K)
    for i in range(K):
        ts = np.array(DT)[labels==i]
        mu[i] = np.mean(ts)
        sigma[i] = np.std(ts)
    return mu,sigma
                    

if __name__ == "__main__":
    # input args: K display outputdir pickle_file
    if os.path.isfile(sys.argv[4]):
        with open(sys.argv[4]) as f:
            [X,Xp,Xl,Xo,X_all,K,Learn,Pz_d_km,Pw_z_km,Pw_z,Pz_d,Pd,Li,\
                    labels,terms,termsp,termsl,termso,terms_all,DT,ind2obj,clusModel]=pickle.load(f)
    else:
        print "no pickle"
        exit(-1)
    Xs = [X,Xp,Xl,Xo]
    data = Xs+[DT]
    if K!=int(sys.argv[1]):
        K=int(sys.argv[1])
        inits = init_all(K,Xs,DT)
    else:
        mu_km, sigma_km= inittime(DT,K,labels)
        inits = [Pz_d_km,Pw_z_km,Pp_z_km,Pl_z_km,Po_z_km,mu_km,sigma_km]
    t0 = time()
    Learn=(1,10)
    selectTime = 1
    numX = 4
    #data = [X, DT]
    wt = 0.5
    lambdaB = 0.5
    # data = [Xs,DT]
    # inits = [Pz_d,Pw_z, Pp_z,Pl_z,Po_z,mu,sigma]        
    
    Pw_zs,Pz_d,Pd,mu,sigma,Li = pLSABet.pLSABet(selectTime,numX,Learn,data,inits,wt,lambdaB,1)
    while Pw_zs == None:
        K-=1
        print("###############reduce K to " +str(K))
        inits = init_all(K,Xs,DT)
        Pw_zs,Pz_d,Pd,mu,sigma,Li = pLSABet.pLSABet(selectTime,numX,Learn,data,inits,wt,lambdaB,1)
    print( "pLSA done in "+str(time() - t0))
    tweetPre="/srv/data/jingjing/eknot/tweets/"
    # tweetPre="/home/wtong8/NewsTwitter/tweets/"
    outfile = sys.argv[3]
    M = 50
    N = 10
    termsList = [terms,termsp,termsl,termso]
    wordIndList = []
    for dim in range(numX):
        wordIndList.append( Pw_zs[dim].argsort(axis=0)[::-1,:] )
    docInd = Pz_d.argsort()[:,::-1]
    t_topK=20000
    for i in range(K):
        if (Pz_d[i,:]>lambdaB-0.1).sum()<20:
            continue
        printCluster(Xs[0],i,termsList,outfile,ind2obj,t_topK,tweetPre,Pw_zs[0],Pz_d,wordIndList,docInd)
    exit(0)
    ###################### split event###########################
    def weightX(X,Pw_z,Pz_d):
        K = Pz_d.shape[0]
        X = X.tocoo()
        docind,wordind,value = (X.row,X.col,X.data)
        # Pz_do_f = Pz_do.*(Pz_do>(1-lambdaB)/double(K-1))
        Pz_d_f = Pz_d*(Pz_d>0.01)
        Pz_dw_ = Pw_z[wordind,:].T*Pz_d_f[:,docind]
        Pw_d = Pz_dw_.sum(axis=0) # 1 x nnz
        Pz_wd = Pz_dw_[:-1,:]/np.tile(Pw_d,(K-1,1))
        n_wdxPz_wd = np.tile(value,(K-1,1))*Pz_wd
        n_wdxPz_wd = n_wdxPz_wd *(n_wdxPz_wd>0.0001) ####
        return n_wdxPz_wd
    
    n_wdxPz_wds = []
    for i in range(numX):
        n_wdxPz_wds.append(  weightX(data[i],Pw_zs[i],Pz_d) )
    from scipy.sparse import coo_matrix
    # get event matrices
    def selectTopic(Xs,n_wdxPz_wds,event):
        Xevents = []    
        for i in range(len(Xs)):
            X = Xs[i]
            n_wdxPz_wd = n_wdxPz_wds[i]
            nDocs,nWords=X.shape
            docind,wordind,value = (X.row,X.col,X.data)     
            value = n_wdxPz_wd[event,:]
            select = (value!=0)
            value_f = value[select]
            row_f = docind[select]  # 1 3 3 5 5 6 6 6 
            col_f = wordind[select]  
            if i==0:
                dID = np.unique(row_f) # 1 3 5 6
            dID2ind = -np.ones(nDocs) # -1 -1 -1 -1 -1 -1 -1 assume nDocs = 7
            dID2ind[dID] = np.arange(len(dID))  # 0 0 1 0 2 3 0
            row_f_new = dID2ind[row_f]  # 0 1 1 2 2 3 3 3
            if i>0:
                select = (row_f_new!=-1)
                Xevent = coo_matrix((value_f[select],(row_f_new[select],col_f[select])),shape=(len(dID),nWords))
            else:
                Xevent = coo_matrix((value_f,(row_f_new,col_f)),shape=(len(dID),nWords))
            Xevents.append(Xevent)        
        return Xevents,dID
    ###################### step 1 ###############
    event = 0 # event number
    Xevents,dID = selectTopic(data[:numX],n_wdxPz_wds,event)
    DTevent = np.array(DT)[dID]
    #data = Xevents+[DTevent] 
    ########################## event example ###############
    Kevent=5
    km = KMeans(n_clusters=Kevent, init='k-means++', max_iter=100, n_init=5)
    km.fit(Xevents[0])
    labels = km.labels_
    centers = km.cluster_centers_
    
    nDocs,nWords = Xevents[0].shape
    Pz_d_km = np.zeros((Kevent,nDocs))
    for i in range(nDocs):
        Pz_d_km[labels[i],i] = 1
    Pz_d_km = Pz_d_km +0.01;
    Pz_d_km = Pz_d_km / np.tile(sum(Pz_d_km),(Kevent,1))
    C = centers.T+1/nWords/nWords
    Pw_z_km = C/np.tile(sum(C),(nWords,1))
    
    t0 = time()
    Learn=(1,10)
    selectTime = 1
    numX = 1
    #K=30
    data = [Xevents[0], DTevent]
    mu_km, sigma_km= inittime(DTevent,Kevent,labels)
    inits = [Pz_d_km,Pw_z_km,mu_km,sigma_km]
    wt = 0.1
    lambdaB = 0.5
    # data = [Xs,DT]
    # inits = [Pz_d,Pw_z, Pp_z,Pl_z,Po_z,mu,sigma]        
    Pw_zs,Pz_d,Pd,mu,sigma,Li = pLSABet.pLSABet(selectTime,numX,Learn,data,inits,wt,lambdaB,1)
    
    print ("pLSA done in "+str(time() - t0))
    #################################################################################
    # print topics
    display = int(sys.argv[2])
    if display == 1:
        M = 50
        N = 10
        wordInd = Pw_zs[0].argsort(axis=0)[::-1,:]
        docInd = Pz_d.argsort()[:,::-1]
        for i in range(Kevent): #
            sys.stdout.write("topic "+str(i))
            for j in range(M):
                sys.stdout.write('\t'+terms[wordInd[j,i]])
            sys.stdout.write('\n')
            for k in range(N):
                print(ind2obj[dID[docInd[i,k]]].title) #
