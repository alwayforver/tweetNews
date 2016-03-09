import operator
import codecs,unidecode
import glob
from time import time

from pLSABet_reduceK import pLSABet,Descriptor
import backendDefs as bk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.stats import entropy
from scipy.sparse import coo_matrix
import numpy as np
import sys,os
DEBUG=1

def calc_Pw_z(labels, X, K):
#    X = X.tocsr()
    nWords = X.shape[1]
    Pw_z = np.zeros((nWords,K))
    for i in range(K):
        Pw_z[:,i] = X.tocsr()[labels==i,:].mean(0)
    C = Pw_z+1e-7 #1.0/nEnt/nEnt
    Pw_z = C/np.tile(sum(C),(nWords,1))
    return Pw_z

def inittime(DT,K,labels):
    mu = np.zeros(K)
    sigma = np.zeros(K)
    if not len(DT):
        return mu,sigma
    for i in range(K):
        ts = np.array(DT)[labels==i]
        mu[i] = np.mean(ts)
        sigma[i] = np.std(ts)
    return mu,sigma
def printTerms(M,terms,wordInd,i):
    for j in range(min(M,wordInd.shape[0])):
        sys.stdout.write('\t'+terms[wordInd[j,i]].split('_|')[0])
    sys.stdout.write('\n')

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
        # convert url to http    
        if raw_text.split()[-1].startswith('http'):
            raw_text = raw_text.split('http')[0] + 'http'
        #if "http" not in raw_text and "RT @" not in raw_text \
        raw_text=unidecode.unidecode(raw_text)
        if not raw_text.startswith('RT @') \
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

   
def printTermsStats(M,terms,wordInd,score_w_z,i,Pw,Lw_z,Sw,Pw_z):
    for j in range(min(M,wordInd.shape[0])):
        w = wordInd[j,i]
        sys.stdout.write('\t'+terms[w].split('_|')[0]+':'+str(score_w_z[w,i])+'|:'+str(Pw[w])+':'+str(Lw_z[w,i])+':'+str(Sw[w])+':'+str(Pw_z[w,i]))
    sys.stdout.write('\n')
    print "--orig: "
    for j in range(min(M,Pw_z.shape[0])):
        w = Pw_z[:,i].argsort()[::-1][j]
        sys.stdout.write('\t'+terms[w].split('_|')[0]+':'+str(Pw_z[w,i]))
    sys.stdout.write('\n')
    print "-----------"

def printNewsClusterStats(i,termsList,wordIndList,score_w_zs,Pws,Lw_zs,Sws,Pw_zs):
    print "Cluster: ",i
    M = 50 # max number of terms to output
    for dim in range(len(termsList)):            
        print("dim "+str(dim)+": ")
        printTermsStats(M,termsList[dim],wordIndList[dim],score_w_zs[dim],i,Pws[dim],Lw_zs[dim],Sws[dim],Pw_zs[dim])
    print("=========")
   
def printNewsClusterText(i,termsList,ind2obj,t_topK,tweetPre,cosine_d_z,Pw_z,Pz_d,docInd,dID=[]):
    print "Cluster: ",i
    tweets = []
    tweetsObj = []
    M = 50 # max number of terms to output
    N = 1000 # max number of news output
    tweetIDset = set()
    tweetSet = set()
    resDocInd = [] # index in X and ind2obj
    resTweetCount = 0 # max index in tweetsObj

    for k in range(min(N,docInd.shape[1])):
        docIDinX = docInd[i,k]
        news_cosine= cosine_d_z[docIDinX,i]
        news_topic_score=Pz_d[i,docIDinX]
        if news_topic_score < 0.4: # threshold Pz_d
            continue
        if not len(dID):
            news = ind2obj[docIDinX]
            resDocInd.append(docIDinX)
        else:
            news = ind2obj[dID[docIDinX]]
            resDocInd.append(dID[docIDinX])

        #print "NEWS-- "+news.ID+"\t"+str(news_cosine)+"\t"+str(news_topic_score)+"\t"+str(news.created_at)+'\t'+news.title)+'\t'+news.raw_text.split('.')[0]
        print "NEWS--"+str(i)+"\t"+news.ID+"\t"+str(news_cosine)+"\t"+str(news_topic_score)+"\t"+'\t'+news.title+'\t'+' '.join(news.raw_text.encode('utf-8').split()[0:60])

        #print("-------")
        if tweetPre == 'null':
            continue
        newsID = news.ID
        dtpure = news.dtpure
        addtweets,addtweetsObj = getRelTweets(newsID,dtpure,tweetPre,tweetIDset,tweetSet)           
        tweets = tweets + addtweets
        tweetsObj = tweetsObj + addtweetsObj
    if tweetPre != 'null' and tweets:
        newsCenter = Pw_z[:,i]
        topTweetsObj,topTweetsScore = rankTweets(tweets,tweetsObj, newsCenter, termsList[0],t_topK)
        print("*******total tweets: "+str(len(tweets)))
        print("top tweets:")
        # for t in sorted(topTweetsObj, key=operator.attrgetter('created_at')): # sort by time
        
        for t in topTweetsObj: # sort by cosine score
            if topTweetsScore[t.ID] > 0.001:
                print("TWEET--"+str(i)+"\t"+str(topTweetsScore[t.ID])+"\t"+str(t.created_at)+"\t" + t.raw_text )
                resTweetCount+=1
    else:
        print("no tweets retrieved")
    print("=========")
    return resDocInd,resTweetCount,topTweetsObj,topTweetsScore

def getStats(Xs,Xinds,Pw_zs,Pz_d,Pd,K):
    t0 = time()
    sys.stderr.write('Get stats...\n')
    # Pws
    Pws = []
    # lift
    Lw_zs = []
    Sws = []
    for i in range(len(Xinds)):
        # Pw
        Pw = np.squeeze(np.asarray(Xs[Xinds[i]].sum(axis=0)))
        Pw+=1e-10  # smoothing to be refined
        Pw = Pw/Pw.sum()
        Pws.append(Pw/np.mean(Pw)) ###### better presentation
        # Lw_z
        Lw_z = Pw_zs[i]/np.tile(Pw,(Pw_zs[i].shape[1],1)).T
        Lw_zs.append(Lw_z)
        # Sw salience: KL(Pz_w,Pz)
        # -Pz
        Pz = Pz_d.dot(Pd)
        # -Pz_w
        numWords = Pw_zs[i].shape[0]
        Pz_w = (Pw_zs[i]*Pz).T
        Pz = Pz[:K+1] # optional remove background topic
        Pz_w = Pz_w[:K+1,:] # optional remove background topic
        Pz_w = normalize(Pz_w+1e-10, norm='l1')    # smoothing. need care
        Sw = entropy(Pz_w,np.tile(Pz,(numWords,1)).T)        
        Sws.append(Sw)
        #  context
    print( "pLSA stats done in "+str(time() - t0))
    return Pws,Lw_zs,Sws


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

def init_all(K,Xs,DT,mini=0,n_init=30,init_size=20000,batch_size=500,verbose=False):
    if mini:
        km = MiniBatchKMeans(n_clusters=K, init='k-means++', n_init=n_init,
            init_size=init_size, batch_size=batch_size, verbose=verbose)
    else:
        km = KMeans(n_clusters=K, init='k-means++', n_init=n_init, verbose=verbose)
    t0=time()
    km.fit(Xs[0])
    print "Kmeans done in (sec): ", time() - t0
    labels = km.labels_
    centers = km.cluster_centers_
    # print number of doc in each cluster
    clus2doc = {}
    for i in range(len(labels)):
        clus2doc[labels[i]] = clus2doc.get(labels[i],set())
        clus2doc[labels[i]].add(i)    
    if len(clus2doc) < K:
        K_ = len(clus2doc)
        sys.stderr.write("kmeans reduce K to "+str(K_))
        exit(0)
    if DEBUG==1:
        print "clustering stats: "
        for i in clus2doc:
            print (str(i)+"\t"+str(len(clus2doc[i])))        
    # init                                                                                
    nDocs,nWords = Xs[0].shape
    Pz_d = np.zeros((K,nDocs))
    for i in range(nDocs):
        Pz_d[labels[i],i] = 1
    Pz_d = Pz_d +0.01;
    Pz_d = Pz_d / np.tile(sum(Pz_d),(K,1))
    Pw_zs = []
    C = centers.T+1/nWords/nWords
    Pw_z = C/np.tile(sum(C),(nWords,1))
    Pw_zs.append(Pw_z)
    for i in range(1,len(Xs)):
        Pw_zs.append( calc_Pw_z(labels, Xs[i], K) )
    mu, sigma = inittime(DT,K,labels)
    Pd = np.ones(Pz_d.shape[1])/float(Pz_d.shape[1])  # To Do: better scheme to initialize Pd
    inits = Descriptor(Pw_zs,Pz_d,Pd,mu,sigma)
    # return Pz_d_km,Pw_z_km, ..., mu_km, sigma_km
    return inits,labels,centers

def subRun(Xs,n_wdxPz_wds,K,params,DT=[],dIDParent=[]):
    Xevents,dID = selectTopic(Xs,n_wdxPz_wds,params.eventID)
    if DT:
        DTevent = np.array(DT)[dID]
    else:
        DTevent = []
        sys.stderr.write('no DT. \n')
    inits,labels,centers = init_all(K,Xevents,DTevent,mini=0)   # labels and centers not used 
    if len(dIDParent)>0:
        dID = dIDParent[dID]
    eventNode = EventNode(Xevents,DTevent,params,inits,dID=dID)
    eventNode.run()
    return eventNode

def nextData(rootNode):
    params = rootNode.params
    desc = rootNode.descriptor
    Xs = rootNode.Xs
    numX = len(params.Xinds)
    n_wdxPz_wds = []
    XsWeighted = []
    for dim in range(numX):
        X = params.ws[dim]*Xs[params.Xinds[dim]].tocoo()
        n_wdxPz_wds.append(weightX(X,
            desc.Pw_zs[dim],
            desc.Pz_d) )
        XsWeighted.append(X)
    return n_wdxPz_wds,XsWeighted


class NodeParams:
    def __init__(self,Xinds,ws,lambdaB,selectTime=0,wt=0.5,Learn=(1,10),eventID=0,debug=1):
        self.eventID = eventID
        self.Xinds = Xinds
        self.ws=ws
        self.wt=wt
        self.selectTime = selectTime
        self.lambdaB=lambdaB
        self.Learn = Learn
        self.debug=debug
        # after run():
        # Pw_zs,Pz_d,Pd,mu,sigma,Li
        
class EventNode:
    def __init__(self,Xs,DT=[],params=None,initsDescriptor=None,descriptor=None,dID=[],debug=1):
        self.Xs = Xs  #
        self.DT = DT  #
        self.params=params   #        
        if initsDescriptor:
            self.initsDescriptor = initsDescriptor  #            
        if descriptor:
            self.descriptor = descriptor  #
        self.dID=dID #
    def run(self):
        t0 = time()
        params = self.params
        numX = len(params.Xinds)
        data = []  
        Pw_zs = []    ####
        for i in range(numX):
            data.append(params.ws[i]*self.Xs[params.Xinds[i]])
            Pw_zs.append(self.initsDescriptor.Pw_zs[params.Xinds[i]])  ####
        self.initsDescriptor.Pw_zs = Pw_zs  ####
        sys.stderr.write('PLSABet run at node '+str(params.eventID)+'...\n')
        self.descriptor = pLSABet(data,self.initsDescriptor,params.lambdaB,
                params.selectTime,self.DT,params.wt,params.Learn,params.debug)  #
        if self.descriptor is None:
            sys.stderr.write("############### Pw_zs = None. Reduce K!!! \n") 
            exit(-1)
        print( "pLSA done in "+str(time() - t0))
    def printCluster(self,vects,ind2obj,
            t_topK=10000,tweetPre='null',switch='text',fromPlsa=True):
        desc = self.descriptor if fromPlsa else self.initsDescriptor
        Pw_zs = desc.Pw_zs
        Pz_d = desc.Pz_d
        Pd = desc.Pd
        K = Pz_d.shape[0]
        if fromPlsa:
            K-=1
        Xinds = self.params.Xinds if fromPlsa else range(len(self.Xs))

        Pws,Lw_zs,Sws = getStats(self.Xs,Xinds,Pw_zs,Pz_d,Pd,K) # be carefully with Xs. Can be data
        sys.stderr.write('Printing stats text...\n')    
        numX = len(Xinds)
        termsList= []
        wordIndList = []
        score_w_zs = []
        for dim in range(numX):
            termsList.append(vects[Xinds[dim]].get_feature_names())    
            score_z_w = (Lw_zs[dim].T>0) * np.log(Lw_zs[dim].T+1) * Pw_zs[dim].T * np.log(Pws[dim]+1) * Sws[dim]
            score_w_zs.append(score_z_w.T)
            wordIndList.append( score_z_w.T.argsort(axis=0)[::-1,:] )
        if switch == 'text':
            cosine_d_z = self.Xs[0].dot(Pw_zs[0])
            #if fromPlsa:
            #    docInd = Pz_d.argsort()[:,::-1]
            #else:
            docInd = cosine_d_z.T.argsort()[:,::-1]
        for i in range(K):
            printNewsClusterStats(i,termsList,wordIndList,score_w_zs,Pws,Lw_zs,Sws,Pw_zs)
            if switch == 'text':
                printNewsClusterText(i,termsList,ind2obj,t_topK,tweetPre,cosine_d_z,Pw_zs[0],Pz_d,docInd,self.dID)

    # to do
    def printSumCluster(self,vects,ind2obj,
            t_topK=10000,tweetPre='null',switch='text',fromPlsa=True):
        desc = self.descriptor if fromPlsa else self.initsDescriptor
        Pw_zs = desc.Pw_zs
        Pz_d = desc.Pz_d
        Pd = desc.Pd
        K = Pz_d.shape[0]
        if fromPlsa:
            K-=1
        Xinds = self.params.Xinds if fromPlsa else range(len(self.Xs))

        Pws,Lw_zs,Sws = getStats(self.Xs,Xinds,Pw_zs,Pz_d,Pd,K) # be carefully with Xs. Can be data
        sys.stderr.write('Printing stats text...\n')    
        numX = len(Xinds)
        termsList= []
        wordIndList = []
        score_w_zs = []
        for dim in range(numX):
            termsList.append(vects[Xinds[dim]].get_feature_names())    
            score_z_w = (Lw_zs[dim].T>0) * np.log(Lw_zs[dim].T+1) * Pw_zs[dim].T * np.log(Pws[dim]+1) * Sws[dim]
            score_w_zs.append(score_z_w.T)
            wordIndList.append( score_z_w.T.argsort(axis=0)[::-1,:] )
        if switch == 'text':
            cosine_d_z = self.Xs[0].dot(Pw_zs[0])
            docInd = cosine_d_z.T.argsort()[:,::-1]
        for i in range(K):
            printNewsClusterStats(i,termsList,wordIndList,score_w_zs,Pws,Lw_zs,Sws,Pw_zs)
            if switch == 'text':
                printNewsClusterText(i,termsList,ind2obj,t_topK,tweetPre,cosine_d_z,Pw_zs[0],Pz_d,docInd,self.dID)
        # print summary
        relNews = getRel
        #relTweets = 
