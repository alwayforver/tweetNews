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
from scipy.sparse import coo_matrix,csr_matrix
import numpy as np
import sys,os
from utils import dbpedia,parseEntity,my_tokenizer,tweet_tokenizer,news_tokenizer,rep1,rep2,processTextEnt,processText,grep_ent
DEBUG=1
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

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

def getRelTweets(newsID,dtpure,tweetPre,tweetsObj):
    t_path = glob.glob(tweetPre+dtpure+"/"+str(newsID)+"_*")
    if len(t_path) != 1:
        print('no tweets for news ',newsID,'len(t_path)',len(t_path))
        return
    if os.path.exists(t_path[0]):
        t = codecs.open(t_path[0], encoding = 'utf-8') 
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

        if ID in tweetsObj or len(raw_text.split())<=5:
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
        raw_text=unidecode.unidecode(raw_text)
        tweet = bk.Tweet(ID,raw_text,created_at,is_retweet,retweet_count,hash_tags,pop=1)
        tweetsObj[ID] = tweet
    t.close()

def dedupTweets(tweetsObj,vocab):
    tweetVectorizerBinary = TfidfVectorizer(stop_words='english',vocabulary=vocab,binary=True,
            tokenizer=lambda text: tweet_tokenizer(text,'reg'))
    i2ID = {}
    tweets = []
    i=0
    for tID in tweetsObj:
        tweets.append(tweetsObj[tID].raw_text)
        i2ID[i] = tID
        i+=1
    X = tweetVectorizerBinary.fit_transform(tweets)

    indices = X.indices
    indptr = X.indptr
    tweetsObjDedup = {}
    signatures = {}
    count = 0 
    for i in range(X.shape[0]):
        if indptr[i+1] - indptr[i] <= 5:
            continue
        inds = indices[indptr[i]:indptr[i+1]]
        sig = inds.tostring()
        tID = i2ID[i]
        if sig not in signatures:
            signatures[sig] = tID
            tweetsObjDedup[tID] = tweetsObj[tID]
            tweetsObjDedup[tID].tokens = [vocab[ind] for ind in inds]
        else:
            tID_exist = signatures[sig]
            tweetsObjDedup[tID_exist].pop += 1
            tweetsObjDedup[tID_exist].dupIDs.add(tID)
        count +=1
        if count%5000==1:            
            sys.stderr.write('Dedup...'+str(count)+'\n')
    # dbpedia get ent
    for tID in tweetsObjDedup:
        t = tweetsObjDedup[tID]
        t.tokens_ent = unidecode.unidecode(processTextEnt(t.raw_text))
        t.repID = tID
        if 'RT @' not in t.raw_text:
            continue
        for duptID in t.dupIDs:
            trep = tweetsObj[duptID]
            if 'RT @' not in trep.raw_text:
                t.repID = duptID
                trep.tokens_ent = unidecode.unidecode(processTextEnt(trep.raw_text))
                break
    return tweetsObjDedup            

    
def rankTweets(tweetsObj, newsVec, vocab):
    tweetVectorizer = TfidfVectorizer(stop_words='english',vocabulary=vocab,use_idf=False,
            tokenizer=lambda text: tweet_tokenizer(text,'reg'))
    tweetsScore = {}
    tweets = []
    i2ID = {}
    i = 0
    for tID in tweetsObj:
        tweets.append(tweetsObj[tID].raw_text)
        i2ID[i] = tID
        i+=1
    X = tweetVectorizer.fit_transform(tweets)
    scores = X.dot(newsVec)
    for i in range(len(tweetsObj)):
        tweetsScore[i2ID[i]] = scores[i]
    #tweetsInd = scores.argsort()[::-1][:t_topK]
    #for i in tweetsInd:
    #    topTweetsScore[i2ID[i]] = scores[i]
    return tweetsScore

   
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
    sys.stderr.write("Cluster: "+str(i)+'\n')
    M = 50 # max number of terms to output
    N = 1000 # max number of news output
    resDocInd = [] # index in X and ind2obj
    tweetsObj = {}
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

        print "NEWS--" + '\t'.join([str(i), news.ID, str(news_cosine), 
            str(news_topic_score),str(news.created_at),news.title, news.raw_text.encode('utf-8')[:200] ])

        if tweetPre == 'null':
            continue
        getRelTweets(news.ID,news.dtpure,tweetPre,tweetsObj)
        sys.stderr.write("*******current tl tweets: "+str(len(tweetsObj))+'\n')
    if not tweetsObj:
        return resDocInd,None,None
    
    sys.stderr.write('Dedup Tweets...\n')
    tweetsObjDedup = dedupTweets(tweetsObj,termsList[0])
    print("*******after dedup: "+str(len(tweetsObjDedup)))
    sys.stderr.write("*******after dedup: "+str(len(tweetsObjDedup))+'\n')
    print("*******total tweets: "+str(len(tweetsObj)))
    sys.stderr.write("*******total tweets: "+str(len(tweetsObj))+'\n')
    if tweetPre != 'null' and tweetsObjDedup:
        newsCenter = Pw_z[:,i]
        tweetsScore = rankTweets(tweetsObjDedup, newsCenter, termsList[0])
        print("top tweets:")
        count = 0
        for tID_score in sorted(tweetsScore.items(), key=operator.itemgetter(1),reverse=True):
            tScore = tID_score[1]
            if tScore > 0.001:
                tID = tID_score[0]
                t = tweetsObjDedup[tID]
                print "TWEET--"+ '\t'.join([ str(i), str(tID), str(tScore), 
                    str(t.pop), str(t.created_at), t.raw_text, t.tokens_ent,
                    tweetsObj[t.repID].raw_text, 
                    tweetsObj[t.repID].tokens_ent, 
                    str(t.tokens) ])
                count += 1
                if t_topK>0 and count >= t_topK:
                    break
            else:
                break
    else:
        print("no tweets retrieved")
    print("=========")
    return resDocInd,tweetsObj,tweetsObjDedup,tweetsScore

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
    def printCluster_i(self,vects,ind2obj,i,
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
        # print i
        printNewsClusterStats(i,termsList,wordIndList,score_w_zs,Pws,Lw_zs,Sws,Pw_zs)
        if switch == 'text':
            resDocInd,tweetsObj,tweetsObjDedup,tweetsScore = printNewsClusterText(i,termsList,ind2obj,t_topK,tweetPre,cosine_d_z,Pw_zs[0],Pz_d,docInd,self.dID)
        return resDocInd,tweetsObj,tweetsObjDedup,tweetsScore


def getEntInd(evocab,Pe_z,topK):
    ent_ind = {}
    inds = Pe_z.argsort()[::-1][:topK]
    ents = []
    count = 0
    for i in inds:
        ent_ind[evocab[i]] = count
        ents.append(evocab[i])
        count+=1
    return ent_ind,ents
def getEntScore(evocab,Pe_z,topK):
    score = {}
    inds = Pe_z.argsort()[::-1][:topK]
    ents = []
    for i in inds:
        score[evocab[i]] = Pe_z[i]
        ents.append(evocab[i])
    return score,ents

def makeEntText(sentence,ent_text,ent_ind,indices,indptr,window):
    if len(sentence) < 50: # or len(sentence)>140:
        return ''
    sentence = sentence.lower()
    ents = grep_ent(sentence,True,True,True,True).split()
#    if not ents:
#        return ''
    s = grep_ent(sentence,False,False,False,False)
    if len(s) < 50:
        return ''
    count = 0
    
    half = window//2
    prev = 0
    for e in ents:        
#        e = e.lower()
        if e in ent_text:            
            cur = sentence.find(e,prev)
            context = ' '.join(sentence[:cur].split()[-half:] + \
                    sentence[cur:].split()[0:half+1])
            s_context = grep_ent(context,False,False,False,False)
            prev = cur+1

            indices.append(ent_ind[e])
            ent_text[e] += ' '
            ent_text[e] += s_context
            count += 1
#    if count:      
    last = indptr[-1]
    indptr.append(last+count)
    return s
#    else:
#        return ''


def makeEntTextOld(sentence,ent_text,ent_ind,indices,indptr):        
    if len(sentence) < 50: # or len(sentence)>140:
        return ''
    ents = grep_ent(sentence,True,True,True,True).split()
    if not ents:
        return ''
    s = grep_ent(sentence,False,False,False,False)
    if len(s) < 50:
        return ''
    count = 0
    for e in ents:
        e = e.lower()
        if e in ent_text:            
            indices.append(ent_ind[e])
            ent_text[e] += ' '
            ent_text[e] += s
            count += 1
    if count:      
        last = indptr[-1]
        indptr.append(last+count)
        return s
    else:
        return ''

class Sentence:
    def __init__(self,raw_text,created_at='',tokens_ent='',title=''):
        self.raw_text=raw_text
        self.created_at = created_at
        self.tokens_ent = tokens_ent
        self.title = title
class SentenceOld:
    def __init__(self,raw_text,created_at='',tokens_ent='',title=''):
        self.raw_text=raw_text
        self.created_at = created_at
        self.tokens_ent = tokens_ent
        self.title = title
def getNewsContext(newsObj,ent_ind,ents,vocab,window):          
    ent_text = {}
    for e in ent_ind:
        ent_text[e] = ''

    sentencesIn = []            
    sentencesInObj= []            
    entsIn = []

    # binary matrix
    
    indices = []
    indptr = [0]
    for news in newsObj:
        h_ent = news.h_ent
        s = makeEntText(h_ent,ent_text,ent_ind,indices,indptr,window)
        if s:
            sentencesIn.append( s )
            sentencesInObj.append(Sentence(s,news.created_at,h_ent,news.title))
        b_ent = news.b_ent
        for sentence in sent_detector.tokenize(b_ent.strip()):
            s = makeEntText(sentence,ent_text,ent_ind,indices,indptr,window)
            if s:
                sentencesIn.append( s )
                sentencesInObj.append(Sentence(s,news.created_at,sentence,news.title))
    newsVectorizer = TfidfVectorizer(stop_words='english',vocabulary=vocab,#use_idf=False,
        tokenizer=lambda text: news_tokenizer(text,'reg'))
    XN = newsVectorizer.fit_transform(sentencesIn) #

    for e in ents:
        entsIn.append(ent_text[e])
    XEn = newsVectorizer.fit_transform(entsIn)    

    NEb = csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(sentencesIn),len(ents) ))
    return XN,XEn,NEb,sentencesIn,sentencesInObj,ent_text

def getTweetContext(tweetsObj,ent_ind,ents,vocab,window):          
    ent_text = {}
    for e in ent_ind:
        ent_text[e] = ''

    t0 = time()
    tweetsIn = []            
    tweetsInObj = []            
    entsIn = []
    indices = []
    indptr = [0]
    for i in tweetsObj:
        tweet = tweetsObj[i]
        tokens_ent = tweet.tokens_ent
        t = makeEntText(tokens_ent,ent_text,ent_ind,indices,indptr,window)
        if t:
            tweetsIn.append( t )
            tweetsInObj.append( tweet )

    print( "append in "+str(time() - t0))
    t0 = time()
    tweetVectorizer = TfidfVectorizer(stop_words='english',vocabulary=vocab,#use_idf=False,
        tokenizer=lambda text: tweet_tokenizer(text,'reg'))
    XT = tweetVectorizer.fit_transform(tweetsIn) 
    print( "vectorize in "+str(time() - t0))
    t0 = time()
    for e in ents:
        entsIn.append(ent_text[e])
    XEt = tweetVectorizer.fit_transform(entsIn)    
    print( "ents append + vec in "+str(time() - t0))

    TEb = csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(tweetsIn),len(ents) ))
    return XT,XEt,TEb,tweetsIn,tweetsInObj,ent_text
def softmax(X):
    indices,indptr,data=X.indices,X.indptr,X.data
    for i in range(len(indptr)-1):
        if indptr[i] == indptr[i+1]:  # To Do: deal with sentences that do not linkt to any ent
            continue            
        rowdata = data[indptr[i]:indptr[i+1]]
        rowdata = np.exp(rowdata-max(rowdata))
        data[indptr[i]:indptr[i+1]] = rowdata
def normBypartite_exp(X):
    Xexp = X.copy()
    softmax(Xexp)
    Xexp2 = Xexp.tocsc().T
    softmax(Xexp2)
    Xexp2 = Xexp2.T

    Xexp.data -= max(Xexp.data)
    Xexp = Xexp.expm1()
    Xexp.data+=1
    
    res1 = normalize(Xexp,axis=1,norm='l1')
    res2 = normalize(Xexp2,axis=0,norm='l1').T
    return res1,res2
def normBypartite(X):
    res1 = normalize(X,axis=1,norm='l1')
    res2 = normalize(X,axis=0,norm='l1').T
    return res1,res2

def printSummary(nScore,tScore,sentences,sentencesObj,tweets,tweetsObj,K):
    nind = nScore.argsort()[::-1][:K]
    tind = tScore.argsort()[::-1][:K]
    for i in nind:
        sobj = sentencesObj[i]
        print i,nScore[i],sobj.created_at,sentences[i],'|',sobj.title
    print "------"
    for i in tind:
        tobj = tweetsObj[i]
        print i,tScore[i],tobj.pop,tobj.created_at,tweets[i],'|',tobj.tokens_ent,tobj.tokens
def maxScore(X,exist,i):
    if not exist:
        return 0
    ind = np.array(exist)
    score = X[ind,:].dot(X[i,:].T)
    if score.nnz==0:
        return -1
    else:
        return max(score.data)
    
def printSummaryOne(nScore,sentences,sentencesObj,X,K,b=0.3,debug=0,outfile=None):    
    nind = nScore.argsort()[::-1]
    exist = []
    count = 0
    for i in nind:
        if maxScore(X,exist,i) > b or maxScore(X,exist,i) < 0 \
                or len(sentences[i])>250:# or len(sentences[i])<50:

            if debug:
                print 'skip',maxScore(X,exist,i),len(sentences[i])
            continue        
        if debug:
            print maxScore(X,exist,i)
        sobj = sentencesObj[i]
        if debug:
            print i,nScore[i],sobj.pop if hasattr(sobj,'pop') else '',sobj.created_at,sentences[i]
        else:
            if outfile is not None:
                str1 = str(sobj.pop) if hasattr(sobj,'pop') else ''
                str2 = ' '.join(sentences[i].split('http')[:-1]) if sentences[i].endswith('http') else sentences[i]
                out = str1 + ' ' + str2+'\n'
                outfile.write(out)
            else:
                print sobj.pop if hasattr(sobj,'pop') else '',\
                    ' '.join(sentences[i].split('http')[:-1]) if sentences[i].endswith('http') else sentences[i]
        exist.append(i)
        count+=1
        if count == K:
            break
def printRankedEnt(score,ents,K,DISPLAY=False):
    ind = score.argsort()[::-1][:K]
    count = 0
    res = {}
    for i in ind:
        if DISPLAY:
            print i-count,i,score[i],ents[i]
        res[ents[i]] = i-count
        count +=1
    return res    

    
def printDictByValue(score,K,reverse=True):
    count = 0
    for i in sorted(score.items(), key=operator.itemgetter(1),reverse=reverse):
        print i[1],i[0]
        count+=1
        if count == K:
            break
