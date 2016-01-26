import operator
import codecs
import glob
from time import time

import numpy as np
import sys,os

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
    for i in range(K):
        ts = np.array(DT)[labels==i]
        mu[i] = np.mean(ts)
        sigma[i] = np.std(ts)
    return mu,sigma
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
