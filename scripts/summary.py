import sys,os
import pickle
from eknot_utils import EventNode,NodeParams
from utils import loadPickle,triHits
from eknot_utils import *
from time import time

def main():
    [resDocInd,tweetsObj,tweetsObjDedup,tweetsScore]= loadPickle(sys.argv[1])
    [Xs,vects,DT,ind2obj] = loadPickle(sys.argv[2])
    rootParams,rootNodeDescriptor = loadPickle(sys.argv[3])

    topK = int(sys.argv[4]) # ent used
    kSummary = int(sys.argv[5]) # summary sentences
    i = int(sys.argv[6])
    window = 5
    

    t0 = time()
    Pw_zs = rootNodeDescriptor.Pw_zs
    Pe_z = Pw_zs[1][:,i]
    evocab = vects[5].get_feature_names()
    vocab = vects[0].get_feature_names()
    
    ent_ind,ents = getEntInd(evocab,Pe_z,topK)  # ents: the order in which EN comes in
    
    print( "entscore in "+str(time() - t0))
    t0 = time()
    
    newsObj = [ind2obj[n] for n in resDocInd]
    XN,XEn,NEb,sentencesIn,sentencesInObj,ent_text_n = getNewsContext(newsObj,ent_ind,ents,vocab,window)
    print( "get news Context in "+str(time() - t0))
    print len(newsObj),len(sentencesIn),len(set(sentencesIn))
    t0 = time()
    
    XT,XEt,TEb,tweetsIn,tweetsInObj,ent_text_t = getTweetContext(tweetsObjDedup,ent_ind,ents,vocab,window)
    print( "get tweet Context in "+str(time() - t0))
    
    print len(tweetsObjDedup),len(tweetsIn),len(set(tweetsIn))
    t0 = time()
    
    newsScore = XN.dot(Pw_zs[0][:,i])
    tweetsScore = XT.dot(Pw_zs[0][:,i])

    
    print( "init score in "+str(time() - t0))
    t0 = time()
    
    NE_ = XN.dot(XEn.T) #.multiply(NEb)
    TE_ = XT.dot(XEt.T) #.multiply(TEb)
    NE,EN = normBypartite(NE_)
    TE,ET = normBypartite(TE_)
    print( "graph constr in "+str(time() - t0))
    t0 = time()
    
    nScore, tScore = triHits(newsScore, tweetsScore, NE, EN, TE, ET, 0.2, 0.2, 5)
    print( "trihits in "+str(time() - t0))
    t0 = time()


    printSummary(newsScore,tweetsScore,sentencesIn,sentencesInObj,tweetsIn,tweetsInObj,kSummary)
    print "*****"
    printSummary(nScore,tScore,sentencesIn,sentencesInObj,tweetsIn,tweetsInObj,kSummary)



if __name__ == "__main__":
    # data_pickle inits_pickle tweetPre out_plsa_pickle
    main()
