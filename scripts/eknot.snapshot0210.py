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
from eknot_utils import printNewsCluster,run,EventNode
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getwriter('utf-8')(sys.stdin)


if __name__ == "__main__":
    # input args: K display outputdir pickle_file
    if os.path.isfile(sys.argv[1]):
        with open(sys.argv[1]) as fdata:
            [Xs,vects,DT,ind2obj] = pickle.load(fdata)
    else:
        sys.stderr.write("no pickle\n")
        exit(-1)
    if os.path.isfile(sys.argv[2]):
        with open(sys.argv[2]) as finits:
            [K,inits,labels,centers] = pickle.load(finits)
            # inits: [Pz_d_km,Pw_z_km, ..., mu_km, sigma_km]
            # [Learn,Pz_d_km,Pw_z_km,Pw_z,Pz_d,Pd,Li]=pickle.load(finits)
    else:
        sys.stderr.write("no pickle\n")
        exit(-1)
    tweetPre=sys.argv[3]

   ### move to argv ### 
    eventID=0
    Xinds = [0,1,2,3]
    ww = 1
    wp=1
    wl=1
    wo=1
    ws = [ww,wp,wl,wo]
    selectTime = 1
    wt = 0.5
    lambdaB = 0.5
    Learn=(1,20)
    ### ###

    rootNode = EventNode(Xs,Xinds,ws,lambdaB,inits,selectTime=selectTime,
            DT=DT,wt=wt,eventID=eventID,Learn=Learn)
    EventNode.run()
    #Pw_zs,Pz_d,Pd,mu,sigma,Li = run(Xs,Xinds,ws,lambdaB,Learn,inits,selectTime,wt,DT)
    if len(sys.argv)>4:
        sys.stderr.write('saving pickle...\n')
        with open(sys.argv[4], 'w') as f:
    #        pickle.dump([K,Xinds,ws,wt,lambdaB,Learn,selectTime,Pw_zs,Pz_d,Pd,mu,sigma,Li],f)
            pickle.dump(rootNode.params,f)          
    
    sys.stderr.write('After pickle, printing...\n')
########### print #######
    numX = len(Xinds)
    termsList= []
    for i in range(len(Xinds)):
        termsList.append(vects[Xinds[i]].get_feature_names())    
    wordIndList = []
    for dim in range(numX):
        wordIndList.append( Pw_zs[dim].argsort(axis=0)[::-1,:] )
    docInd = Pz_d.argsort()[:,::-1]
    t_topK=10000
    cosine_d_z = Xs[0].dot(Pw_zs[0]) ####
    for i in range(K):
    #    if (Pz_d[i,:]>lambdaB-0.1).sum()<20:
    #        continue
        printNewsCluster(i,termsList,ind2obj,t_topK,tweetPre,cosine_d_z, Pw_zs[0],Pz_d,wordIndList,docInd)
