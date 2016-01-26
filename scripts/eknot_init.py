import operator
import codecs
import glob
import backendDefs as bk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time

import numpy as np
import sys,os

import pickle
from eknot_utils import calc_Pw_z, inittime
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getwriter('utf-8')(sys.stdin)
DEBUG = 1

def init_all(K,Xs,DT):
    km = KMeans(n_clusters=K, init='k-means++', n_init=10)
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
    if DEBUG==1:
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
    # return Pz_d_km,Pw_z_km, ..., mu_km, sigma_km
    return inits

                    

if __name__ == "__main__":
    # input args: K display outputdir pickle_file
    if os.path.isfile(sys.argv[4]):
        with open(sys.argv[4]) as f:
            [X,Xp,Xl,Xo,X_all,vect,vectp,vectl,vecto,vect_all,DT,ind2obj] = pickle.load(f)
    else:
        print "no pickle"
        exit(-1)
    K=int(sys.argv[1])
    Xs = [X,Xp,Xl,Xo,X_all]
    print "begin initiating... "
    inits = init_all(K,Xs,DT)

    tweetPre=sys.argv[2]
    outfile = sys.argv[3]
    with open(sys.argv[4], 'w') as f:
        pickle.dump(inits,f)
    terms = vect.get_feature_names()
    termsp = vectp.get_feature_names()
    termsl = vectl.get_feature_names()
    termso = vecto.get_feature_names()
    term_all = vect_all.get_feature_names()

    termsList = [terms,termsp,termsl,termso,terms_all]
    wordIndList = []
    numX = 5
    Pw_zs = inits[1:1+numX] # inits
    for dim in range(numX):
        wordIndList.append( Pw_zs[dim].argsort(axis=0)[::-1,:] )
    Pz_d = inits[0]    
    docInd = Pz_d.argsort()[:,::-1]
    t_topK=2000
    for i in range(K):
    #    if (Pz_d[i,:]>lambdaB-0.1).sum()<20:
    #        continue
        printCluster(Xs[0],i,termsList,outfile,ind2obj,t_topK,tweetPre,Pw_zs[0],Pz_d,wordIndList,docInd)
