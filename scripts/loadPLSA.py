from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time

import numpy as np
import sys,os

import pLSA
import pickle
# input args: K display
with open('test30.pickle') as f:
    [X,Xp,Xl,Xo,X_all,K,Learn,Pz_d_km,Pw_z_km,Pw_z,Pz_d,Pd,Li,\
            labels,terms,termsp,termsl,termso,terms_all,DT,ind2obj,clusModel]=pickle.load(f)
if K!=int(sys.argv[1]):
    km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=100,init_size=1000,
            batch_size=1000,verbose=True)
    km.fit(X)
    labels = km.labels_
    centers = km.cluster_centers_
    clus2doc = {}
    for i in range(len(labels)):
        clus2doc[labels[i]] = clus2doc.get(labels[i],set())
        clus2doc[labels[i]].add(i)    
## print number of docs in each cluster 
    for i in clus2doc:
        print (str(i+1)+"\t"+str(len(clus2doc[i])))

t0 = time()
Learn=(1,10)
Pw_z,Pz_d,Pd,Li,Learn = pLSA.pLSA(X,K,Learn,Pz_d_km,Pw_z_km)
print "pLSA done in "+str(time() - t0)

# print topics
display = int(sys.argv[2])
if display == 1:
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
