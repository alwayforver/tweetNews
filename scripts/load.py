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
#k inputmat outmat
mat = sio.loadmat(sys.argv[2])
X = mat['X']
k=int(sys.argv[1])
km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=50,init_size=1000,
        batch_size=1000,verbose=True)
km.fit(X)
labels = km.labels_
centers = km.cluster_centers_
clus2doc = {}
for i in range(len(labels)):
    clus2doc[labels[i]] = clus2doc.get(labels[i],set())
    clus2doc[labels[i]].add(i)    
sio.savemat(sys.argv[3],dict(labels=labels,centers=centers,K=k))
for i in clus2doc:
    print (str(i+1)+"\t"+str(len(clus2doc[i])))
