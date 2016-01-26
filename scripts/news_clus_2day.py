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
import sys,os
import codecs

def readfile(file,dataop,vocab):
    lines = []
    ind2ID = {}
    count = 0
    for line in file:
        if len(line.strip().split("\t")) != 9:
            continue
        ID,url,title,source,date,authors,keywords,snippets,text = line.strip().split("\t")
        ind2ID[count] = int(ID)
        if dataop == "all":
            lines.append(line)
        if dataop == "text":
            lines.append('\t'.join([url,title,keywords,snippets,text]))
        if dataop == "snippets":
            lines.append('\t'.join([url,title,keywords,snippets]))
        count += 1
    if vocab:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf,vocabulary=vocab)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                min_df=2, stop_words='english',
                use_idf=opts.use_idf)
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    X = vectorizer.fit_transform(lines)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()
    return (X,lines,vectorizer)
def getCluster(X,k,opts):
    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=20,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=20,
                    verbose=opts.verbose)
    
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()
    
    labels = km.labels_
    docdic = {}
    for i in range(len(labels)):
        docdic[labels[i]] = docdic.get(labels[i],set())
        docdic[labels[i]].add(i)    
    return (km,docdic)

def printNewsCluster(i,lines,docdic,km,order_centroids,terms,outfile,opts):
    if not (opts.n_components or opts.use_hashing):
        print("Cluster %d:" % i, end='')
        outfile.write("Cluster %d:" % i)
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
            outfile.write(' %s' % terms[ind])
        print()
        outfile.write('\n')
        for doc in docdic[i]:
            print(lines[doc].split('\t')[1])
            outfile.write(lines[doc].split('\t')[1])
            outfile.write('\n')
            print("-------")
            outfile.write("-------\n")
        print("=========")
        outfile.write("=========\n")
        print()
        outfile.write('\n')
            

def compare2day(X1,X2,k1,k2,lines1,lines2,vectorizer1,vectorizer2,outfile,opts):
    (km1,docdic1) = getCluster(X1,k1,opts)
    (km2,docdic2) = getCluster(X2,k2,opts)
    order_centroids1 = km1.cluster_centers_.argsort()[:, ::-1]
    order_centroids2 = km2.cluster_centers_.argsort()[:, ::-1]
    terms1 = vectorizer1.get_feature_names()
    terms2 = vectorizer2.get_feature_names()

    for i in range(k1):
        closeClus = -1
        maxDist = -1;
        newsVec1 = km1.cluster_centers_[i,:]
        for j in range(k2):
            newsVec2 = km2.cluster_centers_[j,:]
            dist = newsVec1.dot(newsVec2)
            if dist > maxDist:
                closeClus = j
                maxDist = dist
        outfile.write('%s' % maxDist)
        printNewsCluster(i,lines1,docdic1,km1,order_centroids1,terms1,outfile,opts)
        printNewsCluster(closeClus,lines2,docdic2,km2,order_centroids2,terms2,outfile,opts)
        outfile.write('*****************\n')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

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
if len(args) != 6:
    op.error("this script takes 5 arguments: news_input1 news_input2 num_clus1 num_clus2 all/text/snippets outfile")
    sys.exit(1)

file1 = codecs.open(sys.argv[1], encoding = 'utf-8')
file2 = codecs.open(sys.argv[2], encoding = 'utf-8')
k1 = int(sys.argv[3])
k2 = int(sys.argv[4])
dataop = sys.argv[5]

outfile = codecs.open(sys.argv[6], 'w', encoding = 'utf-8')
(X1,lines1,vectorizer1) = readfile(file1,dataop,None)
(X2,lines2,vectorizer2) = readfile(file2,dataop,None)
vocab = set(vectorizer1.vocabulary_.keys()) | set(vectorizer2.vocabulary_.keys())
file1.seek(0)
file2.seek(0)
(X1,lines1,vectorizer1) = readfile(file1,dataop,vocab)
(X2,lines2,vectorizer2) = readfile(file2,dataop,vocab)

compare2day(X1,X2,k1,k2,lines1,lines2,vectorizer1,vectorizer2,outfile,opts)
