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
# import for db
import django
local_base = '/home/jwang112/projects/tweet/demoBasic/tweenews'
sys.path.append(local_base)
#print(sys.path)
os.environ['DJANGO_SETTINGS_MODULE']='tweenews.settings'
django.setup()
from overviews.models import News, Tweet

def getRelTweets(newsID):
    n = News.objects.filter(ID=newsID)
    if n.count() > 0:
        #return News.objects.get(ID=newsID).tweet_set.all()
        return list(n[0].tweet_set.all())
    else:
        return []
def rankTweets(tweets, newsVec, vocab):
    tweetVectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf,vocabulary=vocab)
    lines = []
    re_cts = []
    for tweet in tweets:
        content = tweet.raw_text
        lines.append(content)
        re_ct = tweet.retweet_count
        re_cts.append(re_ct)
    X = tweetVectorizer.fit_transform(lines)
    tweetsInd = X.dot(newsVec).argsort()[::-1][:20]
    topTweets = [lines[i] for i in tweetsInd]
    return topTweets

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
if len(args) != 4:
    op.error("this script takes 3 arguments: news_input num_clus all/text/snippets outfile")
    sys.exit(1)

file = codecs.open(sys.argv[1], encoding = 'utf-8')
dataop = sys.argv[3]
outfile = codecs.open(sys.argv[4], 'w', encoding = 'utf-8')
#data = file.readlines()

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

vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                             min_df=2, stop_words='english',
                             use_idf=opts.use_idf)
print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
X = vectorizer.fit_transform(lines)
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
outfile.write("n_samples: %d, n_features: %d \n" % X.shape)
print()

#if opts.n_components:
#    print("Performing dimensionality reduction using LSA")
#    t0 = time()
#    # Vectorizer results are normalized, which makes KMeans behave as
#    # spherical k-means for better results. Since LSA/SVD results are
#    # not normalized, we have to redo the normalization.
#    svd = TruncatedSVD(opts.n_components)
#    lsa = make_pipeline(svd, Normalizer(copy=False))
#
#    X = lsa.fit_transform(X)
#
#    print("done in %fs" % (time() - t0))
#
#    explained_variance = svd.explained_variance_ratio_.sum()
#    print("Explained variance of the SVD step: {}%".format(
#        int(explained_variance * 100)))
#
#    print()


###############################################################################
# Do the actual clustering
true_k = int(sys.argv[2])
if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=10,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
outfile.write("Clustering sparse data with %s\n" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

labels = km.labels_
docdic = {}
for i in range(len(labels)):
    docdic[labels[i]] = docdic.get(labels[i],set())
    docdic[labels[i]].add(i)    

if not (opts.n_components or opts.use_hashing):
    print("Top terms per cluster:")
    outfile.write("Top terms per cluster:\n")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        outfile.write("Cluster %d:" % i)
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
            outfile.write(' %s' % terms[ind])
        print()
        tweets = []
        for doc in docdic[i]:
            print(lines[doc].split('\t')[1])
            outfile.write(lines[doc].split('\t')[1])
            outfile.write('\n')
            print("-------")
            outfile.write("-------\n")
            newsID = ind2ID[doc]
            tweets = tweets + getRelTweets(newsID)
        if tweets:
            topTweets = rankTweets(tweets, km.cluster_centers_[i,:], vectorizer.vocabulary_)
            print("*******")
            outfile.write("*******top tweets:********\n")
            print("top tweets:")
            for t in topTweets:
                print(t)
                outfile.write(t)
                outfile.write('\n-------\n')
                print("-------")
        else:
            print("no tweets retrieved")
            outfile.write("no tweets retrieved\n")
        print("=========")
        outfile.write("=========\n\n")
        print()
        
