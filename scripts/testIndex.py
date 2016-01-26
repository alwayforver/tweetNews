from whoosh.index import *
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh import scoring
import codecs
import os,sys
def Search(index, keywords, k):
    searcher = index.searcher(weighting=scoring.BM25F)
#    query = QueryParser('tweet', index.schema).parse(' AND '.join(keywords))    
    query = QueryParser('tweet', index.schema).parse(keywords)    
    print query   
    if k == -1:
        results = searcher.search(query, limit = None)
    else:
        results = searcher.search(query, limit = k)
    return results


index = open_dir('/home/wtong8/NewsTwitter/index')
keywords = sys.argv[1]
results = Search(index,keywords,int(sys.argv[2]))
outfile = codecs.open(sys.argv[3], 'w', encoding = 'utf-8')
outall = codecs.open(sys.argv[4], 'w', encoding = 'utf-8')
print results
ct=0
tweets = set()
for r in results: 
    t = r['tweet']
    ts = r['time']
    rt = r['retweet']
    if t in tweets:
        continue
    outall.write(t+"\t"+ts+"\t"+rt+"\n")
    if "http" in t or "RT @" in t:
        continue
    outfile.write(t+"\t"+ts+"\t"+rt+"\n")
    tweets.add(t)
    ct+=1
print ct
#    if ct == 50:
#        break
