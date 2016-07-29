import sys,os
import codecs
import requests,json
import nltk
import pickle
from nltk.corpus import stopwords
import re,string
from datetime import datetime as dt
from datetime import timedelta as tdelta
import numpy as np
from sklearn.preprocessing import normalize
#pattern = r'''[0-9A-Z]+@[0-9A-Z]+|%[A-Z]+|[A-Z]\.(?:[A-Z]\.)+|[\+\-]?\$?\d+(?:[,.:/\-_'@&]\d+)*%?|\w+(?:[/\-_'@&]\w+)*'''
pattern = r'''\S+_[a-z]{3,5}_\||[A-Z]\.(?:[A-Z]\.)+|[\+\-]?\$?\d+(?:[,.:/\-_'@&]\d+)*%?|\w+(?:[/\-_'@&]\w+)*''' # first field: special treatment for dbpedia entities
tweet_pattern = r'''@\w+|[A-Z]\.(?:[A-Z]\.)+|[\+\-]?\$?\d+(?:[,.:/\-_'@&]\d+)*%?|\w+(?:[/\-_'@&]\w+)*''' # first field: special treatment for "@gfda_12fs"
stop = set(stopwords.words('english'))|set(['mr','mrs','would','said','spoke','told','ms'])
tweet_stop = stop|set(['rt','http'])
punc = set(string.punctuation)
#MAX_BODY_LEN = 1200
MAX_BODY_LEN = 2000

rep1 = re.compile('|'.join([r"::::::::", r"_\|"]))
rep2 = re.compile('|'.join([r"'s\b", r"'d\b", r"'ve\b", r"'ll\b", r"'m\b", r"'re\b"])) # 't

tokenizer = 'reg'

def processText(text):
    #text = filter(lambda x: x in string.printable, text.strip()) # CAVEAT: other languages 
    text = rep1.sub(' ', text)
    text = rep2.sub('', text)
    return my_tokenizer(text, tokenizer) 
def processTextEnt(text):
    #text = filter(lambda x: x in string.printable, text.strip()) # CAVEAT: other languages
    text = rep1.sub(' ', text)
    afterDBpedia = parseEntity(text, dbpedia(text))
    #afterDBpedia = rep2.sub('', afterDBpedia) 
    return afterDBpedia
#    return my_tokenizer(afterDBpedia, tokenizer)

def dbpedia(text):
    if not text:
        return []
    url = 'http://dmserv2.cs.illinois.edu:2222/rest/annotate'    #local service
    params = {'confidence': 0.35, 'support': 20, 'text': text}
    reqheaders = {'accept': 'application/json'}
    response = requests.post(url, data=params, headers=reqheaders)
    try:
        results = response.json()
    except:
        print "text,",text
        return []
    if 'Resources' in results:
        return results['Resources']
    else:
        return []

def parseEntity(text,resources):
    if not resources:
        return text
    textout = ''
    st = 0
    for resource in resources:
        surface = resource['@surfaceForm']
        if surface is None:  # to do: why dbpedia returns NONE?
            continue
        offset = int(resource['@offset'])
        textout+= text[st:offset]

        if '_' in surface:
            sys.stderr.write('\nsurface name: '+surface+'\n')
        surface = '_'.join(surface.split()) # CAVEAT: may result in slight difference with original text
        types = resource['@types']
        entity = resource['@URI'][28:] # http://dbpedia.org/resource/entity_name-potential_more_grams
        if ':Person' in types:
            entity2insert = ' '+entity + '_per_|'+surface+'_|'
        elif ':Place' in types:
            entity2insert = ' '+entity + '_loc_|'+surface+'_|'
        elif ':Organisation' in types or ':Organization' in types:
            entity2insert = ' '+entity + '_org_|'+surface+'_|'
        else:
            entity2insert = ' '+entity + '_other_|'+surface+'_|'
        
        textout += entity2insert
        st = offset + len(surface)
    textout += text[st:]
    return textout

def break_non_empty_group(m):
    for i in m.groups():
        if i:
            return ' '.join(i.split('_'))

def grep_ent(text,per,loc,org,other): # pull out entities
    if not text:
        return ''
    ent_keep = []
    if per:
        ent_keep.append(r'\S+_per_\|(?=\S+_\|)')
    if loc:
        ent_keep.append(r'\S+_loc_\|(?=\S+_\|)')
    if org:
        ent_keep.append(r'\S+_org_\|(?=\S+_\|)')
    if other:
        ent_keep.append(r'\S+_other_\|(?=\S+_\|)')

    if ent_keep:
        ent_grep = re.compile('|'.join(ent_keep))
        text = ent_grep.findall(text) # after_ent_selection
        text = ' '.join(text)
    else:
        surface_grep = re.compile(r'\S+_[a-z]{3,5}_\|(\S+)_\|')
        #surface_grep = re.compile('|'.join(surface_keep))
        text = surface_grep.sub(lambda m: ' '.join(m.group(1).split('_')), text)
    return text

def grep_ent_with_context(text,per,loc,org,other): # convient way to put entities in the context
    if not text:
        return ''
    ent_keep = []
    surface_keep =[]
    if per:
        ent_keep.append(r'(?<=_per_\|)\S+_\|') # hacky way to remove '\S+' before _per_ to accomodate ?<=
    else:
        surface_keep.append(r'\S+_per_\|(\S+)_\|')
    if loc:
        ent_keep.append(r'(?<=_loc_\|)\S+_\|')
    else:
        surface_keep.append(r'\S+_loc_\|(\S+)_\|')
    if org:
        ent_keep.append(r'(?<=_org_\|)\S+_\|')
    else:
        surface_keep.append(r'\S+_org_\|(\S+)_\|')
    if other:
        ent_keep.append(r'(?<=_other_\|)\S+_\|')
    else:
        surface_keep.append(r'\S+_other_\|(\S+)_\|')

    if ent_keep:
        ent_grep = re.compile('|'.join(ent_keep))
        text = ent_grep.sub('' ,text) # after_ent_selection
    if surface_keep:
        surface_grep = re.compile('|'.join(surface_keep))
        #text = surface_grep.sub(lambda m: ' '.join(m.group(1).split('_')), text)
        text = surface_grep.sub(break_non_empty_group, text)
    return text 

def parseEntityOld(text,resources,per,loc,org,other): # 
    if not resources:
        return text
    textout = ''
    st = 0
    for resource in resources:
        offset = int(resource['@offset'])
        textout+= text[st:offset]

        surface = resource['@surfaceForm']
        types = resource['@types']
        entity = resource['@URI'][28:] # http://dbpedia.org/resource/entity_name-potential_more_grams
        if per and 'DBpedia:Person' in types:
            entity2insert = entity + '_per_|'
        elif loc and 'DBpedia:Place' in types:
            entity2insert = entity + '_loc_|'
        elif org and 'DBpedia:Organisation' in types:
            entity2insert = entity + '_org_|'
        elif other:
            entity2insert = entity + '_other_|'
        else:
            entity2insert = surface
        
        textout += entity2insert
        st = offset + len(surface)
    textout += text[st:]
    return textout

def my_tokenizer(line, tokenizer):
    if not line:
        return ''
    if tokenizer == 'simple':
        tokens_all = nltk.word_tokenize(line)
    elif tokenizer == 'reg':
        tokens_all = nltk.regexp_tokenize(line, pattern)
    else:
        sys.stderr.write("unknown tokenizer\n")
        exit(-1)
    tokens = []
    for token in tokens_all:
        left = ''.join(ch for ch in token if ch not in punc)
        if left and token.lower() not in stop:
            tokens.append(token)
#    if not tokens:
        
    return u' '.join(tokens)

def news_tokenizer(line, tokenizer):
    if not line:
        return ''
    if tokenizer == 'simple':
        tokens_all = nltk.word_tokenize(line)
    elif tokenizer == 'reg':
        tokens_all = nltk.regexp_tokenize(line, pattern)
    else:
        sys.stderr.write("unknown tokenizer\n")
        exit(-1)
    tokens = []
    for token in tokens_all:
        left = ''.join(ch for ch in token if ch not in punc)
        if left and token.lower() not in stop:
            token = rep2.sub('', token.lower())
            tokens.append(token)
    return tokens

def tweet_tokenizer(line, tokenizer):
    if not line:
        return ''
    if tokenizer == 'simple':
        tokens_all = nltk.word_tokenize(line)
    elif tokenizer == 'reg':
        tokens_all = nltk.regexp_tokenize(line, tweet_pattern)
    else:
        sys.stderr.write("unknown tokenizer\n")
        exit(-1)
    tokens = []
    for token in tokens_all:
        if token.startswith('@'):
            continue
        left = ''.join(ch for ch in token if ch not in punc)
        if left and token.lower() not in tweet_stop:
            token = rep2.sub('', token.lower())            
            tokens.append(token)
    return tokens
def loadPickle(filename):
    if os.path.isfile(filename):
        with open(filename) as f:
            return pickle.load(f)
    else:
        sys.stderr.write("no pickle\n")
        exit(-1)

from nltk import ConcordanceIndex

class ConcordanceIndex2(ConcordanceIndex):
    def create_concordance(self, word, token_width=13):
        "Returns a list of contexts for @word with a context <= @token_width"
        half_width = token_width // 2
        contexts = []
        offsets = self.offsets(word)
        if offsets:
            print "Displaying matches total#: ",len(offsets)
            for i in offsets:
                start = i - half_width if i >= half_width else 0
                context = self._tokens[start:i + half_width + 1]
                contexts.append(context)
        return contexts      
def norm2(x):
    d = np.linalg.norm(x)
    if d == 0:
        return 0
    else:
        return x/d
def norm1(x):
    d = sum(x)
    if d == 0:
        return 0
    else:
        return x/d

def triHits(newsScore, tweetScore, NE, EN, TE, ET, wN=0.2, wT=0.2, passes=5, eps=1e-9):
    ninit = norm1(newsScore)
    tinit = norm1(tweetScore)
    nScore = ninit
    tScore = tinit
    for i in range(passes):
#        tScore_ = (1-wT)*tinit + wT*(nScore.dot(NE).dot(ET))
        tScore_ = (1-wT)*tinit + wT*( ET.T.dot(NE.T.dot(nScore)) )
        tScore_ = norm1(tScore_)
        if np.linalg.norm(tScore-tScore_) < eps:
            print "==t converge",i,np.linalg.norm(tScore-tScore_)
            break
        else:
            tScore = tScore_
#        nScore = (1-wN)*ninit + wN*(tScore.dot(TE).dot(EN))
        nScore_ = (1-wN)*ninit + wN*( EN.T.dot(TE.T.dot(tScore)) )
        nScore_ = norm1(nScore_)
        if np.linalg.norm(nScore-nScore_) < eps:
            print "==n converge",i,np.linalg.norm(nScore-nScore_)
            break
        else:
            nScore = nScore_
    return nScore, tScore

def triHitsOld(newsScore, tweetScore, NE, EN, TE, ET, wN=0.2, wT=0.2, passes=5, eps=1e-9):
    ninit = norm1(newsScore)
    tinit = norm1(tweetScore)
    nScore = ninit
    tScore = tinit
    for i in range(passes):
#        tScore_ = (1-wT)*tinit + wT*(nScore.dot(NE).dot(ET))
        tScore_ = (1-wT)*tinit + wT*( ET.T.dot(NE.T.dot(nScore)) )
        tScore_ = norm1(tScore_)
        if np.linalg.norm(tScore-tScore_) < eps:
            print "==t converge",i,np.linalg.norm(tScore-tScore_)
            break
        else:
            tScore = tScore_
#        nScore = (1-wN)*ninit + wN*(tScore.dot(TE).dot(EN))
        nScore_ = (1-wN)*ninit + wN*( EN.T.dot(TE.T.dot(tScore)) )
        nScore_ = norm1(nScore_)
        if np.linalg.norm(nScore-nScore_) < eps:
            print "==n converge",i,np.linalg.norm(nScore-nScore_)
            break
        else:
            nScore = nScore_
    return nScore, tScore
def coHits(newsScore,entScore,NE,EN,wN=0.2,wE=0.2,passes=5,eps=1e-9):
    ninit = norm1(newsScore)
    einit = norm1(entScore)
    nScore = ninit
    eScore = einit
    for i in range(passes):
        eScore_ = (1-wE)*einit + wE*( NE.T.dot(nScore) )
        eScore_ = norm1(eScore_)
        if np.linalg.norm(eScore-eScore_) < eps:
            print "==e converge",i,np.linalg.norm(eScore-eScore_)
            break
        else:
            eScore = eScore_
        nScore_ = (1-wN)*ninit + wN*( EN.T.dot(eScore) )  
        nScore_ = norm1(nScore_)
        if np.linalg.norm(nScore-nScore_) < eps:
            print "==n converge",i,np.linalg.norm(nScore-nScore_)
            break
        else:
            nScore = nScore_
    return nScore, eScore            

def lexRank(newsScore,X,d=0.15,passes=5,eps=1e-9):
    ninit = norm1(newsScore)
    nScore = ninit
    for i in range(passes):
        nScore_ = (1-d)*X.T.dot(nScore) + d*nScore
#        print sum(nScore_),sum(X.data)
#        print np.linalg.norm(nScore-nScore_)
        if np.linalg.norm(nScore-nScore_) < eps:
            print "==n converge",i,np.linalg.norm(nScore-nScore_)
            break
        else:
            nScore = nScore_
    return nScore
def printTopWords(i,X,vocab,K=-1):
    ind = X.getrow(i).indices
    data = X.getrow(i).data
    inddata = data.argsort()[::-1]
    if K== -1:
        print [(vocab[ind[k]],data[k]) for k in inddata]
    else:
        print [(vocab[ind[k]],data[k]) for k in inddata][:K]
