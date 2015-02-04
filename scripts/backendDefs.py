from datetime import datetime as dt
import codecs
import os
import requests
import json
def getentities(text):
    url = 'http://localhost:2222/rest/annotate'    #local service
    params = {'confidence': 0.5, 'support': 10, 'text': text.encode('ascii','ignore')}
    reqheaders = {'accept': 'application/json'}
    response = requests.post(url, data=params, headers=reqheaders)
    try:
        results = response.json()
        entities = {}
        if 'Resources' in results:
            for result in results['Resources']:
                entity = result['@surfaceForm']
                types = result['@types']
                if 'DBpedia:Person' in types:
                    entity += ':person'
                elif 'DBpedia:Place' in types:
                    entity += ':place'
                elif 'DBpedia:Organisation' in types:
                    entity += ':org'
                if entity in entities:
                    entities[entity] += 1
                else:
                    entities[entity] = 1
        entities_string = ''
        for (entity, count) in entities.items():
            entities_string += entity + ':' + str(count) + '\t'
            
        return entities_string
    except:
        print text
        print "dbpedia error: "+response.text
        return "dbpedia error"
class News:
    def __init__(self,ID,title,raw_text,snippets,key_word,source,created_at,dtpure):
        self.ID = ID
        self.title = title
        self.raw_text = raw_text
        self.snippets = snippets
        self.key_word = key_word
        self.source = source
        self.created_at = dt.strptime(created_at[5:-6],"%d %b %Y %H:%M:%S")
        self.dtpure = dtpure
    local_time_zone = None
    url = None
    def _tz(self, tz):    
        self.local_time_zone = tz
    def _url(self, l):
        self.url = l
    def t_path(self):
        return (self.ID+"_" + \
            "".join(c for c in self.title if c not in (string.punctuation))).replace(' ', '_') \
            +'-' \
            +("".join(c for c in self.source if c not in (string.punctuation))).replace(' ', '_')
    def relTweets(self,prefixDIR):
        return []
    def entities(self):
        return getentities(self.title) + " || "+ getentities(self.raw_text)

class Tweet:
    def __init__(self,ID,raw_text,created_at,is_retweet,retweet_count,hash_tags):
        self.ID = ID
        self.raw_text = raw_text
        try: 
            self.created_at = dt.strptime(created_at[4:-11] + " " +created_at[-4:],"%b %d %H:%M:%S %Y")
        except:
            print("time error")
            self.created_at = dt.today()
        self.is_retweet = is_retweet
        self.retweet_count = retweet_count
        self.hash_tags = hash_tags
    local_time_zone = None
    newsID = None
    def _tz(self, tz):
        self.local_time_zone = tz
    def _newsID(self, nid):
        self.newsID = nid
    def entities(self):
        return getentities(self.raw_text)


class metaNews:
    def __init__(self,ID,title,raw_text,snippets,key_word,source,created_at):
        self.ID = ID
        self.title = title
        self.raw_text = raw_text
        self.snippets = snippets
        self.key_word = key_word
        self.source = source
        self.created_at = created_at
    local_time_zone = None
    url = None
    def _tz(self, tz):    
        self.local_time_zone = tz
    def _url(self, l):
        self.url = l
    def t_path(self):
        return (self.ID+"_" + \
            "".join(c for c in self.title if c not in (string.punctuation))).replace(' ', '_') \
            +'-' \
            +("".join(c for c in self.source if c not in (string.punctuation))).replace(' ', '_')
    def relTweets(self,prefixDIR):
        return []

class metaTweet:
    def __init__(self,ID,raw_text,created_at,is_retweet,retweet_count,hash_tags):
        self.ID = ID
        self.raw_text = raw_text
        self.created_at = created_at
        self.is_retweet = is_retweet
        self.retweet_count = retweet_count
        self.hash_tags = hash_tags
    local_time_zone = None
    newsID = None
    def _tz(self, tz):
        self.local_time_zone = tz
    def _newsID(self, nid):
        self.newsID = nid




