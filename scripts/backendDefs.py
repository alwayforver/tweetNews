from datetime import datetime as dt
class News:
    def __init__(self,ID,title,raw_text,snippets,key_word,source,created_at):
        self.ID = ID
        self.title = title
        self.raw_text = raw_text
        self.snippets = snippets
        self.key_word = key_word
        self.source = source
        self.created_at = dt.strptime(created_at[5:-6],"%d %b %Y %H:%M:%S")
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

class Tweet:
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




