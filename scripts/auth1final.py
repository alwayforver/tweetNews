import sys,os

from twython import Twython
APP_KEY = 'PoLS6KUqbS7gZE0dtmgeUIPhQ'
APP_SECRET = 'YeBvKvjDHV7bbxjTuCtsomsMBDnAyiq0ueCTrGFZXUASgOMZFj'

tmp = sys.argv[1]
tmp = open(tmp)
OAUTH_TOKEN = tmp.readline()[:-1]
OAUTH_TOKEN_SECRET = tmp.readline()[:-1]
tmp.close()
oauth_verifier = "6050648"
twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
final_step = twitter.get_authorized_tokens(oauth_verifier)
OAUTH_TOKEN = final_step['oauth_token']
OAUTH_TOKEN_SECRET = final_step['oauth_token_secret']
ot = sys.argv[2]
file = open(ot,'w')

print >> file, OAUTH_TOKEN
print >> file, OAUTH_TOKEN_SECRET

