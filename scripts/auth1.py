import sys,os

from twython import Twython
APP_KEY = 'PoLS6KUqbS7gZE0dtmgeUIPhQ'
APP_SECRET = 'YeBvKvjDHV7bbxjTuCtsomsMBDnAyiq0ueCTrGFZXUASgOMZFj'

twitter = Twython(APP_KEY, APP_SECRET)
auth = twitter.get_authentication_tokens()
OAUTH_TOKEN = auth['oauth_token']
OAUTH_TOKEN_SECRET = auth['oauth_token_secret']


tmp = open('tmp.txt','w')
print >> tmp, OAUTH_TOKEN
print >> tmp, OAUTH_TOKEN_SECRET
print auth['auth_url']
##2736708
#oauth_verifier = "2736708"
#twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
#final_step = twitter.get_authorized_tokens(oauth_verifier)
#OAUTH_TOKEN = final_step['oauth_token']
#OAUTH_TOKEN_SECRET = final_step['oauth_token_secret']
#ot = sys.argv[1]
#file = open(ot,'w')
#
#print >> file, OAUTH_TOKEN
#print >> file, OAUTH_TOKEN_SECRET

