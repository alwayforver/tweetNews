import sys,os

from twython import Twython
APP_KEY = 'PoLS6KUqbS7gZE0dtmgeUIPhQ'
APP_SECRET = 'YeBvKvjDHV7bbxjTuCtsomsMBDnAyiq0ueCTrGFZXUASgOMZFj'

twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()
#at = "~/share/twitterAccessToken"
at = sys.argv[1]
file = open(at,'w')
print >>file, ACCESS_TOKEN
