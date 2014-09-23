import os,sys
# python streamTest.py ~/share/twitterOauth
from twython import TwythonStreamer

class MyStreamer(TwythonStreamer):
	def on_success(self, data):
		if 'text' in data:
			#print data
			print data['text'].encode('utf-8')

	def on_error(self, status_code, data):
		print status_code
		# Want to stop trying to get data because of the error?
		# Uncomment the next line!
	        # self.disconnect()

ot = sys.argv[1]
file = open(ot)
OAUTH_TOKEN = file.readline()[:-1]
OAUTH_TOKEN_SECRET = file.readline()[:-1]
file.close()

APP_KEY = 'PoLS6KUqbS7gZE0dtmgeUIPhQ'
APP_SECRET = 'YeBvKvjDHV7bbxjTuCtsomsMBDnAyiq0ueCTrGFZXUASgOMZFj'
stream = MyStreamer(APP_KEY, APP_SECRET,
		                    OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
stream.statuses.filter(track='iwatch')
