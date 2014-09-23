import urllib2
import simplejson

#url = ('https://ajax.googleapis.com/ajax/services/feed/find?' +
#		       'v=1.0&q=Google%20news')
#s = 'http://news.google.com/news?pz=1&cf=all&ned=us&hl=en&topic=h&num=9&output=rss'
s = 'http://news.google.com/?output=rss'
feed = urllib2.quote(s.encode("utf-8"))
url = ('https://ajax.googleapis.com/ajax/services/feed/load?v=1.0&q='+ feed + '&num=9&userip=192.17.236.216')

request = urllib2.Request(url, None)
response = urllib2.urlopen(request)

# Process the JSON string.
results = simplejson.load(response)
# now have some fun with the results...
#for i in results:
#	print i
#	print results[i]
#	print "******************"
print results
print "========================================"
print len(results['responseData']['feed']['entries'])
ent =  results['responseData']['feed']['entries'][0]
for i in ent:
	print i
	print ent[i]
	print "******************"


