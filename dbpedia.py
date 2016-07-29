import codecs
import os
import requests
import urllib2
import simplejson
import datetime

start = datetime.datetime.now()

text = 'President Obama called Wednesday on Congress to extend a tax break for students included in last year\'s economic stimulus package, arguing that the policy provides more generous assistance.'
text = 'Brazilian state-run giant oil company Petrobras signed a three-year technology and research cooperation agreement with oil service provider Halliburton.'
url = 'http://spotlight.sztaki.hu:2222/rest/annotate'
url = 'http://localhost:2222/rest/annotate'

params = {'confidence': 0.3, 'support': 10, 'text': text}
reqheaders = {'accept': 'application/json'}

response = requests.post(url, data=params, headers=reqheaders)

result = response.json()

if 'Resources' in result:
	for entity in result['Resources']:
		print entity['@surfaceForm'], entity['@types']

print datetime.datetime.now() - start

