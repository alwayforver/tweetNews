# coding=<utf-8>

from whoosh.index import *
from whoosh.fields import *
from whoosh.qparser import QueryParser
import codecs
import os

def CreateIndex(tweets_folder, index_folder):
	index = open_dir(index_folder)
	writer = index.writer()
	
	files = os.listdir('tweets')
	for file in files:
		input = codecs.open('tweets/' + file, encoding = 'utf-8')
		lines = input.readlines()
	
		line_number = 0
		for line in lines:
			info = line.strip().split('\t')
			if len(info) <= 1:
				line_number += 1
				continue
	
			tweet_text = info[1]
			urls = info[3]
			if urls != '':
				urls = urls.split(',')
				for url in urls:
					tweet_text = tweet_text.replace(url, '')
	
			writer.add_document(tweet = tweet_text, path = ('tweets/' + file).decode('utf-8'), line = line_number)
			line_number += 1
	
	writer.commit()


def Search():
	index = open_dir('index')
	searcher = index.searcher()
	query = QueryParser('tweet', index.schema).parse('other')
	results = searcher.search(query, limit = None)
	results = sorted(results, key = lambda k: k['path'])
	print results

if __name__ == '__main__':
	index_folder = 'index'
	tweets_folder = 'tweets/'
	start = '2014-12-20'
	end = '2014-12-20'
	Search()

	if len(sys.argv) > 1:
		start = sys.argv[1]
		end = sys.argv[2]

	if not os.path.exists(index_folder):
		os.makedirs(index_folder)
		schema = Schema(tweet = TEXT, path = ID (stored = True), line = NUMERIC (stored = True))
		index = create_in(index_folder, schema)

	start = datetime.date(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))
	end = datetime.date(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))

	date = start
	while date <= end:
		if os.path.exists(tweets_folder):
			CreateIndex(tweets_folder, index_folder)

		date += datetime.timedelta(days = 1)







