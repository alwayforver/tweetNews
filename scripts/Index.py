# coding=<utf-8>

from whoosh.index import *
from whoosh.fields import *
from whoosh.qparser import QueryParser
import codecs
import os

def CreateIndex(tweet_folder, index_folder):
	index = open_dir(index_folder)
	writer = index.writer()
	
	files = os.listdir(tweet_folder)
	for file in files:
		input = open(tweet_folder + file)
		lines = input.readlines()
		print tweet_folder + file
	
		line_number = 0
		for line in lines:
			line = line.decode('utf-8')
			info = line.strip().split('\t')
			if len(info) < 2:
				line_number += 1
				continue

			tweet_text = info[1]
			create_time = info[2]
			create_time = create_time[:19] + ' ' + create_time[-4:]

			retweet_num = info[13].strip()
			if retweet_num == '':
				retweet_num = '0'
				
			writer.add_document(tweet = tweet_text, path = ('tweets/' + file).decode('utf-8'), line = line_number, time = create_time, retweet = retweet_num)

			line_number += 1

	writer.commit()

#To use this function, you need to create an index object first using
#index = whoosh.index.open_dir(index_folder)
#and then pass the index to the function Searcu(), together with a keyword list

#The reason why I don't open an index in the function is for efficiency. I don't want to reload the index again and again.
def Search(index, keywords):
	searcher = index.searcher()
	query = QueryParser('tweet', index.schema).parse(' AND '.join(keywords))
	results = searcher.search(query, limit = None)
	return results

if __name__ == '__main__':
	index_folder = 'index'
	tweet_folder = 'tweets/'
	start = '2014-12-20'
	end = '2015-01-04'

	if len(sys.argv) > 1:
		start = sys.argv[1]
		end = sys.argv[2]

	if not os.path.exists(index_folder):
		os.makedirs(index_folder)
		schema = Schema(tweet = TEXT (stored = True), path = STORED, line = STORED, time = STORED, retweet = STORED)
		index = create_in(index_folder, schema)

	start = datetime.date(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))
	end = datetime.date(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))

	date = start
	while date <= end:
		if os.path.exists(tweet_folder + str(date)):
			CreateIndex(tweet_folder + str(date) + '/', index_folder)

		date += datetime.timedelta(days = 1)







