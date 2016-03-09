import sys,os
import codecs,unidecode
import re,string,requests,json
import nltk
from datetime import datetime as dt
from datetime import timedelta as tdelta
from utils import dbpedia,parseEntity,my_tokenizer,rep1,rep2


sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
tokenizer = 'reg'

def processTextEnt(text):
    #text = filter(lambda x: x in string.printable, text.strip()) # CAVEAT: other languages
    text = rep1.sub(' ', text)
    afterDBpedia = parseEntity(text, dbpedia(text))
    #afterDBpedia = rep2.sub('', afterDBpedia) 
    return afterDBpedia
#    return my_tokenizer(afterDBpedia, tokenizer)

count = 0
for line in sys.stdin:
    if count%10 == 1:
        sys.stderr.write("processing: " + str(count) +'\r'),
    print processTextEnt(line.strip())
    count += 1

sys.stderr.write('\n')
sys.stderr.write("final count = " + str(count) + '\n')
