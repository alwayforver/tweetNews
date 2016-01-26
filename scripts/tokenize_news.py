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

def processText(text):
    text = filter(lambda x: x in string.printable, text.strip()) # CAVEAT: other languages 
    text = rep1.sub(' ', text)
    text = rep2.sub('', text)
    return my_tokenizer(text, tokenizer) 
def processTextEnt(text):
    text = filter(lambda x: x in string.printable, text.strip()) # CAVEAT: other languages
    text = rep1.sub(' ', text)
    afterDBpedia = parseEntity(text, dbpedia(text))
    #afterDBpedia = rep2.sub('', afterDBpedia) 
    return afterDBpedia
#    return my_tokenizer(afterDBpedia, tokenizer)
def parsefile(f,inPre,outPre):
    fin = codecs.open(inPre+f, encoding = 'utf-8')
    fout = codecs.open(outPre+f, 'w', encoding = 'utf-8')
    for line in fin:
        if len(line.strip().split("\t")) == 10: #### after dbpedia entities are stored
            ID,url,title,source,created_at,authors,key_word,snippets,raw_text,entities = line.strip().split("\t")
        elif len(line.strip().split("\t")) == 9:
            ID,url,title,source,created_at,authors,key_word,snippets,raw_text = line.strip().split("\t")
        else:
            continue
        # body can be empty
        if raw_text.startswith("The page you've requested either does not exist or is currently unavailable.") or len(raw_text)<len(snippets):
            raw_text=snippets
        title = unidecode.unidecode(title.strip())
        raw_text = unidecode.unidecode(raw_text.strip())
        h_tokens = processText(title)
        b_tokens = processText(raw_text)
        h_tokens_ent = processTextEnt(title)
        b_tokens_ent = processTextEnt(raw_text)

        lineout = '\t'.join([ID,url,title,source,created_at,authors,key_word,snippets,raw_text,\
                h_tokens, b_tokens, \
                h_tokens_ent, b_tokens_ent]) 
        
        fout.write(lineout+'\n')
    fin.close()
    fout.close()

count = 0

newsPre = sys.argv[1]
newsPreOut = sys.argv[2]
s_dt = dt.strptime(sys.argv[3],"%Y-%m-%d")
e_dt = dt.strptime(sys.argv[4],"%Y-%m-%d")

numDays = (e_dt - s_dt).days
for x in range(numDays+1):
    fileDate = s_dt + tdelta(days = x)
    filename = fileDate.strftime("%Y-%m-%d")+'.txt'
    
    if os.path.isfile(newsPre+filename):
        sys.stderr.write("file processing: " + filename +'\r'),
        parsefile(filename,newsPre, newsPreOut)
        count += 1

sys.stderr.write('\n')
sys.stderr.write("final count = " + str(count) + '\n')
