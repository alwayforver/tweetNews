import sys,os
import codecs
import re
import nltk
from nltk.corpus import stopwords
import string

pattern = r'''[0-9A-Z]+@[0-9A-Z]+|%[A-Z]+|[A-Z]\.(?:[A-Z]\.)+|[\+\-]?\$?\d+(?:[,.:/\-_'@&]\d+)*%?|\w+(?:[/\-_'@&]\w+)*'''
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
stop = set(stopwords.words('english'))
punc = set(string.punctuation)


tokenizer = sys.argv[1]
count = 0
for line in sys.stdin:
    line = filter(lambda x: x in string.printable, line.strip().replace('::::::::',' '))
    print "||||||||||||"+line
    if tokenizer == 'simple':
        tokens_all = nltk.word_tokenize(line)
    elif tokenizer == 'reg':
        tokens_all = nltk.regexp_tokenize(line, pattern)
    else:
        sys.stderr.write("unknown tokenizer\n")
        exit(-1)
    tokens = []
    for token in tokens_all:
        left = ''.join(ch for ch in token if ch not in punc)
        if left and left not in stop and \
                token not in  set(["'s", "'d", "'ve", "'ll", "'m", "'re"]) and token not in stop:
            tokens.append(token)
    if not tokens:
        continue
    print u' '.join(tokens)
    count +=1
    if count%1000 == 1:
        sys.stderr.write("lines generated: "+str(count)+'\r'),
sys.stderr.write('\n')
sys.stderr.write("final count = " + str(count)+'\n')
