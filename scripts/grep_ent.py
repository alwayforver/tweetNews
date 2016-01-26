import sys,os
import codecs
import re,string
import nltk
from utils import dbpedia,parseEntity,my_tokenizer
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

ent = sys.argv[1]
ent_grep = re.compile(r'(\S+_'+ent+r'_\|)\S+_\|')
surface_grep = re.compile(r'\S+_[a-z]{3,5}_\|(\S+)_\|')
for line in sys.stdin:
    after_ent_selection =  ent_grep.sub(r'\1',line.strip())
    print surface_grep.sub(lambda m: ' '.join(m.group(1).split('_')), after_ent_selection)
