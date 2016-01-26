from utils import dbpedia, parseEntity
import os,sys

text = sys.argv[1]
print parseEntity(text, dbpedia(text))
