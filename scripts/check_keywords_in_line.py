import sys,os

# argv 1 string of keywords
# argv 2 number of matched keywords
# argv 3 debug mode

keywords = set(sys.argv[1].strip().split())
debug = int(sys.argv[3])
#keywords = {}
lineNum = 0
linesInterested = 0
for line in sys.stdin:
    count = 0
    if lineNum%10000==1 and debug==1:
        print lineNum
    for w in keywords:
        if w in line:
            count += 1
    if count>=int(sys.argv[2]):
        print str(count)+'\t'+line.strip()
        linesInterested+=1
    lineNum+=1
print 'total',lineNum
print 'interested',linesInterested
