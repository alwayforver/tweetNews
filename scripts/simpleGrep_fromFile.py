import sys,os
words = set()
for line in open(sys.argv[1]):
    words = words|set(line.strip().split())
for line in sys.stdin:
    count = 0
    for w in words:
        if w.lower() in line.lower():
            count += 1
    if count > 0:        
        print str(count)+ '\t' + line.strip()
