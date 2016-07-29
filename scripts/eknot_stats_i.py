import sys
from eknot_utils import EventNode
from utils import loadPickle
import pickle

if __name__ == "__main__":
    # input args: data_pickle inits/plsa_pickle tweetPre switchtext i outpickle
    [Xs,vects,DT,ind2obj] = loadPickle(sys.argv[1])
    picklename = sys.argv[2]
    tweetPre = sys.argv[3]
    switch = sys.argv[4]
    i = int(sys.argv[5])
    outPickle = sys.argv[6]
    if 'inits_' in picklename:
        [inits,labels,centers] = loadPickle(picklename)
        rootNode = EventNode(Xs,initsDescriptor=inits)
        sys.stderr.write('Printing...\n')
        rootNode.printCluster(vects,ind2obj,tweetPre=tweetPre,switch=switch,fromPlsa=0)
    elif 'plsa_' in picklename:
        rootParams,rootNodeDescriptor = loadPickle(picklename)
        rootNode = EventNode(Xs,params=rootParams,descriptor=rootNodeDescriptor)
        sys.stderr.write('Printing...\n')
        resDocInd,tweetsObj,tweetsObjDedup,tweetsScore = rootNode.printCluster_i(vects,
                ind2obj,i,tweetPre=tweetPre,switch=switch,fromPlsa=1)
        sys.stderr.write('saving pickle...\n')
        with open(outPickle, 'w') as f:
            pickle.dump([resDocInd,tweetsObj,tweetsObjDedup,tweetsScore],f)          

    else:
        sys.stderr.write("wrong plsa/inits pickle name\n")
        exit(-1)

